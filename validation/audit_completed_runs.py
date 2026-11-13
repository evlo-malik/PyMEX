"""Audit completed PyMEX runs without recalculating or modifying their outputs.

Checks inputs, provenance, finite arrays, complete state counts, MPI agreement,
and degenerate subspaces. The subspace checks are diagnostic analysis, not
replacement eigenvectors, optical output, or a scientific calculation.
"""
import argparse
from contextlib import contextmanager
import gzip
import hashlib
import json
from pathlib import Path
import platform
import shutil
import tempfile

import h5py
from mpi4py import MPI
import numpy as np
import scipy
import yaml


FILES = (
    "H_exciton_tb.hdf5", "Eigval_exciton_tb.hdf5", "Eigvec_exciton_tb.hdf5",
    "Conductivity_exciton_tb.hdf5", "H_k_tb.hdf5", "Grad_H_k_tb.hdf5",
)
TOL = 1e-9
GROUP_TOL_EV = 1e-10


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@contextmanager
def restored_file(folder, name, scratch):
    path = folder / name
    if path.exists():
        yield path
        return
    archive = folder / (name + ".gz")
    expected = json.loads((folder / "status.json").read_text())["file_hashes"][name]
    with tempfile.TemporaryDirectory(prefix="audit-", dir=scratch) as temporary:
        restored = Path(temporary) / name
        with gzip.open(archive, "rb") as source, restored.open("wb") as destination:
            shutil.copyfileobj(source, destination)
        assert digest(restored) == expected, (folder.name, name, "archive hash")
        yield restored


def groups(energies):
    start = 0
    for end in range(1, len(energies)):
        if energies[end] - energies[start] > GROUP_TOL_EV:
            yield start, end
            start = end
    yield start, len(energies)


def diagonal(folder):
    with h5py.File(folder / "H_exciton_tb.hdf5", "r") as source:
        matrix = source["exciton_H"][0]
    n = int(np.prod(matrix.shape[:3]))
    matrix = matrix.reshape(n, n)
    values = matrix.diagonal().copy()
    np.fill_diagonal(matrix, 0)
    assert np.count_nonzero(matrix) == 0, "Independent matrix has off-diagonal entries"
    return values


def degeneracy_audit(reference, candidate, scratch):
    """Compare invariant quantities without aligning or altering native arrays."""
    diagonals = [diagonal(folder) for folder in (reference, candidate)]
    energies, vectors, strengths = [], [], []
    for folder in (reference, candidate):
        with h5py.File(folder / "Eigval_exciton_tb.hdf5", "r") as source:
            energies.append(source["Q0"][:])
        with restored_file(folder, "Eigvec_exciton_tb.hdf5", scratch) as path:
            with h5py.File(path, "r") as source:
                vectors.append(source["Q0"][:])
        with h5py.File(folder / "Conductivity_exciton_tb.hdf5", "r") as source:
            strengths.append(np.abs(source["optical_conductivity"][:]) ** 2)
    boundaries = list(groups(energies[0]))
    assert boundaries == list(groups(energies[1])), "Degenerate groups differ"
    residuals = []
    for d, values, vecs in zip(diagonals, energies, vectors):
        worst = 0.0
        for start in range(0, len(values), 64):
            v = vecs[:, start:start + 64]
            residual = d[:, None] * v - v * values[None, start:start + 64]
            worst = max(worst, float(np.max(np.linalg.norm(residual, axis=0))))
        residuals.append(worst)
    projector_error = 0.0
    orthogonality_error = 0.0
    sums = [[], []]
    for start, end in boundaries:
        a, b = [v[:, start:end] for v in vectors]
        projector_error = max(projector_error, float(
            np.linalg.norm(b - a @ (a.conj().T @ b)) / np.sqrt(end - start)))
        for v in (a, b):
            orthogonality_error = max(orthogonality_error, float(
                np.max(np.abs(v.conj().T @ v - np.eye(end - start)))))
        for i, intensity in enumerate(strengths):
            sums[i].append(intensity[start:end].sum(axis=0))
    x, y = map(np.asarray, sums)
    absolute = float(np.max(np.abs(x - y)))
    relative = absolute / float(np.max(np.abs(x)))
    assert max(residuals + [projector_error, orthogonality_error, relative]) < TOL
    return dict(analysis_only=True, energy_group_tolerance_eV=GROUP_TOL_EV,
                groups=len(boundaries), largest_group=max(b - a for a, b in boundaries),
                eigenpair_residual_norm_eV=residuals,
                subspace_projection_error=projector_error,
                within_group_orthogonality_error=orthogonality_error,
                grouped_optical_strength_max_absolute_difference=absolute,
                grouped_optical_strength_max_difference_relative_to_peak=relative)


def audit(record, root, scratch):
    candidate = scratch / record["case"]
    reference = root / record["reference"]
    if not reference.exists():
        reference = scratch / record["reference"]
    original_config = yaml.safe_load((reference / "pymex_tb.yaml").read_text())
    config = yaml.safe_load((candidate / "pymex_tb.yaml").read_text())
    original_config["bse"]["tb"]["model"] = "model.py"
    assert config == original_config, (record["case"], "input mismatch")
    assert digest(reference / "model.py") == digest(candidate / "model.py")
    assert config["diagonalize"] == {"library": "lapack"}, "Full LAPACK required"
    include = config["eh_interaction"]["include"]
    expected_flags = {"independent": (False, False), "direct": (True, False),
                      "direct-exchange": (True, True)}[record["mode"]]
    assert (include["direct"], include["exchange"]) == expected_flags
    assert config["bse"]["tb"]["kgrid"] == [record["mesh"], record["mesh"], 1]
    qpoints = config["bse"].get("qpoints", config["bse"].get("Qpoints"))
    assert qpoints == {"hspoints": [[0.0, 0.0, 0.0]], "num_points": 1}
    for name, expected in {**record["input_hashes"], **record["native_output_hashes"]}.items():
        assert digest(candidate / name) == expected, (record["case"], name, "candidate hash")
    bands = config["bse"]["tb"]["bands"]
    nc = bands["cb_idx"][1] - bands["cb_idx"][0]
    nv = bands["vb_idx"][1] - bands["vb_idx"][0]
    nk = record["mesh"] ** 2
    states = nc * nv * nk
    assert states == nk * (1 if record["material"] == "hBN" else 2)
    status = reference / "status.json"
    saved_hashes = json.loads(status.read_text())["file_hashes"] if status.exists() else {}
    hashes, errors = {}, {}
    for name in FILES:
        with restored_file(reference, name, scratch) as path:
            hashes[name] = digest(path)
            if name in saved_hashes:
                assert hashes[name] == saved_hashes[name], (record["case"], name, "reference hash")
            with h5py.File(path, "r") as a, h5py.File(candidate / name, "r") as b:
                assert set(a) == set(b), (name, "dataset names")
                for key in a:
                    x, y = a[key], b[key]
                    assert x.shape == y.shape and x.dtype == y.dtype
                    if name == "H_exciton_tb.hdf5":
                        assert x.shape == (1, nc, nv, nk, nc, nv, nk)
                    elif name == "Eigval_exciton_tb.hdf5":
                        assert x.shape == (states,)
                        assert np.all(np.diff(x[:]) >= 0) and np.all(np.diff(y[:]) >= 0)
                    elif name == "Eigvec_exciton_tb.hdf5":
                        assert x.shape == (states, states)
                    elif name == "Conductivity_exciton_tb.hdf5":
                        assert x.shape == (states, 3)
                    largest = 0.0
                    for start in range(0, x.shape[0], 64):
                        xx, yy = x[start:start + 64], y[start:start + 64]
                        assert np.isfinite(xx).all() and np.isfinite(yy).all()
                        largest = max(largest, float(np.max(np.abs(xx - yy))))
                    errors[name + "/" + key] = largest
    for name in ("pymex_tb.yaml", "model.py", "sigma_unpolarized_tb.txt"):
        hashes[name] = digest(reference / name)
        if name in saved_hashes:
            assert hashes[name] == saved_hashes[name], (record["case"], name, "reference hash")
    x = np.loadtxt(reference / "sigma_unpolarized_tb.txt")
    y = np.loadtxt(candidate / "sigma_unpolarized_tb.txt")
    assert x.shape == y.shape and x.shape[1] == 4
    assert np.isfinite(x).all() and np.isfinite(y).all()
    assert np.all(np.diff(x[:, 0]) > 0) and np.all(np.diff(y[:, 0]) > 0)
    errors["photon_energy_axis_max_difference_eV"] = float(np.max(np.abs(x[:, 0] - y[:, 0])))
    errors["native_spectrum_max_difference_relative_to_peak"] = float(
        np.max(np.abs(x[:, 1:] - y[:, 1:])) / np.max(np.abs(x[:, 1:])))
    cross_platform = record["reference"].startswith("mac-archive/")
    if cross_platform:
        assert all(v < TOL for k, v in errors.items()
                   if not k.startswith(("Eigvec_exciton", "Conductivity_exciton")))
    else:
        assert all(v == 0 for v in errors.values()), (record["case"], errors)
    result = dict(case=record["case"], passed=True, full_state_count=states,
                  physical_inputs_identical=True, model_hash_identical=True,
                  all_arrays_finite=True, candidate_hashes_verified=True,
                  reference_hashes=hashes,
                  reference_hashes_checked_against_status=bool(saved_hashes),
                  cross_platform_reference=cross_platform, errors=errors)
    if cross_platform:
        result["degeneracy_diagnostics"] = degeneracy_audit(reference, candidate, scratch)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    args = parser.parse_args()
    root, scratch = args.project_root, args.scratch
    test = root / "independent-shortcut-validation"
    summaries = [json.loads((test / name).read_text())
                 for name in ("summary.json", "parent-benchmark.json")]
    records = [record for summary in summaries for record in summary["results"]]
    assert len(records) == 46 and len({r["case"] for r in records}) == 46
    original_hashes = json.loads((root / "mpi-validation-20261113/production-source-hashes.json").read_text())
    current_hashes = {p.name: digest(p) for p in (root / "code/PyMEX/src").glob("*.py")}
    assert current_hashes == original_hashes, "Production source changed"
    optimized_hashes = {p.name: digest(p) for p in (test / "src").glob("*.py")}
    assert optimized_hashes == summaries[0]["source_hashes"]
    results = []
    for record in records:
        results.append(audit(record, root, scratch))
        print("AUDITED", record["case"], flush=True)
    # Directly compare serial and MPI result hashes; both also matched references.
    by_case = {r["case"]: r for r in records}
    mpi_pairs = []
    for name, record in by_case.items():
        if name.endswith("-p4"):
            serial = name[:-1] + "1"
            assert record["native_output_hashes"] == by_case[serial]["native_output_hashes"]
            mpi_pairs.append([serial, name])
    assert current_hashes == {p.name: digest(p) for p in (root / "code/PyMEX/src").glob("*.py")}
    extensions = list((root / "code/.build/pymex").glob("cyfunc*.so"))
    assert len(extensions) == 1
    result = dict(audited_native_runs=len(results), all_passed=True,
                  acceptance=dict(same_platform="exact elementwise agreement",
                                  cross_platform_absolute_or_peak_relative=TOL,
                                  degeneracy_group_tolerance_eV=GROUP_TOL_EV),
                  production_source_hashes=current_hashes, production_sources_unchanged=True,
                  optimized_source_hashes=optimized_hashes,
                  native_extension=dict(filename=extensions[0].name, sha256=digest(extensions[0])),
                  environment=dict(python=platform.python_version(), system=platform.system(),
                                   architecture=platform.machine(), numpy=np.__version__,
                                   scipy=scipy.__version__, h5py=h5py.__version__,
                                   hdf5=h5py.version.hdf5_version, yaml=yaml.__version__,
                                   mpi=MPI.Get_library_version().splitlines()[0]),
                  script_hashes={name: digest(test / name) for name in
                                 ("driver.py", "test_independent_shortcut.py",
                                  "benchmark_parent.py", "audit_completed_runs.py")},
                  mpi_pairs_with_identical_output_hashes=mpi_pairs,
                  results=results)
    (test / "audit.json").write_text(json.dumps(result, indent=2))
    print("AUDIT PASSED", len(results), flush=True)


if __name__ == "__main__":
    main()
