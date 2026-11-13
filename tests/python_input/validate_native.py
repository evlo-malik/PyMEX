"""Compare this input-only branch with saved native independent PyMEX runs.

Uses unchanged saved inputs for hBN and MoS2 at 6x6 and 9x9, with 1 and 4
MPI processes. Reference-root contains material/independent/Nxxx/validation-p1.
Raw outputs stay in a new work directory; only a compact report is published.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import h5py
import numpy as np
import yaml


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "reference-root", "extension-dir", "work-dir", "output", "mpiexec"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=False)
    source = args.source_root / "src"
    source_hashes = {p.name: digest(p) for p in source.glob("*.py")}
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", OMP_THREAD_LIMIT="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH=os.pathsep.join((str(source), str(args.extension_dir))))
    driver = args.source_root / "Examples/python_hamiltonian/calc.py"
    outputs = ("H_exciton_tb.hdf5", "Eigval_exciton_tb.hdf5", "Eigvec_exciton_tb.hdf5",
               "Conductivity_exciton_tb.hdf5", "H_k_tb.hdf5", "Grad_H_k_tb.hdf5",
               "Eigval_k_tb.hdf5", "Eigvec_k_tb.hdf5")
    records = []
    for material, orbitals, nc in (("hBN", 2, 1), ("MoS2", 3, 2)):
        for mesh in (6, 9):
            reference = args.reference_root / material / "independent" / f"N{mesh:03d}/validation-p1"
            for ranks in (1, 4):
                name = f"{material}-N{mesh:03d}-p{ranks}"
                folder = args.work_dir / name
                folder.mkdir()
                config = yaml.safe_load((reference / "pymex_tb.yaml").read_text())
                config["bse"]["tb"]["model"] = "model.py"
                assert config["diagonalize"] == {"library": "lapack"}
                assert not config["eh_interaction"]["include"]["direct"]
                assert not config["eh_interaction"]["include"]["exchange"]
                (folder / "pymex_tb.yaml").write_text(yaml.safe_dump(config))
                shutil.copy2(reference / "model.py", folder / "model.py")
                print("RUN", name, flush=True)
                with (folder / "run.log").open("w") as log:
                    subprocess.run(["nice", "-n", "10", str(args.mpiexec), "-n", str(ranks),
                                    sys.executable, str(driver)], cwd=folder, env=env,
                                   stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300)
                errors = {}
                for filename in outputs:
                    with h5py.File(reference / filename, "r") as a, h5py.File(folder / filename, "r") as b:
                        assert set(a) == set(b)
                        for key in a:
                            x, y = a[key][:], b[key][:]
                            assert x.shape == y.shape and x.dtype == y.dtype
                            assert np.isfinite(x).all() and np.isfinite(y).all()
                            np.testing.assert_array_equal(x, y, err_msg=name + ":" + filename)
                            errors[filename + "/" + key] = float(np.max(np.abs(x - y)))
                        if filename == "Eigval_exciton_tb.hdf5":
                            assert b["Q0"].shape == (nc * mesh * mesh,)
                        if filename == "Eigvec_exciton_tb.hdf5":
                            assert b["Q0"].shape == (nc * mesh * mesh,) * 2
                x = np.loadtxt(reference / "sigma_unpolarized_tb.txt")
                y = np.loadtxt(folder / "sigma_unpolarized_tb.txt")
                assert np.isfinite(x).all() and np.isfinite(y).all()
                np.testing.assert_array_equal(x, y)
                errors["native_TXT_including_energy_axis"] = float(np.max(np.abs(x - y)))
                with np.load(folder / "model_geometry.npz") as geometry:
                    assert geometry["lattice"].shape == (3, 3)
                    assert geometry["centres"].shape == (orbitals, 3)
                    np.testing.assert_array_equal(geometry["kgrid"], [mesh, mesh, 1])
                files = (*outputs, "sigma_unpolarized_tb.txt", "pymex_tb.yaml", "model.py")
                record = dict(case=name, material=material, mesh=mesh, ranks=ranks,
                              full_eigenstate_count=nc * mesh * mesh, passed=True,
                              reference=str(reference.relative_to(args.reference_root)),
                              input_changes="Only model file location normalized to model.py",
                              source_sha256=source_hashes["bse.py"],
                              reference_hashes={f: digest(reference / f) for f in files},
                              candidate_hashes={f: digest(folder / f) for f in files}, errors=errors)
                records.append(record)
                print("PASS", name, flush=True)
    assert source_hashes == {p.name: digest(p) for p in source.glob("*.py")}
    extensions = list(args.extension_dir.glob("cyfunc*.so"))
    assert len(extensions) == 1
    report = dict(actual_native_runs=len(records), all_passed=True,
                  description="Input adapter only; upstream BSE construction, full LAPACK and native optics retained.",
                  environment=dict(python=platform.python_version(), numpy=np.__version__, h5py=h5py.__version__),
                  source_hashes=source_hashes,
                  extension_hash=digest(extensions[0]),
                  driver_sha256=digest(driver), validation_script_sha256=digest(Path(__file__)),
                  results=records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
