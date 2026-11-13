"""Load and check the inputs to the Python tight-binding reader."""
from copy import deepcopy
import importlib.util
import sys

import numpy as np


def uses_python_model(config):
    """Select this reader when the YAML requests the Python tight-binding engine."""
    bse = config.get("bse", {})
    return bse.get("method") == "tb" and bse.get("tb", {}).get("engine") == "python"


def prepare_python_config(config):
    """Expand one band selection into the two views used by the existing solver."""
    if not uses_python_model(config):
        return
    tb = config["bse"]["tb"]
    # Input files and mesh: read the Python model on a 2D grid.
    if "wannier_io" in config["bse"]:
        raise ValueError("engine: python uses tb.model; remove the wannier_io block.")
    if not isinstance(tb.get("model"), str):
        raise ValueError("Set bse.tb.model to a Python model filename.")
    grid = tb.get("kgrid")
    if (not isinstance(grid, list) or len(grid) != 3
            or any(type(n) is not int or n < 1 for n in grid) or grid[2] != 1):
        raise ValueError("Python input requires a positive 2D kgrid, e.g. [6, 6, 1].")
    # Supported calculation: a single monolayer, Q = 0 and a fresh CPU run.
    if config.get("material") != "2d":
        raise ValueError("Python input currently supports material: 2D only.")
    q = config["bse"].get("qpoints", {})
    if q.get("hspoints") != [[0.0, 0.0, 0.0]] or q.get("num_points") != 1:
        raise ValueError("Python input currently supports one Q = [0, 0, 0] only.")
    if config.get("restart") != "from_scratch":
        raise ValueError("Use restart: from_scratch to avoid reusing a different Python model's data.")
    if config.get("system") != "cpu" or config.get("diagonalize", {}).get("library") != "lapack":
        raise ValueError("Python input is currently validated with CPU/LAPACK only.")
    if config.get("absorption", {}).get("spin") != "unpolarized":
        raise ValueError("Python input currently supports spinless models (spin: unpolarized).")
    # Electron–hole potential: check the chosen interaction settings.
    interaction = config.get("eh_interaction", {})
    real = interaction.get("real", {})
    if (interaction.get("space") != "real" or real.get("potential") != "keldysh"
            or real.get("layer") != "monolayer" or real.get("type") != "analytic"):
        raise ValueError("Python input currently supports real-space analytic monolayer Keldysh interactions.")
    include = interaction.get("include", {})
    if include.get("exchange") and include.get("screen_exchange") and not include.get("direct"):
        raise ValueError("Screened exchange in this solver requires direct: true.")
    if "gamma_k_shift" in tb or tb.get("electric_field_z") is not None:
        raise ValueError("Band shifts and external fields are not supported by the Python prototype.")
    # Band selection: apply one set of indices to both energies and coefficients.
    bands = tb.get("bands")
    if not isinstance(bands, dict):
        raise ValueError("Set tb.bands.vb_idx and tb.bands.cb_idx (end index excluded).")
    bands.setdefault("engine", "python")
    bands.setdefault("vb_skip", 0)
    bands.setdefault("cb_skip", 0)
    if bands["engine"] != "python":
        raise ValueError("tb.bands.engine must agree with tb.engine: python.")
    # The solver slices energies with "bands" and eigenvectors with "cnmk".
    if "cnmk" in tb:
        for key in ("vb_idx", "cb_idx", "vb_skip", "cb_skip"):
            default = 0 if key.endswith("_skip") else None
            if tb["cnmk"].get(key, default) != bands.get(key):
                raise ValueError("bands and cnmk must select the same bands for Python input.")
    tb["cnmk"] = deepcopy(bands)


def validate_band_windows(bands, norb):
    """Check that the valence and conduction ranges fit the model and do not overlap."""
    windows = []
    for name in ("vb", "cb"):
        bounds = bands.get(name + "_idx")
        skip = bands.get(name + "_skip", 0)
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise ValueError(f"{name}_idx must contain a start and an exclusive end index.")
        if any(type(index) is not int for index in (*bounds, skip)):
            raise ValueError(f"{name}_idx and {name}_skip must use integer indices.")
        start, end = bounds
        start += skip
        end += skip
        if not 0 <= start < end <= norb:
            raise ValueError(f"{name} band window is outside the {norb}-orbital model.")
        windows.append((start, end))
    if windows[0][1] > windows[1][0]:
        raise ValueError("The valence band window must end before the conduction bands begin.")


def read_geometry(model, geometry):
    """Read lattice rows and orbital centres in angstroms from YAML or model.py."""
    settings = vars(model) if geometry is None else geometry
    for key in ("lattice", "centres"):
        if key not in settings:
            raise ValueError(f"Please define {key} in tb.geometry or the Python model.")
    _check_units_and_basis(vars(model))
    if geometry is not None:
        _check_units_and_basis(geometry)

    # Geometry: three independent lattice vectors and one position per orbital.
    lattice = np.asarray(settings["lattice"], dtype=float)
    centres = np.asarray(settings["centres"], dtype=float)
    if lattice.shape != (3, 3) or not np.isfinite(lattice).all():
        raise ValueError("lattice must have three finite Cartesian row vectors.")
    if abs(np.linalg.det(lattice)) < 1e-10:
        raise ValueError("lattice vectors must be independent.")
    if centres.ndim != 2 or centres.shape[1] != 3 or len(centres) < 2:
        raise ValueError("centres must have an [x, y, z] row for each orbital in Hamiltonian order.")
    if not np.isfinite(centres).all():
        raise ValueError("Orbital centres must have finite coordinates.")
    # This reader treats a single sheet lying in the xy plane.
    in_plane = np.allclose(lattice[:2, 2], 0, atol=1e-10)
    perpendicular = np.allclose(lattice[2, :2], 0, atol=1e-10)
    if not (in_plane and perpendicular):
        raise ValueError("Use in-plane lattice vectors in xy and a perpendicular third vector.")
    if not np.allclose(centres[:, 2], centres[0, 2], atol=1e-10):
        raise ValueError("Orbital centres must lie in the same xy plane.")
    return lattice, centres


def _check_units_and_basis(settings):
    """The solver assumes orthonormal orbitals, energies in eV and lengths in Å."""
    if settings.get("basis", "orthonormal") != "orthonormal" or "overlap" in settings:
        raise ValueError("Only orthonormal localized orbitals are supported.")
    energy_unit = str(settings.get("energy_unit", "eV")).lower()
    length_unit = settings.get("length_unit", "angstrom")
    if energy_unit != "ev" or length_unit != "angstrom":
        raise ValueError("Use eV and angstrom; k is in inverse angstrom.")


def load_model(path):
    """Load the selected model file so the solver can call its hamiltonian(k)."""
    spec = importlib.util.spec_from_file_location("_pymex_model", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load Python model: {path}")
    model = importlib.util.module_from_spec(spec)
    # Models may import a parameter file from their own directory.
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(model)
    finally:
        sys.path.pop(0)
    if not callable(getattr(model, "hamiltonian", None)):
        raise ValueError("Python model must define hamiltonian(k).")
    return model


def check_model(reader):
    """At a few sample points, check periodicity in x/y and independence of kz."""
    reciprocal = reader.get_reciprocal()
    for fractional_k in ([0., 0., 0.], [.137, .293, 0.], [.371, .117, 0.]):
        k = fractional_k @ reciprocal
        matrix = reader.hamiltonian(k)
        for vector in reciprocal[:2]:
            if not np.allclose(reader.hamiltonian(k + vector), matrix, atol=1e-8, rtol=1e-8):
                raise ValueError("hamiltonian(k) must repeat after a reciprocal lattice translation. Use cell translations in its phases and Cartesian k in inverse angstroms.")
        if not np.allclose(reader.hamiltonian(k + [0, 0, .137]), matrix, atol=1e-10, rtol=1e-10):
            raise ValueError("The 2D model must not depend on kz.")

    # When the user supplies derivatives, compare them with numerical differences.
    if hasattr(reader.model, "gradient"):
        for fractional_k in ([0., 0., 0.], [.137, .293, 0.]):
            k = fractional_k @ reciprocal
            _, gradients = reader.get_Hk_and_grad_Hk(kpt=k)
            supplied = gradients[:, :, 0, :]
            if not np.allclose(supplied, reader._numerical_gradient(k), atol=1e-6, rtol=1e-6):
                raise ValueError("gradient(k) disagrees with finite differences. Check the units and axis order.")


def check_matrix(matrix, norb, k):
    """Check the size, entries and Hermiticity of the matrix returned by the model."""
    if matrix.shape != (norb, norb) or not np.isfinite(matrix).all():
        raise ValueError(f"hamiltonian(k) must return a finite {norb} x {norb} matrix.")
    if not np.allclose(matrix, matrix.conj().T, atol=1e-10, rtol=1e-10):
        raise ValueError(f"hamiltonian(k) is not Hermitian at k={k}.")
