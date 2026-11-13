"""Read the periodic matrix H(k) from model.py and supply it to PyMEX."""
from pathlib import Path

import numpy as np

from python_tb_config import check_matrix, check_model, load_model, read_geometry, validate_band_windows
# Keep the configuration functions available to older imports from this module.
from python_tb_config import prepare_python_config, uses_python_model


class PythonTB:
    """Provide the same matrix and geometry requests that the PyMEX solver uses."""

    def __init__(self, filename, kgrid, derivative_step=1e-5, geometry=None):
        # Load the Python file and the lattice and orbital positions (in Å).
        self.path = Path(filename).resolve()
        self.model = load_model(self.path)
        self.lattice, self.centres = read_geometry(self.model, geometry)
        self.norb = len(self.centres)

        # Store the mesh and the small k step used to calculate derivatives.
        self.kgrid = np.asarray(kgrid, dtype=int)
        self.step = float(derivative_step)
        if not np.isfinite(self.step) or self.step <= 0:
            raise ValueError("derivative_step must be positive and finite (inverse angstrom).")
        check_model(self)

    def hamiltonian(self, k):
        """Ask model.py for H(k) in eV. Its rows and columns are orbitals."""
        matrix = np.asarray(self.model.hamiltonian(k), dtype=complex)
        check_matrix(matrix, self.norb, k)
        return matrix

    def _numerical_gradient(self, k):
        """Differentiate every matrix entry with respect to kx and ky (eV Å)."""
        # The last index is x, y or z. The z derivative is zero for a 2D sheet.
        gradient = np.zeros((self.norb, self.norb, 3), dtype=complex)
        for axis in (0, 1):
            # Move in just one direction. Sample H on both sides of k.
            dk = np.zeros(3)
            dk[axis] = self.step
            one_step = self.hamiltonian(k + dk) - self.hamiltonian(k - dk)
            two_steps = self.hamiltonian(k + 2 * dk) - self.hamiltonian(k - 2 * dk)

            # Five-point derivative: matrix subtraction acts on all entries at once.
            gradient[:, :, axis] = (8 * one_step - two_steps) / (12 * self.step)
        return gradient

    def get_Hk_and_grad_Hk(self, method="tb", kpt=None, Ez=None, map_wf=None):
        """PyMEX calls this once per mesh point to get H and its derivatives."""
        # This solver interface passes one Cartesian k point and no vertical field.
        if method != "tb" or Ez is not None:
            raise ValueError("PythonTB supports the field-free tb path only.")
        k = np.asarray(kpt, dtype=float)
        if k.shape != (3,) or not np.isfinite(k).all():
            raise ValueError("kpt must be one finite Cartesian vector of length 3.")

        # Get H directly. Use the model's derivative if supplied, otherwise calculate it.
        matrix = self.hamiltonian(k)
        if hasattr(self.model, "gradient"):
            gradient = np.asarray(self.model.gradient(k), dtype=complex)
        else:
            gradient = self._numerical_gradient(k)

        # Each direction must contain a finite Hermitian matrix, with zero kz derivative.
        if gradient.shape != (self.norb, self.norb, 3) or not np.isfinite(gradient).all():
            raise ValueError("gradient(k) must have shape (norb, norb, 3), in eV angstrom.")
        if not np.allclose(gradient, gradient.swapaxes(0, 1).conj(), atol=1e-8, rtol=1e-8):
            raise ValueError("gradient(k) must be Hermitian in its orbital indices.")
        if not np.allclose(gradient[:, :, 2], 0, atol=1e-8):
            raise ValueError("gradient(k) must have zero kz derivative for this 2D model.")

        # Add the one-point k axis expected by PyMEX: H[a,b,k], gradient[a,b,k,xyz].
        return matrix[:, :, None], gradient[:, :, None, :]

    def validate_bands(self, bands):
        """Check that the selected bands fit this model."""
        validate_band_windows(bands, self.norb)

    def get_lattice(self):
        """Real-space unit-cell vectors, one row per vector (Å)."""
        return self.lattice.copy()

    def get_reciprocal(self):
        """Reciprocal vectors used to build the k mesh (inverse Å)."""
        return 2 * np.pi * np.linalg.inv(self.lattice).T

    def get_WF_loc(self, using_mlwf=False):
        """Positions of the orbitals, in the same order as the matrix rows (Å)."""
        return self.centres.copy()

    def get_num_wann(self):
        """Number of orbitals, which is the dimension of H(k)."""
        return self.norb

    def get_kgrid(self):
        """Number of mesh points along each reciprocal lattice vector."""
        return self.kgrid.copy()

    def map_WF(self, using_mlwf=False, mode="atom_centered"):
        """Give PyMEX one row per orbital: site, start index, end index, x, y, z."""
        # PyMEX uses an exclusive end index, so orbital i has the range [i, i+1).
        indices = np.arange(self.norb)
        return np.column_stack((indices, indices, indices + 1, self.centres)).astype(object)
