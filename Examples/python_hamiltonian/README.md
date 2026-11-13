# Python tight-binding Hamiltonian input

This experimental input adapter lets a Python file supply the small electronic
Hamiltonian `H(k)` directly to PyMEX. It avoids generating Wannier90 input files
for a model that already defines its Hamiltonian. PyMEX still solves the
electronic eigenproblem, constructs the BSE matrix, performs full LAPACK
diagonalization and computes its native optical output.

The examples are the project's two-band hBN model and spinless three-band TMD
model, with MoS2 selected by default. Both examples use the existing 6×6
independent-case inputs. The TMD file also contains the project's MoSe2, WS2 and
WSe2 parameter sets; only MoS2 was rerun for this branch's validation.

This branch contains only input support. The potential routines, Cython kernel,
matrix construction, eigensolvers and optical routines are unchanged from
upstream `23939d5de4aecc05e805af519fd2b07058e311b5`. In particular, it does not
contain the separate independent-case performance optimization or older
project-specific potential changes. It has not been deployed to production.

## Run an example

Use an environment with PyMEX's dependencies and compiled `cyfunc` extension.
From the repository root, when the extension has been built in `src`:

```sh
export PYTHONPATH="$PWD/src:$PYTHONPATH"
cd Examples/python_hamiltonian/hbn
python ../calc.py
```

Use `tmd` instead of `hbn` for the three-band example. If `cyfunc` lives in a
separate build directory, add that directory to `PYTHONPATH` too. Run with
`mpiexec -n 4 python ../calc.py` to use four MPI processes.

Native HDF5 files and `sigma_unpolarized_tb.txt` are written in the current
directory. Repeating a run there overwrites those outputs. The geometry snapshot
`model_geometry.npz` is written by the root MPI process only.

## Model contract

The Python file defines:

```python
import numpy as np

lattice = np.array([...])  # (3, 3): lattice vectors as Cartesian rows, in angstrom
centres = np.array([...])  # (norb, 3): orbital centres in Hamiltonian row order

def hamiltonian(k):
    # k is a three-component Cartesian vector in inverse angstrom.
    return ...  # finite Hermitian (norb, norb) matrix, in eV
```

Use a periodic, orthonormal localized-orbital basis: `H(k + G) = H(k)`. Put cell
translations in hopping phases. Orbital-centre contributions to optical matrix
elements are supplied separately through `centres`. The reader samples
periodicity, Hermiticity and independence of kz during its input checks.

An optional `gradient(k)` returns a `(norb, norb, 3)` matrix of Cartesian
derivatives in eV Å. Without it, the adapter uses the project's existing
five-point finite difference, with default `derivative_step: 1e-5` in inverse
angstrom. Supplied gradients are checked against numerical differences at sample
points. The third derivative must be zero for the supported 2D models.

Lattice vectors and orbital centres can instead be supplied in `bse.tb.geometry`.
They must describe the same Hamiltonian and basis. `basis`, `energy_unit` and
`length_unit`, if supplied, must be `orthonormal`, `eV` and `angstrom`.

## YAML selection

The relevant part of the hBN example is:

```yaml
bse:
  method: tb
  tb:
    engine: python
    model: model.py
    kgrid: [6, 6, 1]
    bands:
      vb_idx: [0, 1]
      cb_idx: [1, 2]
  Qpoints:
    hspoints: [[0.0, 0.0, 0.0]]
    num_points: 1
```

`model` is resolved relative to the YAML file. Band indices are zero-based and
the end index is excluded. The TMD example uses `cb_idx: [1, 3]` to retain both
conduction bands. One band selection is copied into the energy and eigenvector
windows; conflicting windows are rejected. Do not include `wannier_io` with
`engine: python`. Existing Wannier90 configurations continue to use their normal
input path.

The complete example YAMLs retain the project's screening, native broadening
and photon-energy settings. In this PyMEX version, photon-energy `min` and `max`
are offsets from the lowest BSE eigenvalue, and `sigma` is the Gaussian standard
deviation. When selecting another TMD parameter set, its lattice and screening
inputs must agree with that material.

## Current limits

- A single, gapped 2D monolayer in the xy plane; all orbital centres share one z.
- Spinless/unpolarized optical calculations at one Q = 0 point.
- Fresh CPU runs with LAPACK; no restart reuse, GPU or alternative eigensolver support in this adapter.
- Orthonormal localized orbitals; no overlap matrix or nonperiodic Bloch convention.
- Real-space analytic monolayer Keldysh configuration; no multilayer input support.
- No band shifts or external electric field. Screened exchange requires direct interaction to be enabled.

These limits are enforced by the reader where applicable. This is a focused
interface for the two model families, not a general replacement for Wannier90
or a claim of support for arbitrary Hamiltonians.

## Validation

Eight input-contract tests passed. Eight fresh native PyMEX workflows also
passed: hBN and MoS2, meshes 6×6 and 9×9, with one and four MPI processes.
All checked numerical arrays and native broadened spectra matched the saved
native independent-case references exactly, including the energy axis.

The checks include the complete BSE matrix, all BSE eigenvalues/eigenvectors,
optical amplitudes, electronic Hamiltonians and derivatives, electronic
eigenvalues/eigenvectors, full state counts and the readable geometry snapshot.
No broadening, normalization, energy shift or state truncation was added.

These new integration checks cover independent cases. Interacting kernels are
unchanged upstream code; this clean branch does not claim a new interacting
regression test or equivalence to the project's separate potential modifications.
The existing Wannier90 selection path has a configuration regression test, but
no new full Wannier90 calculation was run. No convergence claim is made.

Run the input checks from the repository root:

```sh
python -m unittest discover -s tests/python_input -v
```

The [native validation runner](../../tests/python_input/validate_native.py) takes
explicit source, extension, reference and scratch locations; see its `--help`.
It requires the private saved reference corpus and refuses to overwrite its work
directory. The [compact validation record](../../tests/python_input/results/native_validation.json)
contains source, reference and output hashes. Raw calculations and large logs
are not included in Git.
