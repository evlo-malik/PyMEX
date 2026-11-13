# Independent-transition matrix construction: validation

Validated on 13 November 2026. **All 46 actual PyMEX runs passed the final
audit.** This is an isolated, modified PyMEX experiment. Production adoption
is pending; existing production sources, results and running jobs were left
unchanged. These checks establish implementation agreement, not mesh convergence
or validation of the underlying interaction model.

## Change

In the tight-binding (`method: tb`) branch of `BSE.write_exciton_H`, when both
`eh_interaction.include.direct` and `exchange` are false, each row is filled
directly using

```text
H[cvk, c'v'k'] = (E_c(k+Q) - E_v(k)) delta_cc' delta_vv' delta_kk'.
```

This skips the general interaction contractions with zero potentials. The
`(c, v, k)` basis order, dense complex128 matrix, MPI distribution/gathering and
HDF5 layout are preserved. Full native LAPACK diagonalization, all selected
eigenstates, optical amplitudes, photon-energy settings and native Gaussian
broadening are retained. The interacting and DFT-input branches are unchanged.
There is no replacement eigensolver or modification to `cyfunc.pyx`.

The optimization is a separate commit above parent
`67165cc63e719f75a1e5cc62e0ab04b9af959a8e`. That parent records the project's
existing Python Hamiltonian adapter and previously validated MPI input support
above upstream `23939d5de4aecc05e805af519fd2b07058e311b5`.
Only `src/bse.py` differs between the parent and optimized runtime sources.

## Coverage

| Calculations | Meshes | MPI processes | Runs |
|---|---|---|---:|
| Independent: hBN, MoS2, MoSe2, WS2, WSe2 | 6×6, 9×9 | 1 and 4 | 20 |
| Independent: the same five models | 30×30, 48×48 | 1 | 10 |
| Direct and direct + exchange: hBN, MoS2 | 9×9 | 1 and 4 | 8 |
| Original timing pairs: hBN, MoS2 | 30×30 | 1 | 4 |
| Exact-parent timing pairs: hBN, MoS2 | 30×30 | 1 | 4 |
| **Total** | | | **46** |

All runs use the existing spinless, Q=0 Python-model inputs. Every physical
configuration field was compared with its reference; only the model file
location was normalized to `model.py`. Model file hashes match. Broadening is
the original sigma = 0.025 eV. Every run retained its complete eigenstate set:
N² for hBN and 2N² for the TMDs, up to 2,304 and 4,608 states respectively.

Coverage does not establish finite-Q, DFT-input, exchange-only, ELPA, other band
windows or larger-mesh behavior. The formula applies generally to the independent
TB matrix, but those cases were not run in this experiment.

## Numerical results

The final audit reread the saved arrays and independently checked inputs,
shapes, dtypes, finiteness, complete state counts, output hashes, the full BSE
matrix, every eigenvalue and eigenvector, optical amplitudes, single-particle
Hamiltonians, Hamiltonian derivatives, and native broadened TXT spectra.

**All 42 same-platform comparisons agree exactly element by element in the
checked arrays and native spectra**, including the interacting controls and
the two exact-parent optimized runs. All 14 serial/four-process pairs also
have identical native-output file hashes.

Four comparisons use hBN references originally calculated on macOS. Maximum
differences across these comparisons are:

| Quantity | Maximum difference |
|---|---:|
| Full BSE matrix | 7.11e-15 eV |
| All BSE eigenenergies | 3.55e-15 eV |
| Single-particle Hamiltonian | 3.32e-15 eV |
| Hamiltonian derivatives | 3.85e-10 eV Å |
| Native photon-energy axis | 0 eV |
| Native spectrum, maximum absolute difference / reference peak | 3.95e-12 |
| Summed optical strengths within degenerate groups, relative to peak group strength | 3.99e-12 |

Individual cross-platform eigenvectors and optical amplitudes are **not**
elementwise identical: maximum differences are 1.0 and 11.41 respectively.
Degenerate states can be reordered or rotated and their phases are arbitrary.
The audit therefore also checked eigenpair residuals and degenerate subspaces
using a fixed 1e-10 eV grouping tolerance. Both platforms have zero eigenpair
residuals here; subspace projection and within-group orthogonality errors are
zero. Grouped optical strengths agree to the precision reported above.

Those group/subspace calculations are separate diagnostic analysis of native
outputs. They do not alter the eigenvectors, amplitudes, energy axis, native
broadening or saved spectra. No smoothing, fitting, alignment, normalization
of outputs, or eigenstate truncation was applied.

Same-platform acceptance requires exact array agreement. Cross-platform
acceptance uses the original 1e-9 absolute tolerance for matrices/energies/
gradients/energy axes and 1e-9 peak-relative tolerance for spectra and grouped
strengths. Tolerances were not increased to accommodate discrepancies.

## Measured performance

The primary benchmarks use fresh runs from the **exact parent commit** and
the optimized source, on the same Linux machine, with one MPI process and one
thread per run. All Python source files except the patched `bse.py` match.
Both use the same unchanged native compiled extension, copied inputs and
RAM-backed working storage.

| Model, 30×30 | Matrix: parent → optimized | Matrix speedup | Whole workflow: parent → optimized | Workflow speedup |
|---|---:|---:|---:|---:|
| hBN | 30.215 s → 0.103 s | 293.8× | 31.499 s → 1.391 s | 22.7× |
| MoS2 | 83.686 s → 0.288 s | 290.5× | 87.705 s → 4.316 s | 20.3× |

Matrix time includes native matrix construction and HDF5 writing. Whole-workflow
time is the sum of the saved `setup_s`, `matrix_s`, `diagonalization_s` and
`optical_s` measurements. It excludes Python/MPI launch, later comparisons,
hashing and archive decompression. The original harness's outer
`elapsed_seconds` includes validation overhead and is not used for speedups.

These are single matched observations on a shared machine with background jobs,
not guaranteed performance or statistical timing estimates. They do not measure
75×75 or substantiate a 100× whole-workflow speedup at that mesh. Dense matrix
storage, full diagonalization and native optical processing still limit larger
calculations; this patch does not solve their memory/storage scaling.

The original 42-run suite is preserved unchanged. Its timing baseline used the
production copy, whose `unit2super` helper differs from this branch's upstream
parent (`A @ S` versus `S @ A`), and whose Python adapter retains a serial-only
guard. The lattice expressions coincide for these square in-plane meshes and
the same outputs were obtained. Nevertheless, four additional runs against the
exact parent remove those source differences from the primary timing comparison.

## Evidence and reproducibility

- [Original 42-run records](results/independent-shortcut-summary.json): native stage timings, input/output hashes and original numerical comparisons.
- [Four exact-parent runs](results/parent-benchmark.json): the primary timing comparisons and both source manifests.
- [Final 46-run audit](results/audit.json): independent array checks, reference hashes, input/state-count checks, degeneracy diagnostics, serial/MPI hash comparisons and software versions.
- [Provenance](results/provenance.json): commit IDs and source/extension identities.
- [Original integration harness](test_independent_shortcut.py), [exact-parent benchmark](benchmark_parent.py), [audit](audit_completed_runs.py) and [native workflow driver](driver.py).

The scripts require the existing private reference corpus and compiled native
PyMEX environment. Raw HDF5 files, large logs and environments are intentionally
not committed. Reference paths in the reports are relative to the private
project root; actual roots are supplied as arguments/environment variables.
This is a reproducible experiment against that corpus, not a standalone CI test.

For a fresh experiment, stage the optimized `src/*.py` files under
`$VALIDATION_PROJECT_ROOT/independent-shortcut-validation/src`, and the exact
parent's `src` directory under `independent-shortcut-validation/baseline/src`.
Copy the validation scripts there and retain the reference layout named in the
reports. Use a new scratch directory with sufficient RAM-backed space. Then:

```sh
python test_independent_shortcut.py --project-root "$VALIDATION_PROJECT_ROOT" --scratch "$VALIDATION_SCRATCH"
python benchmark_parent.py --project-root "$VALIDATION_PROJECT_ROOT" --scratch "$VALIDATION_SCRATCH"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python audit_completed_runs.py --project-root "$VALIDATION_PROJECT_ROOT" --scratch "$VALIDATION_SCRATCH"
```

The run scripts refuse to overwrite existing case directories. The audit only
reads completed outputs, temporarily restoring archived reference eigenvectors
in scratch and verifying their recorded SHA-256 before use. References that
have `status.json` are checked against its original hashes. Small MPI references
have no such status manifest; the audit records their current hashes and checks
them against the new runs, without claiming an earlier stored hash exists.

The production Python source hashes still match the inventory taken before the
earlier MPI validation. The compiled kernel's source is byte-identical to
upstream commit `23939d5`; its SHA-256 and the shared compiled extension's
SHA-256 are recorded. No production calculation was switched to this branch.
