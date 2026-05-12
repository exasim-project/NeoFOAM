---
plan: 05-04
phase: 05-end-to-end-validation
status: partial-pass
self_check: PASSED (local) / PENDING (HPC E2E)
key-files:
  modified:
    - examples/neoIcoFoam/neoIcoFoam.cpp
    - src/NeoN/src/finiteVolume/cellCentred/fields/surfaceField.cpp
  created:
    - .planning/phases/05-end-to-end-validation/05-04-SUMMARY.md
---

## Summary

Two bugs identified and fixed across two iterations. CTest confirms no new regressions
introduced. HPC E2E run with `compare_fields.py` is the remaining validation step.

---

## Iteration 1: Missing U.correctBoundaryConditions() (from 05-03)

**Fix:** `examples/neoIcoFoam/neoIcoFoam.cpp:100` — added `U.correctBoundaryConditions()`
after `UEqn.solve(-1.0 * dsl::exp::grad(p))` in the `if (piso.momentumPredictor())` block.

**Root cause:** NeoN's `PDESolver::solve()` does not automatically call
`correctBoundaryConditions()` after convergence (unlike OpenFOAM's `fvMatrix::solve()`).
Ghost cells remained at IC value `(1,0,0)` after every momentum predictor solve.

**Commit:** `785f74fd fix(05-04): add U.correctBoundaryConditions() after momentum predictor solve`

---

## Iteration 2: syncProcFaceInternalVector sign inversion (MPI-02 regression)

**Root cause:** `SurfaceField::correctBoundaryConditions()` contained a `syncProcFaceInternalVector`
parallelFor that copied RECEIVED ghost values from `boundaryData().value()` proc-tail into
`internalVector()` proc-face slots after the Alltoallv exchange. This inverted the sign convention
that three operators depend on:

| Operator | What it reads | Expected sign | After MPI-02 |
|----------|--------------|--------------|--------------|
| `computeUpwindInterpolationWeights` | `faceFlux.internalVector()[procFaceIdx]` | LOCAL phi (positive = leaving) | NEIGHBOR phi (opposite sign) → inverted weights |
| `computeDivProcBoundImpl` | `faceFluxV[facei]` = F_local | LOCAL phi for diagContrib/offDiagContrib | NEIGHBOR phi → swapped matrix entries |
| `surfaceIntegrate` | `internalVector()[procFaceIdx]` | LOCAL flux | NEIGHBOR flux |

**Why procFaceCheck masked it:** `checkProcFaceConsistency` reads `boundaryData().value()` and
verifies FlipExpected symmetry — which passed even when `internalVector()` had wrong-sign values.

**Fix:** Removed the `parallelFor + fence` block from `SurfaceField::correctBoundaryConditions()`.
`internalVector()` proc-face slots now retain LOCAL values throughout. Ghost values (received after
exchange) remain in `boundaryData().value()` proc-tail, which is already the location where
`computeLinearInterpolation` and `computeUpwindInterpolation` read them.

**Files:**
- `src/NeoN/src/finiteVolume/cellCentred/fields/surfaceField.cpp` lines 47-62

**Commits:**
- `382cfacc9` (NeoN) `fix(05-04): remove syncProcFaceInternalVector from SurfaceField::correctBoundaryConditions`
- `5b50f26b` (NeoFOAM) `fix(05-04): bump NeoN submodule — remove syncProcFaceInternalVector`

---

## CTest Results (post-fix)

```
79% tests passed, 4 tests failed out of 19
```

| Test | Status | Notes |
|------|--------|-------|
| `distributedPressureVelocityCoupling` | FAIL (3 assertions) | Pre-existing: HbyA/updateFaceVelocity/solvePEqn proc-boundary sign mismatch |
| `distributedPressureVelocityCoupling_mpi2` | FAIL (3 assertions) | Same pre-existing |
| `distributedMomentum` | FAIL (SIGSEGV + 3 assertions) | Pre-existing: crash in distributedMomentum test |
| `distributedMomentum_mpi2` | FAIL | Pre-existing |
| All other 15/19 tests | PASS | — |

**Baseline comparison:** All 4 failures reproduce with the syncProcFaceInternalVector code
PRESENT (stash verified), confirming none are regressions from this fix.

**Remaining pre-existing issue — HbyA proc-boundary sign:** After `correctBoundaryConditions()`,
`nfHbyA.boundaryData().value()` proc-tail has opposite sign compared to
`HbyA.boundaryField()[processorPatch]` in OpenFOAM. Root cause not yet identified;
internal cell values of HbyA DO match (common.hpp:150 passes). Only the RECEIVED ghost
values (after MPI exchange) have opposite sign. This affects the HbyA, updateFaceVelocity,
and solvePEqn comparison sections. Investigation deferred — does not block the E2E run
since neoIcoFoam uses HbyA internal values, not its proc-boundary values, for divergence.

---

## HPC E2E Validation — PENDING

The HPC run on cylinder3D (4 MPI ranks, endTime 5e-3) is required to confirm the field
L∞ norms are within tolerance. Previous run (pre-fix, 05-03): L∞_U=4.77, L∞_p=83.5
(catastrophic divergence). Target post-fix: L∞_U < 1e-3, L∞_p < 1e-3.

Steps to complete:
```bash
cd tutorials/cylinder3D
./Allrun.reference        # 4-rank icoFoam reference
./Allrun_parallel         # 4-rank neoIcoFoam with both fixes
python compare_fields.py --threshold 1e-3
```
