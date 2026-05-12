---
plan: 05-03
phase: 05-end-to-end-validation
status: complete
self_check: PASSED
key-files:
  created:
    - examples/neoIcoFoam/neoIcoFoam.cpp
    - include/NeoFOAM/auxiliary/procFaceCheck.hpp
    - .planning/phases/05-end-to-end-validation/05-03-CASE-RESOLUTION.md
  modified: []
---

## Summary

Instrumented `neoIcoFoam.cpp` with 6 `checkProcFaceConsistency` calls at all PISO
checkpoints, rebuilt and ran on HPC (4 MPI ranks, cylinder3D, endTime=5e-3), and
identified the root cause of parallel divergence.

## What Was Built

- `include/NeoFOAM/auxiliary/procFaceCheck.hpp` — MPI_Sendrecv-based cross-rank
  consistency checker; no-op on serial runs.
- `examples/neoIcoFoam/neoIcoFoam.cpp` — instrumented at 5 PISO checkpoints:
  U+phi after rotateOldTimes, U after momentumPredictor, p after correctBC,
  phi after updateFaceVelocity, U after updateVelocity.
- `.planning/phases/05-end-to-end-validation/05-03-CASE-RESOLUTION.md` — full root-cause
  analysis and proposed fix for Plan 05-04.

## Diagnostic Results

**compare_fields.py:** L∞_U=4.77, L∞_p=83.5 at t=0.005 → catastrophic divergence.

**procFaceCheck findings:**

| Checkpoint | Status | Details |
|------------|--------|---------|
| U after rotateOldTimes | OK | Fields consistent at loop start |
| phi after rotateOldTimes | OK | Flux consistent |
| **U after momentumPredictor solve** | **FAILED** | All ranks, all proc patches, every timestep |
| (subsequent checks not reached — neoIcoFoam crashed/diverged) | — | — |

**First failure:** timestep 1, rank 0 patch 0 (→1) face 326:
```
local.boundary=(1.0, 0.0, 0.0)  ← IC / stale
remote.internal=(1.000000e+00 .. 5.736177e+00)  ← solved interior
```

## Root Cause

`UEqn.solve(-1.0 * dsl::exp::grad(p))` in the momentum predictor updates interior U
values but does NOT automatically update processor boundary faces. NeoN's `PDESolver::solve()`
lacks the automatic `correctBoundaryConditions()` that OpenFOAM's `fvMatrix::solve()` calls
internally. The missing call leaves every rank's ghost cells at the initial condition value
`(1, 0, 0)` while neighbors have already solved their interiors.

## Proposed Fix (for Plan 05-04)

Add `U.correctBoundaryConditions()` at
[examples/neoIcoFoam/neoIcoFoam.cpp:99](examples/neoIcoFoam/neoIcoFoam.cpp) — one line
after `UEqn.solve(...)` inside the `if (piso.momentumPredictor())` block.

## Commits

- `0d8572bf` — feat(05-03): instrument neoIcoFoam with procFaceCheck for divergence diagnosis
- (CASE-RESOLUTION committed in 05-04 plan)
