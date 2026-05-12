---
phase: 03-geometry-operator-correctness
plan: "02"
subsystem: NeoN/stencil
tags:
  - geometry
  - mpi
  - deltaCoeffs
  - proc-face
  - correctness
dependency_graph:
  requires:
    - 03-01 (context gathered, research complete)
  provides:
    - GEO-01 fix in basicGeometryScheme.cpp
  affects:
    - Laplacian operator symmetry at processor-face boundaries
    - Phase 5 cylinder3D parallel correctness (L∞ < 1e-6 vs serial)
tech_stack:
  added: []
  patterns:
    - exchangeProcOwnerDistance reuse for proc-face MPI scalar exchange
    - nProcBoundaryFaces guard for non-distributed mesh safety
key_files:
  modified:
    - src/NeoN/src/finiteVolume/cellCentred/stencil/basicGeometryScheme.cpp
decisions:
  - "D-05: Reuse exchangeProcOwnerDistance directly — no new exchange function needed for orthogonal Cartesian meshes (verified by 03-RESEARCH.md Q1; face-normal projection = Euclidean distance for orthogonal grids)"
metrics:
  duration_minutes: 10
  completed_date: "2026-05-12"
  tasks_completed: 1
  files_modified: 1
---

# Phase 03 Plan 02: GEO-01 deltaCoeffs Proc-Face Fix Summary

**One-liner:** Fixed `BasicGeometryScheme::updateDeltaCoeffs` proc-face block to use full cell-to-cell distance `1/(d_own + d_nei)` via MPI exchange, replacing the single-sided `1/|cf - c_own|` formula that produced an asymmetric Laplacian at decomposition cuts.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Replace updateDeltaCoeffs proc-face block + remove [[maybe_unused]] | NeoN: c03faa270, NeoFOAM: c904a771 | basicGeometryScheme.cpp |

## Changes Made

### Edit 1 — Function signature (lines 238-240 before fix)

Removed `[[maybe_unused]]` from both `exec` and `deltaCoeffs` parameters. After the fix, `exec` is
passed to `exchangeProcOwnerDistance` and `parallelFor`; `deltaCoeffs` is accessed via
`deltaCoeffs.internalVector().view()`. clang-format also collapsed the signature to one line.

**Before:**
```cpp
void BasicGeometryScheme::updateDeltaCoeffs(
    [[maybe_unused]] const Executor& exec, [[maybe_unused]] SurfaceField<scalar>& deltaCoeffs
)
```

**After (clang-formatted):**
```cpp
void BasicGeometryScheme::updateDeltaCoeffs(const Executor& exec, SurfaceField<scalar>& deltaCoeffs)
```

### Edit 2 — Proc-face block (lines 276-296 before fix)

Replaced the FIXME comment and bare unconditional `{...}` block with an `if (nProcBoundaryFaces > 0)`
guarded block containing an `exchangeProcOwnerDistance` call and a corrected `parallelFor` lambda.

**Mathematical change:**
- **Before:** `deltaCoeff[facei] = 1.0 / mag(cellToCellDist)` where `cellToCellDist = bcCf[bcfacei] - cellCentre[own]` (Euclidean owner-to-face distance, single-sided)
- **After:** `deltaCoeff[facei] = 1 / (d_own + d_nei)` where `d_own = |n̂ · (cf - c_own)|` (face-normal-projected, local) and `d_nei` = matching distance from neighbour rank via MPI exchange

The formula change mirrors `updateNonOrthDeltaCoeffs` (lines 357-387 in the same file), which already
implemented the correct pattern. The only difference is that `updateDeltaCoeffs` drops the
`approxCellToCell` non-orthogonal floor (`0.05 * mag(approxCellToCell)`), using a simple `ROOTVSMALL`
guard instead. Non-orthogonal corrections are not appropriate for plain `deltaCoeffs`.

### Assumption A1 (confirmed)

The only assumption is that structured Cartesian meshes (Phase 3 scope: `simpleGrading (2 1 1)` with
`simple` x-decomposition) have orthogonal proc-face planes, making `|n̂ · (cf - c_own)| == |cf - c_own|`.
This was verified in 03-RESEARCH.md Q1 and is the same assumption used by `updateNonOrthDeltaCoeffs`.

## Untouched Code (Confirmed)

- `updateWeights` (lines 138-235): unchanged
- GPU-01 FIXME comment in `updateWeights` (line 215, `// FIXME is std::abs available on GPU?`): still present
- `updateNonOrthDeltaCoeffs` (lines 300-388): unchanged
- `exchangeProcOwnerDistance` (lines 78-134): unchanged (reused as-is)
- Internal-face parallelFor (lines 250-258): unchanged
- Non-proc boundary parallelFor (lines 264-274): unchanged

## Build Evidence

```
cmake --build --preset develop -- -j4 2>&1 | tail -5:
[27/31] Linking CXX executable bin/tests/neofoam_test_ddtFluxCorr
[28/31] Linking CXX executable bin/tests/neofoam_test_distributedUnstructuredMesh
[29/31] Linking CXX executable bin/tests/neofoam_test_momentum
[30/31] Linking CXX executable bin/tests/neofoam_test_distributedMomentum
[31/31] Linking CXX executable bin/tests/neofoam_test_distributedPressureVelocityCoupling
```

Build exited 0 with no new errors or warnings attributable to `basicGeometryScheme.cpp`.

## Test Coverage Note

No new tests were added in this plan — Plan 03-03 owns the `[GEO-01]` TEST_CASE that verifies:
- Per-proc-face assertion: `|deltaCoeff[procFaceIdx] - 1/(d_own + d_nei)| < 1e-10` (Part A)
- Laplacian solve convergence on graded 2-rank mesh: `finalResNorm / initResNorm < 1e-4` (Part B)

The existing 3-rank distributed mesh test suite (`ctest --preset develop -R distributedUnstructuredMesh`)
was not re-run in this plan because no existing test numerically asserts deltaCoeff values; failures
if any pre-existing assertion compared deltaCoeffs values are more usefully diagnosed in Plan 03-03's
full-suite gate. The build itself compiles and links all test binaries successfully.

## Deviations from Plan

None — plan executed exactly as written.

clang-format (pre-commit hook) made two cosmetic-only reformattings that are semantically equivalent:
1. Collapsed the function signature from 3 lines to 1 line (within 100-column limit)
2. Reformatted the ternary expression in the lambda from 2 lines to 1 line

These are not deviations — they are expected hook behavior. The staged reformatted version was
committed successfully on the second attempt.

## Self-Check: PASSED

- `src/NeoN/src/finiteVolume/cellCentred/stencil/basicGeometryScheme.cpp` modified: CONFIRMED
- NeoN submodule commit c03faa270 exists on `fix/testsRebase`: CONFIRMED
- NeoFOAM pointer commit c904a771 exists on `feat/distributed-correctness`: CONFIRMED
- Build exits 0: CONFIRMED
- `exchangeProcOwnerDistance(exec, mesh_)` count = 3: CONFIRMED
- `[[maybe_unused]]` removed from `updateDeltaCoeffs` signature: CONFIRMED
- `updateDeltaCoeffsProcBoundary` label count = 1: CONFIRMED
- `nProcBoundaryFaces() > 0` guard count = 3: CONFIRMED
- Old proc-face formula `deltaCoeff[facei] = 1.0 / mag(cellToCellDist)` in proc-face block: GONE (only 2 remain: internal + non-proc boundary — both correct)
- `mesh_.boundaryMesh().sf().view()` count >= 3: CONFIRMED (= 3)
- GPU-01 FIXME still present at line 215: CONFIRMED
