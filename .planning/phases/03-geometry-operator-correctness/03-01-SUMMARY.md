---
phase: 03-geometry-operator-correctness
plan: "01"
subsystem: test-fixtures
tags: [mesh, openfoam, graded-mesh, mpi, fixture]
dependency_graph:
  requires: []
  provides: [test/setup_pressureVelocityCoupling_graded/]
  affects: [03-03-PLAN.md (graded CTest entry consumes this fixture)]
tech_stack:
  added: []
  patterns: [OpenFOAM blockMesh + decomposePar workflow; committed pre-generated mesh]
key_files:
  created:
    - test/setup_pressureVelocityCoupling_graded/system/blockMeshDict
    - test/setup_pressureVelocityCoupling_graded/system/controlDict
    - test/setup_pressureVelocityCoupling_graded/system/decomposeParDict
    - test/setup_pressureVelocityCoupling_graded/system/fvSchemes
    - test/setup_pressureVelocityCoupling_graded/system/fvSolution
    - test/setup_pressureVelocityCoupling_graded/system/simulationParameters
    - test/setup_pressureVelocityCoupling_graded/constant/transportProperties
    - test/setup_pressureVelocityCoupling_graded/0/U
    - test/setup_pressureVelocityCoupling_graded/0/p
    - test/setup_pressureVelocityCoupling_graded/constant/polyMesh/boundary
    - test/setup_pressureVelocityCoupling_graded/constant/polyMesh/faces
    - test/setup_pressureVelocityCoupling_graded/constant/polyMesh/neighbour
    - test/setup_pressureVelocityCoupling_graded/constant/polyMesh/owner
    - test/setup_pressureVelocityCoupling_graded/constant/polyMesh/points
    - test/setup_pressureVelocityCoupling_graded/processor0/0/U
    - test/setup_pressureVelocityCoupling_graded/processor0/0/p
    - test/setup_pressureVelocityCoupling_graded/processor0/constant/polyMesh/ (9 files)
    - test/setup_pressureVelocityCoupling_graded/processor1/0/U
    - test/setup_pressureVelocityCoupling_graded/processor1/0/p
    - test/setup_pressureVelocityCoupling_graded/processor1/constant/polyMesh/ (9 files)
  modified: []
decisions:
  - "D-01: 2 MPI ranks (matches mpi2 case, consistent decomposeParDict)"
  - "D-02: New directory only; setup_pressureVelocityCoupling_mpi2 not touched"
  - "D-03: simpleGrading (2 1 1) with same NX=3 as existing pressure-velocity coupling cases"
metrics:
  duration: "~5 minutes"
  completed: "2026-05-12T05:27:42Z"
  tasks_completed: 2
  files_created: 36
---

# Phase 03 Plan 01: Graded-Mesh Fixture Summary

**One-liner:** 2-rank graded-mesh OpenFOAM case fixture with simpleGrading (2 1 1) — only content delta from mpi2 is one blockMeshDict line; blockMesh and decomposePar output committed for CTest direct-read.

## What Was Done

Created `test/setup_pressureVelocityCoupling_graded/` as a pre-generated 2-rank decomposed OpenFOAM case with x-direction grading factor 2. This fixture is required for [GEO-01] and [GEO-02] tests (Plan 03-03) to detect proc-face weight-reversal bugs that uniform meshes (`w = 0.5`) cannot reveal.

### Task 1: Copy and edit

Copied all 9 source-controlled input files from `test/setup_pressureVelocityCoupling_mpi2/` verbatim. Applied exactly one edit to `system/blockMeshDict`:

```
- hex (0 1 2 3 4 5 6 7) ($NX $NX $NX) simpleGrading (1 1 1)
+ hex (0 1 2 3 4 5 6 7) ($NX $NX $NX) simpleGrading (2 1 1)
```

All other files (`controlDict`, `decomposeParDict`, `fvSchemes`, `fvSolution`, `simulationParameters`, `transportProperties`, `0/U`, `0/p`) are byte-identical to their mpi2 counterparts.

Commit: `305514fb` — `chore(03-01): create graded case dir and apply simpleGrading (2 1 1)`

### Task 2: Generate and commit mesh

Ran the following from `test/setup_pressureVelocityCoupling_graded/`:

```bash
source /usr/lib/openfoam/openfoam2412/etc/bashrc
blockMesh       # regenerates constant/polyMesh/ from blockMeshDict with (2 1 1)
decomposePar    # partitions into processor0/ and processor1/ per decomposeParDict
```

Generated files committed:
- `constant/polyMesh/{boundary, faces, neighbour, owner, points}` — 5 files, base mesh
- `processor0/0/{U, p}` — rank-0 initial fields
- `processor0/constant/polyMesh/{boundary, boundaryProcAddressing, cellProcAddressing, faceProcAddressing, faces, neighbour, owner, pointProcAddressing, points}` — 9 files
- `processor1/0/{U, p}` — rank-1 initial fields
- `processor1/constant/polyMesh/{boundary, boundaryProcAddressing, cellProcAddressing, faceProcAddressing, faces, neighbour, owner, pointProcAddressing, points}` — 9 files

Commit: `d38a2ba7` — `chore(03-01): commit blockMesh + decomposePar output for graded fixture`

Total: 36 files, matching `setup_pressureVelocityCoupling_mpi2/` file count.

## Verification Results

| Check | Result |
|-------|--------|
| `simpleGrading (2 1 1)` in blockMeshDict (count=1) | PASS |
| `simpleGrading (1 1 1)` in blockMeshDict (count=0) | PASS |
| blockMeshDict identical to mpi2 except grading line | PASS |
| decomposeParDict byte-identical to mpi2 | PASS |
| simulationParameters byte-identical to mpi2 (NX 3) | PASS |
| constant/polyMesh/points exists | PASS |
| processor0/constant/polyMesh/points exists | PASS |
| processor1/constant/polyMesh/points exists | PASS |
| processor0/0/U exists | PASS |
| processor1/0/U exists | PASS |
| processor0 polyMesh file list matches mpi2 (9 files each) | PASS |
| Graded points differ from uniform mpi2 points | PASS |
| mpi2 case unmodified (D-02) | PASS |
| All 36 files committed (git status clean) | PASS |

## Deviations from Plan

None — plan executed exactly as written.

The worktree was reset to `feat/distributed-correctness` at task start because the spawned worktree-agent branch was initialized from an older commit that predated the mpi2 fixture. This was a setup issue, not a plan deviation; the reset was safe (the worktree-agent branch had no prior GSD work).

## Known Stubs

None. This plan produces a pure file-system fixture with no code stubs.

## Threat Flags

Not applicable — test fixture files only, no network endpoints, auth, or secrets.

## Self-Check: PASSED

- `test/setup_pressureVelocityCoupling_graded/system/blockMeshDict` — FOUND
- `test/setup_pressureVelocityCoupling_graded/constant/polyMesh/points` — FOUND
- `test/setup_pressureVelocityCoupling_graded/processor0/constant/polyMesh/points` — FOUND
- `test/setup_pressureVelocityCoupling_graded/processor1/constant/polyMesh/points` — FOUND
- Commit `305514fb` — FOUND
- Commit `d38a2ba7` — FOUND
