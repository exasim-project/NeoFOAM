---
phase: 04-piso-loop-completeness
plan: "01"
subsystem: algorithms/datastructures
tags: [PDESolver, assembled-guard, MPI, rank-guard, HLTH-02, PISO-02]
dependency_graph:
  requires: []
  provides: [PDESolver.isAssembled, PDESolver.setReference-any-rank]
  affects: [include/NeoFOAM/datastructures/pdeSolver.hpp, src/algorithms/pressureVelocityCoupling.cpp]
tech_stack:
  added: []
  patterns: [assert-before-use, call-site-gate]
key_files:
  created: []
  modified:
    - include/NeoFOAM/datastructures/pdeSolver.hpp
    - src/algorithms/pressureVelocityCoupling.cpp
decisions:
  - "D-01: isAssembled_ private flag initialized to false in both primary and copy constructors; set to true only by assemble()"
  - "D-02: setReference() sets needReference_ = true unconditionally; call-site gate (pRefCell >= 0) at neoIcoFoam.cpp enforces owning-rank-only semantics per OpenFOAM's setRefCell() contract"
  - "D-03: SetReference::operator() rank guard removed; new contract comment explains pRefCell >= 0 call-site gate"
  - "D-09: computeRAU and computeRAUandHByA throw std::runtime_error when isAssembled() is false, replacing the TODO comment"
metrics:
  duration: "2m 5s"
  completed: "2026-05-12T07:52:13Z"
  tasks_completed: 3
  tasks_total: 3
  files_modified: 2
---

# Phase 4 Plan 01: PDESolver assembled-guard and rank-guard removal Summary

**One-liner:** Added `isAssembled_` tracking flag to `PDESolver` with runtime throw guards in `computeRAU`/`computeRAUandHByA`, and removed MPI rank-0 guards from `setReference`/`SetReference` so the pressure reference cell can be pinned on any owning rank.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Add isAssembled_ guard to PDESolver (HLTH-02) | 153c8153 | include/NeoFOAM/datastructures/pdeSolver.hpp |
| 2 | Remove rank guards from setReference / SetReference (PISO-02) | 65cec84c | include/NeoFOAM/datastructures/pdeSolver.hpp |
| 3 | Add isAssembled() guard to computeRAU and computeRAUandHByA (HLTH-02) | c26e2828 | src/algorithms/pressureVelocityCoupling.cpp |

## What Was Built

### PDESolver isAssembled_ tracking (HLTH-02)

Added a `bool isAssembled_` private member to `PDESolver<ValueType>`:
- Primary constructor: initialized to `false` (`isAssembled_(false)`)
- Copy constructor: propagates the flag (`isAssembled_(expr.isAssembled_)`)
- `bool isAssembled() const` public accessor added after `exec()`
- `assemble()` single-arg overload sets `isAssembled_ = true` after `expr_.assemble()`

### Rank guard removal (PISO-02)

Two related rank guards removed from `pdeSolver.hpp`:

1. `SetReference::operator()`: Removed the `NeoN::mpi::Environment mpiEnv; if (mpiEnv.isInitialized() && mpiEnv.rank() != 0) { return; }` early-return block. Replaced with a contract comment explaining that OpenFOAM's `setRefCell()` assigns `pRefCell = -1` to non-owning ranks, and the `pRefCell >= 0` gate at the `neoIcoFoam` call site guarantees `setReference()` is only invoked on the owning rank.

2. `setReference()`: Removed the `if (runTime_.mpiEnvironment.rank() == 0)` guard. The method now unconditionally sets `needReference_ = true`, `pRefCell_`, and `pRefValue_`. This is safe because the caller (`neoIcoFoam.cpp:134`) already gates on `pRefCell >= 0`.

### computeRAU / computeRAUandHByA guards (HLTH-02, D-09)

Both free functions in `pressureVelocityCoupling.cpp` now open with:
```cpp
if (!expr.isAssembled())
{
    throw std::runtime_error(
        "PDESolver::computeRAU called on unassembled system — call assemble() first"
    );
}
```
The `// TODO this assumes an assembled matrix` comment in `computeRAU` was removed.

## Deviations from Plan

### Plan acceptance criteria inconsistency (non-blocking)

The plan's Task 1 acceptance criteria states `grep -c "isAssembled_(false)"` should return 2, implying both constructors use `isAssembled_(false)`. However the plan's own Edit 2 action spec shows the copy constructor using `isAssembled_(expr.isAssembled_)` — which is also what the `<behavior>` block specifies ("Copy constructor copies isAssembled_ from source"). The implementation follows the behavior spec and the action spec (propagates the value), not the contradictory count. Grep count for `isAssembled_(false)` is 1 (primary ctor only).

## Known Stubs

None — both modified files are algorithmic implementation; no placeholder data or hardcoded empty values introduced.

## Threat Surface Scan

No new network endpoints, auth paths, file access patterns, or schema changes introduced. The changes are purely internal to `PDESolver` class mechanics. The T-04-02 threat (elevation of privilege via SetReference without rank guard) is mitigated by the call-site `pRefCell >= 0` gate as documented in the new contract comment.

## Self-Check: PASSED

| Item | Status |
|------|--------|
| include/NeoFOAM/datastructures/pdeSolver.hpp | FOUND |
| src/algorithms/pressureVelocityCoupling.cpp | FOUND |
| .planning/phases/04-piso-loop-completeness/04-01-SUMMARY.md | FOUND |
| commit 153c8153 (Task 1) | FOUND |
| commit 65cec84c (Task 2) | FOUND |
| commit c26e2828 (Task 3) | FOUND |
