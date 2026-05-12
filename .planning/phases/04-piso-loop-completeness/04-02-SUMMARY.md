---
phase: 04-piso-loop-completeness
plan: "02"
subsystem: examples/neoIcoFoam
tags: [continuity-error, piso-loop, mpi, neon-dsl, diagnostics]
dependency_graph:
  requires: []
  provides: [PISO-03]
  affects: [examples/neoIcoFoam/neoIcoFoam.cpp]
tech_stack:
  added: []
  patterns:
    - NeoN dsl::Expression<scalar> wrapper to access returning explicitOperation(localIdx) overload
    - NeoN parallelReduce over |div(phi)| for local rank continuity sum
    - NeoN::mpi::allReduce(Sum) for global continuity error across MPI ranks
key_files:
  modified:
    - examples/neoIcoFoam/neoIcoFoam.cpp
decisions:
  - Use Expression<scalar> wrapper (not SpatialOperator directly) to obtain the returning explicitOperation overload
  - Compute sumLocal per rank, then MPI allReduce for globalErr — matches icoFoam continuityErrs.H semantics
  - cumulativeContErr declared before the time loop (never reset per step) to accumulate across all time steps
metrics:
  duration: "< 5 minutes"
  completed: "2026-05-12"
  tasks_completed: 2
  tasks_total: 2
  files_changed: 1
---

# Phase 04 Plan 02: Continuity Error Reporting (PISO-03) Summary

NeoN-native continuity error reporting added inline to the PISO loop in neoIcoFoam.cpp, replacing the `// #include "continuityErrs.H"` comment. Output format matches icoFoam exactly for diffable logs.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Declare cumulativeContErr before the time loop | b502233b | examples/neoIcoFoam/neoIcoFoam.cpp |
| 2 | Replace continuityErrs.H comment with inline continuity computation (PISO-03) | 8838db32 | examples/neoIcoFoam/neoIcoFoam.cpp |

## What Was Built

- `NeoN::scalar cumulativeContErr = 0.0` declared before `while (runTime.loop())` — accumulates across all time steps and PISO iterations, never reset (matching icoFoam's `continuityErrs.H` behaviour)
- Inside `while (piso.correct())`, after the `correctNonOrthogonal` loop and before `nf::updateVelocity`:
  - `NeoN::dsl::Expression<NeoN::scalar> divExpr(rt.exec)` wraps `dsl::exp::div(phi)` to use the returning `explicitOperation(nCells)` overload (the `SpatialOperator` overload takes a pre-allocated `Vector&` with void return; only `Expression::explicitOperation(localIdx)` allocates and returns)
  - `NeoN::parallelReduce` sums `|divPhi[i]|` over all cells on this rank → `sumLocal`
  - `NeoN::mpi::allReduce(globalErr, ReduceOp::Sum, rt.mpiEnvironment.comm())` gathers global error
  - `cumulativeContErr += globalErr` accumulates
  - `NeoN::Logging::info("time step continuity errors : sum local = {} global = {} cumulative = {}", ...)` matches icoFoam's exact label string

## Deviations from Plan

None — plan executed exactly as written.

## Decisions Made

1. `Expression<scalar>` wrapper pattern — the plan explicitly specified this and it is correct: `SpatialOperator::explicitOperation` has void return requiring a pre-allocated `Vector&`, while `Expression::explicitOperation(localIdx)` allocates and returns a `Vector<ValueType>`.
2. `cumulativeContErr` declared at outermost scope (before the time loop), not inside the PISO body — ensures it persists across all time steps.
3. Log format string is byte-identical to icoFoam's `continuityErrs.H` output: `"time step continuity errors : sum local = {} global = {} cumulative = {}"` — this allows direct log diffing between serial icoFoam and parallel neoIcoFoam runs.

## Known Stubs

None. All data is live: `phi` is the active NeoN SurfaceField updated each PISO iteration.

## Threat Flags

No new trust-boundary surface introduced. The `allReduce` call is a blocking MPI collective matching the existing halo-exchange pattern already present in the code.

## Self-Check: PASSED

| Check | Result |
|-------|--------|
| examples/neoIcoFoam/neoIcoFoam.cpp exists | FOUND |
| 04-02-SUMMARY.md exists | FOUND |
| Commit b502233b (Task 1) | FOUND |
| Commit 8838db32 (Task 2) | FOUND |
