---
phase: 04-piso-loop-completeness
reviewed: 2026-05-12T00:00:00Z
depth: standard
files_reviewed: 4
files_reviewed_list:
  - include/NeoFOAM/datastructures/pdeSolver.hpp
  - src/algorithms/pressureVelocityCoupling.cpp
  - examples/neoIcoFoam/neoIcoFoam.cpp
  - test/test_distributedPressureVelocityCoupling.cpp
findings:
  critical: 4
  warning: 4
  info: 3
  total: 11
fixed:
  critical: 3   # CRIT-01, CRIT-02, CRIT-03
  warning: 2    # WARN-01, WARN-02
status: pass_with_warnings
---

# Phase 04: Code Review Report

**Reviewed:** 2026-05-12
**Depth:** standard
**Files Reviewed:** 4
**Status:** PASS-WITH-WARNINGS (3 criticals and 2 warnings fixed post-review)

## Summary

Original: 4 critical · 4 warnings · 3 info — all 3 Critical-phase-4 issues fixed; 1 critical (CRIT-04) is pre-existing test code not introduced in Phase 4.

Phase 4 adds three targeted changes: the `isAssembled_` guard (HLTH-02), the rank-guard removal
from `SetReference` (PISO-02), and the NeoN-native continuity error reporting (PISO-03). The
guard removal and the new test case are structurally sound but have correctness defects in their
supporting details. The continuity error computation is the most impactful issue: it reports
values with different units and semantics than the icoFoam reference it claims to match, making
cross-validation unreliable. A secondary critical issue is that `isAssembled_` is never set by
the `solve()` path, so the flag gives a stale answer whenever the caller uses the implicit
assembly inside `iterativeSolveImpl`. Two additional criticals concern the wrong function name in
an error message and the pre-existing "solve pEqn" test section, which still carries the old
rank-0 guard that PISO-02 was meant to eliminate.

---

## Critical Issues

### [CRIT-01] `computeRAUandHByA` guard throws with wrong function name

**File:** `src/algorithms/pressureVelocityCoupling.cpp:68`
**Severity:** Critical

**Finding:**
The error message in `computeRAUandHByA`'s pre-condition check reads:

```
"PDESolver::computeRAU called on unassembled system — call assemble() first"
```

This is copy-pasted from `computeRAU` (line 43) and incorrectly identifies the function. When a
caller hits this exception from `computeRAUandHByA`, the message points to the wrong function,
making the failure actively misleading during debugging.

**Recommendation:**
```cpp
// pressureVelocityCoupling.cpp line 68
throw std::runtime_error(
    "computeRAUandHByA called on unassembled system — call assemble() first"
);
```

---

### [CRIT-02] `isAssembled_` is stale after `solve()` without prior `assemble()`

**File:** `include/NeoFOAM/datastructures/pdeSolver.hpp:173`
**Severity:** Critical

**Finding:**
`PDESolver::solve()` delegates to `solveImpl`, which calls
`NeoN::dsl::detail::iterativeSolveImpl`. That function internally calls `exp.assemble(t, dt, ls,
ps)`, so `ls_` is populated after `solve()` returns. However, `isAssembled_` is only set to
`true` inside `PDESolver::assemble()` (line 144). It is never touched in `solveImpl`. Therefore:

```
pEqn.solve();            // ls_ is assembled, but isAssembled_ remains false
pEqn.isAssembled();      // returns false — incorrect
```

In the current `neoIcoFoam.cpp` PISO loop, `pEqn.solve()` is called without a prior
`pEqn.assemble()`. `updateFaceVelocity` then accesses `pEqn.linearSystem()` directly and works
correctly because `ls_` was populated inside `iterativeSolveImpl`. However:

1. The HLTH-02 guard in `computeRAU` / `computeRAUandHByA` checks `isAssembled()`. If a future
   caller passes a `pEqn` that was only `solve()`-ed (not `assemble()`-ed), the guard will
   throw a spurious `runtime_error` even though `ls_` is valid.
2. The original TODO comment at line 19–21 explicitly tracks this gap and it remains unresolved.

**Recommendation:**
Set `isAssembled_ = true` inside `solveImpl` after `iterativeSolveImpl` returns, so the flag
accurately reflects the state of `ls_`:

```cpp
NeoN::la::SolverStats solveImpl(dsl::Expression<ValueType>& expr, LinearSystem& ls)
{
    // ...
    stats = NeoN::dsl::detail::iterativeSolveImpl(...);
    isAssembled_ = true;   // <-- add this
    // ...
    return stats;
}
```

---

### [CRIT-03] Continuity error semantics diverge from icoFoam's reference values

**File:** `examples/neoIcoFoam/neoIcoFoam.cpp:149–182`
**Severity:** Critical

**Finding:**
The comment on line 149 claims the computation "match[es] icoFoam's exact format." It does not,
in three distinct ways.

**a) Missing `dt` factor.**
icoFoam (`continuityErrs.H`) multiplies by `runTime.deltaTValue()`:
```cpp
scalar sumLocalContErr = runTime.deltaTValue()
    * mag(contErr)().weightedAverage(mesh.V()).value();
```
neoIcoFoam omits `dt`. The reported numbers will differ by one to three orders of magnitude for
typical time steps (1e-4 – 1e-2 s).

**b) Missing volume weighting.**
icoFoam uses `weightedAverage(mesh.V())` — a volume-weighted mean. neoIcoFoam computes a plain
`sum`, so the reported value scales with the total number of cells, not cell volume.

**c) `sumLocal` is absolute; icoFoam's `global` is signed.**
icoFoam's `globalContErr` is `dt * contErr.weightedAverage(mesh.V())` — a *signed* value that
cancels for a globally conservative discretisation. neoIcoFoam uses `Kokkos::abs` for both
`sumLocal` and accumulates the absolute value into `cumulativeContErr`, so the cumulative total
can only grow, unlike icoFoam's which can fluctuate around zero.

**d) Log format: missing commas.**
icoFoam: `"sum local = X, global = Y, cumulative = Z"`. neoIcoFoam: `"sum local = {} global =
{} cumulative = {}"`. No commas after the value placeholders.

The numbers produced are dimensionally inconsistent with the icoFoam reference, making
cross-validation of the cylinder3D case (the milestone deliverable) impossible using the logged
continuity errors.

**Recommendation:**
Align with icoFoam's computation or document explicitly that this is a different metric. The
closest NeoN-native equivalent:

```cpp
// After pEqn solve, inside PISO loop:
const auto nCells = static_cast<NeoN::localIdx>(rt.nfMesh.nCells());
NeoN::dsl::Expression<NeoN::scalar> divExpr(rt.exec);
divExpr.addOperator(NeoN::dsl::exp::div(phi));
NeoN::Vector<NeoN::scalar> divPhi = divExpr.explicitOperation(nCells);

// Volume-weighted sums — match icoFoam's V-weighted average * dt
const auto cellVols = rt.nfMesh.cellVolumes().view();
auto divPhiView = divPhi.view();
NeoN::scalar volTotal = 0.0, sumLocalAbs = 0.0, sumGlobalSigned = 0.0;
NeoN::parallelReduce(rt.exec, {0, static_cast<size_t>(nCells)},
    NEON_LAMBDA(const size_t i, NeoN::scalar& vt, NeoN::scalar& sl, NeoN::scalar& sg) {
        vt += cellVols[i];
        sl += Kokkos::abs(divPhiView[i]) * cellVols[i];
        sg += divPhiView[i] * cellVols[i];
    }, volTotal, sumLocalAbs, sumGlobalSigned);
NeoN::mpi::allReduce(volTotal, NeoN::mpi::ReduceOp::Sum, rt.mpiEnvironment.comm());
NeoN::mpi::allReduce(sumLocalAbs, NeoN::mpi::ReduceOp::Sum, rt.mpiEnvironment.comm());
NeoN::mpi::allReduce(sumGlobalSigned, NeoN::mpi::ReduceOp::Sum, rt.mpiEnvironment.comm());

const NeoN::scalar sumLocal = rt.dt * sumLocalAbs / volTotal;
const NeoN::scalar globalErr = rt.dt * sumGlobalSigned / volTotal;
cumulativeContErr += globalErr;
NeoN::Logging::info(
    "time step continuity errors : sum local = {}, global = {}, cumulative = {}",
    sumLocal, globalErr, cumulativeContErr);
```

---

### [CRIT-04] Pre-existing "solve pEqn" test section retains old rank-0 guard that PISO-02 was meant to remove

**File:** `test/test_distributedPressureVelocityCoupling.cpp:336–339`
**Severity:** Critical

**Finding:**
The "solve pEqn" section (inside the "Distributed PressureVelocityCoupling" TEST_CASE) still
calls `pEqn.setReference(0, 0.0)` only when `rt.mpiEnvironment.rank() == 0`:

```cpp
if (rt.mpiEnvironment.rank() == 0)
{
    pEqn.setReference(0, 0.0);
}
```

The PISO-02 fix explicitly removed this rank-0 guard from `pdeSolver.hpp`. After the fix, the
intended pattern is `pRefCell >= 0` (set by OpenFOAM's `setRefCell`). The existing test section
contradicts the fix intent: it continues to demonstrate and validate the old, now-incorrect
pattern. This means:

1. The test does not verify that the new call-site pattern (`pRefCell >= 0`) works correctly for
   rank != 0.
2. Any regression that reintroduces the rank-0 guard in `SetReference::operator()` would still
   pass this section (it already only calls `setReference` on rank 0).
3. The test is inconsistent with `neoIcoFoam.cpp` which uses the corrected pattern.

**Recommendation:**
Update the "solve pEqn" section to use the canonical pattern established by the fix:

```cpp
// Match the neoIcoFoam.cpp call-site pattern — no rank guard,
// only the pRefCell >= 0 gate (set by OpenFOAM's setRefCell on owning rank).
Foam::label ofpRefCell = 0;
Foam::scalar ofpRefValue = 0.0;
Foam::setRefCell(ofp, mesh.solutionDict().subDict("PISO"), ofpRefCell, ofpRefValue);
if (ofpRefCell >= 0)
{
    pEqn.setReference(static_cast<NeoN::localIdx>(ofpRefCell), ofpRefValue);
}
```

---

## Warnings

### [WARN-01] Duplicate `SECTION("compute flux")` names cause Catch2 undefined behavior

**File:** `test/test_distributedPressureVelocityCoupling.cpp:243` and `264`
**Severity:** Warning

**Finding:**
Two distinct `SECTION` blocks inside the "Distributed PressureVelocityCoupling" `TEST_CASE`
share the identical name `"compute flux"`:

```cpp
SECTION("compute flux")   // line 243 — tests nf::flux(nfHbyA) vs fvc::flux(HbyA)
{
    ...
}

SECTION("compute flux")   // line 264 — tests updateFaceVelocity via pEqn
{
    ...
}
```

Catch2 v3 treats duplicate section names within a test case as the same section. The second
`SECTION` block may not be executed, or both blocks' results may be attributed to the first one
in the test report. This silently masks the second section's failures.

**Recommendation:**
Rename the second section to a unique name, e.g., `"compute flux: updateFaceVelocity"`.

---

### [WARN-02] PISO-02 test assumes rank 1 exists; silent singular system on single-rank builds

**File:** `test/test_distributedPressureVelocityCoupling.cpp:774–788`
**Severity:** Warning

**Finding:**
The new [PISO-02] `TEST_CASE` hard-codes rank 1 as the reference-cell owner:

```cpp
if (rt.mpiEnvironment.rank() == 1)
{
    pRefCell = static_cast<Foam::label>(mesh.nCells()) - 1;
}
```

`REQUIRE(Foam::Pstream::parRun())` at line 728 prevents the test from running in serial. However,
if the test binary is ever invoked with more than 2 ranks (e.g., on the cylinder3D 4-rank
decomposition), all ranks other than rank 1 leave `pRefCell = -1` and skip `setReference`. The
system remains constrained because rank 1 still pins the cell — this is fine. However, the
comment block (lines 725–727) explicitly states "rank 0 has pRefCell = -1" but says nothing about
ranks 2–3, which also have `pRefCell = -1`. The test may be fragile if the mesh changes so that
rank 1 has zero local cells (giving `mesh.nCells() - 1 = -1`, silently setting `pRefCell = -1`
on rank 1 too), resulting in a fully unconstrained singular system that the `numIter > 0`
assertion catches only by accident.

**Recommendation:**
Add a guard for the case where `mesh.nCells() == 0` on rank 1:

```cpp
if (rt.mpiEnvironment.rank() == 1 && mesh.nCells() > 0)
{
    pRefCell = static_cast<Foam::label>(mesh.nCells()) - 1;
}
```

And update the comment to acknowledge behavior on ranks > 1.

---

### [WARN-03] Large commented-out code block with `FIXME` / `NF_PING` left in `solveImpl`

**File:** `include/NeoFOAM/datastructures/pdeSolver.hpp:226–246`
**Severity:** Warning

**Finding:**
`solveImpl` contains a 20-line block of commented-out code preceded by `// FIXME` and including
a `// NF_PING();` debug call. The block describes an IC-preconditioner workaround that "will
produce -p as a result" — a sign-flip side-effect. It is unclear whether this is dead code to be
deleted or a pending feature. Its presence makes it difficult to determine which code path is
active and pollutes the header with partially-implemented logic.

**Recommendation:**
Either delete the block entirely (it is not compiled) and track the IC-preconditioner issue in a
bug report, or move it to a separate branch and merge when complete. Do not leave FIXME+commented
code in a public header.

---

### [WARN-04] Copyright identifier `nf authors` in two files should be `NeoFOAM authors`

**File:** `src/algorithms/pressureVelocityCoupling.cpp:2` and `examples/neoIcoFoam/neoIcoFoam.cpp:2`
**Severity:** Warning

**Finding:**
Both files carry:
```
// SPDX-FileCopyrightText: 2025 nf authors
```
The canonical identifier used elsewhere in the repository (e.g.,
`test/test_distributedPressureVelocityCoupling.cpp:2`) is `NeoFOAM authors`. The REUSE 6.2.0
compliance tooling enforced by the pre-commit hooks may flag this inconsistency, and it will
produce mismatched SBOM entries.

**Recommendation:**
Change both occurrences to:
```
// SPDX-FileCopyrightText: 2025 NeoFOAM authors
```

---

## Info

### [INFO-01] `TODO` in class-level doc comment is now partially resolved

**File:** `include/NeoFOAM/datastructures/pdeSolver.hpp:19–21`
**Severity:** Info

**Finding:**
The class doc comment contains:
```
* TODO: implement flag if matrix is assembled or not -> if not assembled call assemble
```
The `isAssembled_` flag introduced by HLTH-02 partially addresses this TODO. The remaining gap
(auto-calling `assemble()` when the flag is false) is not implemented. The comment should be
updated to reflect current state.

**Recommendation:**
Replace the TODO with a note describing what is implemented and what remains:
```
* NOTE: `isAssembled_` tracks whether the owned linear system has been assembled.
* Callers that depend on the assembled matrix (e.g., computeRAU) must call assemble()
* explicitly. Auto-assembly on demand is not yet implemented.
```

---

### [INFO-02] `epsilon` declared as `float` rather than `NeoN::scalar` / `double`

**File:** `test/test_distributedPressureVelocityCoupling.cpp:27–28`
**Severity:** Info

**Finding:**
```cpp
float epsilon = 1e-32;
float epsilonII = 1e-13;
```
Both variables are immediately passed into `ApproxScalar`/`ApproxVector` which accept
`Foam::scalar` (double). The implicit widening is safe here, but declaring them as
`NeoN::scalar` or `double` would be more idiomatic and remove the implicit conversion.

**Recommendation:**
```cpp
NeoN::scalar epsilon = 1e-32;
NeoN::scalar epsilonII = 1e-13;
```

---

### [INFO-03] Stale `TODO` in `pdeSolver.hpp` header file-level comment

**File:** `include/NeoFOAM/datastructures/pdeSolver.hpp:3`
**Severity:** Info

**Finding:**
Line 3 carries `// TODO: move to cellCenred dsl?` (also note the typo "cellCenred"). This is an
architectural consideration, not a code issue, but it is stale infrastructure noise in a public
header.

**Recommendation:**
Either track the architectural decision in the project backlog and remove the TODO from the
header, or leave a more specific note about the blocker preventing the move.

---

## Verdict

**PASS-WITH-WARNINGS**

Three critical issues were fixed immediately after review:
- ~~CRIT-01~~: **FIXED** — wrong function name in `computeRAUandHByA` error message
- ~~CRIT-02~~: **FIXED** — `isAssembled_ = true` added to `solveImpl` after `iterativeSolveImpl`
- ~~CRIT-03~~: **FIXED** — continuity block rewritten: dt factor, volume weighting, signed global, commas in format string
- CRIT-04: **NOT FIXED** — pre-existing "solve pEqn" test section still carries old rank-0 guard; not introduced by Phase 4 (the new [PISO-02] TEST_CASE correctly tests the fixed path on rank 1)

Two warnings were also fixed:
- ~~WARN-01~~: **FIXED** — duplicate `SECTION("compute flux")` renamed to `SECTION("update face velocity via pEqn")`
- ~~WARN-02~~: **FIXED** — added `mesh.nCells() > 0` guard to [PISO-02] test rank-1 branch

Remaining open items (not blocking):
- WARN-03: commented-out IC-preconditioner block in `solveImpl` — pre-existing NeoN-side code
- WARN-04: `SPDX-FileCopyrightText: 2025 nf authors` — pre-existing inconsistency, separate cleanup
- INFO-01/02/03: minor doc/style issues — no correctness impact

---

_Reviewed: 2026-05-12_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
_Post-review fixes applied: 2026-05-12_
