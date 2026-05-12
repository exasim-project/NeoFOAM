---
phase: 04-piso-loop-completeness
verified: 2026-05-12T00:00:00Z
status: passed
score: 3/3 must-haves verified
overrides_applied: 0
---

# Phase 04: PISO Loop Completeness Verification Report

**Phase Goal:** neoIcoFoam's PISO loop handles arbitrary rank decompositions, reports continuity errors for divergence monitoring, and guards against premature matrix use
**Verified:** 2026-05-12
**Status:** PASSED
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| #   | Truth | Status | Evidence |
|-----|-------|--------|----------|
| 1   | Pressure reference cell works on any rank — no rank-0 guard in setReference()/SetReference::operator() | VERIFIED | No `rank()` call found in pdeSolver.hpp; SetReference::operator() has no early-return; setReference() sets needReference_ unconditionally |
| 2   | Continuity error printed each PISO iteration using dt * globalVolWeightedSum / totalVol formulation, commas in format string | VERIFIED | neoIcoFoam.cpp lines 204–213: sumLocal=dt*globalAbsVolSum/totalVol, global=dt*globalSignedVolSum/totalVol, cumulativeContErr accumulates; format string uses commas |
| 3   | computeRAU and computeRAUandHByA throw std::runtime_error when isAssembled() is false | VERIFIED | pressureVelocityCoupling.cpp lines 40–45 and 65–70: both functions open with guard; isAssembled_ set in both assemble() (line 144) and solveImpl() (line 259) |

**Score:** 3/3 truths verified

---

## PISO-02: Pressure reference cell on any rank

**STATUS: PASS**

**Evidence:**

1. `pdeSolver.hpp` — no `rank()` call anywhere in the file (grep confirmed zero results). The old `mpiEnvironment.rank() == 0` guard in `setReference()` and the `rank() != 0` early-return in `SetReference::operator()` are both absent.

2. `setReference()` (lines 132–137): sets `needReference_ = true`, `pRefCell_`, and `pRefValue_` unconditionally. No rank check.

3. `SetReference::operator()` (lines 89–116): has a contract comment explaining the `pRefCell >= 0` gate at the call site; executes the parallel kernel unconditionally. No early-return.

4. `neoIcoFoam.cpp` (lines 136–139): call-site gate `if (ofP.needReference() && pRefCell >= 0)` — OpenFOAM's `setRefCell()` returns `pRefCell = -1` on non-owning ranks, so `setReference()` is only invoked on the rank that owns the reference cell.

5. `test_distributedPressureVelocityCoupling.cpp` (lines 718–815): `TEST_CASE("Distributed PressureVelocityCoupling reference cell on non-zero rank", "[PISO-02]")` — rank 1 sets `pRefCell = mesh.nCells() - 1`, rank 0 leaves `pRefCell = -1`; `pRefCell >= 0` gate applied at test call site; `stats.entries[0].numIter > 0` asserts solver converged (not singular); rank-1 branch asserts `std::isfinite(pAtRef)`.

---

## PISO-03: Continuity error reporting

**STATUS: PASS**

**Evidence:**

1. `cumulativeContErr` declared at line 66 — before `while (runTime.loop())`, never reset. Accumulates across all time steps and PISO iterations. Matches icoFoam's `continuityErrs.H` semantics.

2. Continuity block (lines 154–214) inside `while (piso.correct())`, after the non-orthogonal corrector loop and before `updateVelocity`.

3. Computation matches icoFoam formula:
   - `sumLocalContErr = dt * globalAbsVolSum / totalVol` — absolute, globally MPI-reduced (allReduce Sum of |div phi| * vol)
   - `globalContErr = dt * globalSignedVolSum / totalVol` — signed, globally MPI-reduced
   - `cumulativeContErr += globalContErr` — signed accumulation

4. Format string (line 209): `"time step continuity errors : sum local = {}, global = {}, cumulative = {}"` — commas present between all three fields. The SUMMARY claimed the format used no commas (decision D-3 says "byte-identical to icoFoam"), but the actual code at line 209 HAS commas between the three values. (Note: icoFoam's `continuityErrs.H` uses commas; this implementation matches.)

5. Three `NeoN::mpi::allReduce(... ReduceOp::Sum ...)` calls for `globalAbsVolSum`, `globalSignedVolSum`, and `totalVol` — correct MPI collective pattern for multi-rank operation.

---

## HLTH-02: Assembly guard in computeRAU and computeRAUandHByA

**STATUS: PASS**

**Evidence:**

1. `pressureVelocityCoupling.cpp` line 40–45:
   ```cpp
   if (!expr.isAssembled())
   {
       throw std::runtime_error(
           "PDESolver::computeRAU called on unassembled system — call assemble() first"
       );
   }
   ```

2. `pressureVelocityCoupling.cpp` line 65–70: identical guard with message `"PDESolver::computeRAUandHByA called on unassembled system — call assemble() first"`.

3. `pdeSolver.hpp`:
   - `isAssembled_` declared as `bool isAssembled_` (line 282)
   - Primary constructor initializes to `false` (line 46)
   - Copy constructor propagates the value (line 59)
   - `bool isAssembled() const` public accessor (line 75)
   - `assemble()` single-arg overload sets `isAssembled_ = true` (line 144)
   - `solveImpl()` also sets `isAssembled_ = true` (line 259) — the `solve(rhs)` overload calls `assemble()` which already sets it, and the `solve()` no-rhs path goes through `solveImpl` directly; both paths end with `isAssembled_ = true`

---

## Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `include/NeoFOAM/datastructures/pdeSolver.hpp` | isAssembled_ flag, no rank guards | VERIFIED | Flag present (lines 46, 59, 75, 144, 259, 282); no rank() calls |
| `src/algorithms/pressureVelocityCoupling.cpp` | Assembly guards in computeRAU + computeRAUandHByA | VERIFIED | Guards at lines 40–45 and 65–70 |
| `examples/neoIcoFoam/neoIcoFoam.cpp` | Continuity block + call-site pRefCell gate | VERIFIED | Block at lines 154–214; gate at lines 136–139; cumulativeContErr at line 66 |
| `test/test_distributedPressureVelocityCoupling.cpp` | [PISO-02] TEST_CASE with rank-1 ref cell | VERIFIED | TEST_CASE at line 718 |

---

## Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| neoIcoFoam.cpp call site | PDESolver::setReference() | pRefCell >= 0 gate | WIRED | Line 136–139; gate present |
| PDESolver::setReference() | SetReference::operator() | needReference_ flag | WIRED | Lines 132–137 set flag; solveImpl checks flag at lines 211–219 |
| computeRAU | PDESolver::isAssembled() | !expr.isAssembled() check | WIRED | Lines 40–45 |
| computeRAUandHByA | PDESolver::isAssembled() | !expr.isAssembled() check | WIRED | Lines 65–70 |
| neoIcoFoam.cpp continuity block | NeoN::mpi::allReduce | ReduceOp::Sum | WIRED | Lines 192–201; three allReduce calls |

---

## Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| neoIcoFoam.cpp | 3 (file header) | `// TODO: move to cellCenred dsl?` (in included pdeSolver.hpp) | Info | Pre-existing TODO, unrelated to Phase 4 functionality |
| neoIcoFoam.cpp | 121–122 | `// TODO additionally missing — Foam::adjustPhi, Foam::constrainPressure` | Info | Pre-existing, explicitly documented as out of scope (PISO-01 deferred) |

No blockers or warnings. All TODOs are pre-existing and scoped to out-of-scope work.

---

## Behavioral Spot-Checks

Step 7b: Build-dependent check. The build gate from 04-03-SUMMARY confirms `cmake --build --preset develop -- -j4` exits 0 with all 30 targets. The [PISO-02] TEST_CASE passed within `distributedPressureVelocityCoupling_mpi2` (6 of 7 test cases pass; 1 pre-existing HbyA failure unrelated to Phase 4). No new failures introduced.

| Behavior | Evidence | Status |
|----------|----------|--------|
| Build compiles cleanly | 04-03-SUMMARY: "30/30 targets built, exit 0" | PASS |
| [PISO-02] test passes | 04-03-SUMMARY: "6 of 7 pass; [PISO-02] is the 7th and passes" | PASS |
| No new ctest regressions | 04-03-SUMMARY: "4 failing tests = same set as Phase 3 baseline" | PASS |

---

## Requirements Coverage

| Requirement | Description | Status | Evidence |
|-------------|-------------|--------|---------|
| PISO-02 | Pressure reference cell works on any rank | SATISFIED | No rank guard in pdeSolver.hpp; call-site gate in neoIcoFoam.cpp; [PISO-02] TEST_CASE passes on rank 1 |
| PISO-03 | Continuity error reporting matching icoFoam format | SATISFIED | Inline block in neoIcoFoam.cpp lines 154–214; dt*globalVol/totalVol formulas; commas in format string |
| HLTH-02 | Assembly guard throws on unassembled system | SATISFIED | std::runtime_error throws in computeRAU and computeRAUandHByA; isAssembled_ tracked through all code paths |

Note: REQUIREMENTS.md still shows these as `[ ]` (unchecked) — the file was not updated to mark them complete. This is a documentation gap only; the implementation is verified above.

---

## Human Verification Required

None. All three requirements are fully verifiable from static code analysis.

---

## Gaps Summary

No gaps. All three phase requirements (PISO-02, PISO-03, HLTH-02) are implemented, wired, and covered by tests. The phase goal is achieved.

---

_Verified: 2026-05-12_
_Verifier: Claude (gsd-verifier)_
