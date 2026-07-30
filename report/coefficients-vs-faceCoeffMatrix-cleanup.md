<!--
SPDX-FileCopyrightText: 2026 NeoN authors

SPDX-License-Identifier: MIT
-->

# `coefficients.hpp` vs `faceCoeffMatrix.hpp` — what is legacy, what can collapse

Read-only audit of

* `src/NeoN/include/NeoN/blockAmr/linearAlgebra/coefficients.hpp`
* `src/NeoN/include/NeoN/blockAmr/linearAlgebra/faceCoeffMatrix.hpp`

All paths below are relative to `src/NeoN/` unless they start with `report/`.
Every claim is either **[V] verified in code** (a `file:line` I read) or
**[I] inferred** (reasoning over what I read, flagged as such).

**Tree state matters for this report.** The submodule working tree is dirty: the
`PROTOTYPE (C1)` change that stops `FaceCoeffOp` from reading a stored diagonal is
**uncommitted** (`git diff -- src/blockAmr/linearAlgebra/matrixFree/faceCoeffOp.cpp`,
hunks at `faceCoeffOp.cpp:127-133`, `:172-178`, `:354-358`). At `HEAD`
(`d039eb9777`) the stored diagonal *was* consumed by both stencils. So several
findings below read "dead in the working tree, live at HEAD". [V]

---

## Verdict 1 — "there should only be a coefficient matrix for matrix free"

**Mostly holds, but not as a pure deletion: `CsrMatrix` is a test instrument, not a
production format — and deleting it costs three specific test capabilities and
makes two pieces of `coefficients.hpp` unreachable.** `CsrMatrix`
(`faceCoeffMatrix.hpp:449-572`) has no non-test C++ caller anywhere: its only
constructions are `ginkgoSolve.cpp:670-671` (the `_la_matrix_*`/`_la_system_*`
test seams, whose own header comment says they exist "to make … reachable from
pytest, which is the only place this component is testable at all",
`ginkgoSolve.cpp:600-613`) and `ginkgoSolve.cpp:1481,1488` (`Matrix.csr_symmetric`
/ `csr_asymmetric`, surfaced in Python as `linear_algebra.CsrMatrix`,
`python/blockamr/linear_algebra.py:121-151`). It is reachable from the Python API,
but nothing in the shipped Python uses it: the only importers of
`blockamr.linear_algebra` are `test/blockAmr/test_la_python_api.py:56` and
`benchmarks/blockAmr/bench_solvers2.py:120`, and the benchmark builds
`MFFaceCoeffs` only (`bench_solvers2.py:226`) — `bench_solvers.py`'s `"csr"` row is
the *legacy* `FaceCoeffCsrSolver` facade (`bench_solvers.py:161`), not this format.
[V, greps: `grep -rn "CsrMatrix" --include=*.cpp --include=*.hpp --include=*.py .`
and `grep -rn "linear_algebra" --include=*.py src/blockAmr/python benchmarks test`]
It delivers **no numerical capability the matrix-free path lacks**: there is no
direct/sparse factorization anywhere in `blockAmr`
(`grep -rn "matrix::Csr\|matrix::Coo\|Lu\|Cholesky" include/NeoN/blockAmr src/blockAmr`
returns only `sparse/csr.{hpp,cpp}`, the `dynamic_pointer_cast` in the probe
binding at `ginkgoSolve.cpp:1260`, and unrelated matches), the GMG bottom solve is
matrix-free, and `CsrMatrix` is *weaker* on preconditioning — it declines `gmg` and
`gmg_kokkos` outright (`faceCoeffMatrix.hpp:518-523`). **But** the hypothesis's
wider form — "the whole `sparse/` CSR assembly path may be legacy" — is **false**:
`assembleFaceCoeffCsr` has a second, heavily tested caller in the legacy facade
`FaceCoeffCsrSolver` (`solve/persistent.cpp:1010`, class at
`solve/persistent.hpp:163`), exercised by `test_ginkgo_bc.py` §3
(`:349-358,:784,:847,:904-905,:934`), `test_ginkgo_controls.py:140,211,225-242`,
`test_ginkgo_singular.py:142`, `test_ginkgo_gmg.py:246` and
`test_ginkgo_gmg_generality.py:1168-1174`. `sparse/` stays either way. [V]

## Verdict 2 — "`coefficients.hpp` is more polished but lacks some features"

**Holds on "polished"; the "missing features" are not stubs but *unconsumed
surface* — nine of the ten `IsMatrix` requirements exist, but four are reached
only from test bindings, and one documented invariant is not actually enforced by
the types.** Nothing in `coefficients.hpp` is declared-and-never-defined: every
member is inline and every concept requirement is satisfied by both formats
(`faceCoeffMatrix.hpp:574-575`) and by the stubs in
`src/blockAmr/linearAlgebra/coefficientsConcepts.cpp:36-52`. [V] The gaps are:
(1) `Coefficients::symmetric()` (`coefficients.hpp:105`) has **zero** callers;
(2) `MatrixCoefficients::symmetric()` (`:77`) has exactly one, the probe binding
`ginkgoSolve.cpp:1081`; (3) `IsMatrix::isAssembled()` (`:145`) and
`IsMatrix::symmetry()` (`:154`) are consumed only by test bindings — no production
code branches on either; (4) `IsMatrix::name()` (`:166`) is reached from exactly
one place, `solver.hpp:76`, and only when a format declines a preconditioner,
which only `CsrMatrix` ever does; (5) `IsOperator` (`:172-177`) is satisfied by
exactly **one** real type, `ops::Laplacian` (`operators/laplacian.hpp:114`); and
(6) the header's claim that "diag, upper and rhs are **non-nullable by
construction**, so a format cannot express handing back a missing one"
(`coefficients.hpp:37-40`) is a convention, not a type property —
`CellFieldLevel`/`FaceFieldLevel` are aggregates over `shared_ptr`
(`core/fieldLevel.hpp:27-47`) and an empty one is constructible and *already
constructed elsewhere in this codebase* (`matrixFree/faceCoeffOp.hpp:108`,
`const CellFieldLevel& diag = {}`). The `static_assert`s that claim to pin this
(`coefficientsConcepts.cpp:161-167`) only check that the member *types* are not
`optional`. [V]

---

## Overlap / duplication between the two files

| Responsibility | `coefficients.hpp` | `faceCoeffMatrix.hpp` | Belongs in |
|---|---|---|---|
| `mesh/diag/upper/lower` field list | `MatrixCoefficients:70-78` | — | coefficients.hpp |
| The same four fields **again** | `Coefficients:93-96` | — | coefficients.hpp (but as *one* declaration — see S1) |
| `symmetric()` predicate | twice, `:77` and `:105`, identical bodies | `FaceCoeffFields::coefficients():197` re-derives it from `sym` | coefficients.hpp, once |
| "interface `lower` vs stored `lower`" doctrine | `:56-61` | `:104-109`, `:206-217`, `:302-305`, `:365-367` | one place; currently stated 5x |
| negSumDiag / "diag is the SOURCE" doctrine | `:50-51` | `:179-193`, `:316-336` | faceCoeffMatrix.hpp (it owns the derived field) |
| Executor plumbing rationale | `:98-103` | `:113`, `:416`, `:552` | coefficients.hpp |

The **structural** duplication is one item: `Coefficients` re-declares
`MatrixCoefficients`' four members and its predicate rather than holding one
(`coefficients.hpp:93-96` + `:105` vs `:72-77`), and the ctor then copies them
member-by-member (`:111-113`). Everything else in the table is *comment*
duplication, which the concurrent comment pass may already be reducing.

---

## `faceCoeffMatrix.hpp` cache audit — which caches are consumed?

`Assembly {csr, dirty}` (`:560-564`) — **consumed and load-bearing.** Read at
`:474-497` (`CsrMatrix::op()`), invalidated at `:538` and `:544`; pinned by
`test_la_matrix_formats.py::test_csr_assembles_once_until_written` (`:346-359`) and
`::test_op_before_write_does_not_freeze_the_matrix` (`:243`), via the
`assemble_before_write` / `via_copy` knobs at `ginkgoSolve.cpp:988-1000`. [V]

`Diagonal {diag, dirty}` (`:421-425`) — **not consumed by anything that computes a
result, in the working tree.** Chain, verified end to end:

* `MFFaceCoeffs::diagonal()` (`:338-352`) is the only reader of `state_->diag` and
  the only caller of `computeFaceCoeffDiag`
  (`grep -rn "computeFaceCoeffDiag"` → decl `matrixFree/faceCoeffOp.hpp:40`, def
  `matrixFree/faceCoeffOp.cpp:21`, one call site `faceCoeffMatrix.hpp:342`). [V]
* `diagonal()`'s only callers are `ginkgoSolve.cpp:1360,1382,1387` — the
  `_la_stored_diagonal` **test binding** (`:1318-1413`), consumed by
  `test/blockAmr/test_la_stored_diagonal.py` (4 tests, `:160,:180,:208,:242`).
  `grep -rn "diagonal" src/blockAmr include/NeoN/blockAmr` finds no other call. [V]
* `MFFaceCoeffs::op()` still *passes* the field (`:312`, `CellFieldLevel
  {state_->diag}`), but `FaceCoeffOp`'s constructor discards it —
  `faceCoeffOp.cpp:175-178`: `(void)diag; const amrex::MultiFab* diagField =
  &(*alpha);` — and both stencils recompute `alpha - sum(faces)` inline
  (`faceCoeffOp.cpp:130-133`, `:357-358`). The refresh call that used to sit at the
  top of `op()` is now the comment at `faceCoeffMatrix.hpp:292-293`. [V]

So: **half-finished migration, currently mid-experiment, not settled dead code.**
At `HEAD` the stencils read the stored field; the working tree reverts them to the
inline derivation and leaves the storage, the flag, the refresh function and its
one allocation in place. Collateral in the same working tree: `FaceCoeffOp`'s
`diagOwned_` member (`matrixFree/faceCoeffOp.hpp:173`) is now assigned nowhere and
only `.reset()`-ed (`faceCoeffOp.cpp:218`); the `diag` constructor parameter
(`faceCoeffOp.hpp:108`) and its documented "empty handle means compute it here"
contract are inert. [V] Cost while it sits there: **one extra cell-centred
`MultiFab` allocated per `MFFaceCoeffs`** (`faceCoeffMatrix.hpp:432-434`) that no
solve reads. [V]

---

## Ranked work-list

### (A) Safe to delete now

**A1 — `Coefficients::symmetric()` (`coefficients.hpp:105`).**
Reachability set: **empty.** C++ callers: none. Bindings: none. Python: none.
Tests: none. Benchmarks: none. Grep run:
`grep -rn "symmetric()" src/blockAmr include/NeoN/blockAmr test benchmarks | grep -v "MFFaceCoeffs::\|CsrMatrix::\|Matrix.mf_\|Matrix.csr_\|def symmetric\|bool symmetric() const"`
→ the only live hit is `ginkgoSolve.cpp:1081`, and that `c` is
`matrix.coefficients()`, i.e. a `MatrixCoefficients` (`ginkgoSolve.cpp:1072`), so
it exercises `:77`, not `:105`. Nothing breaks. [V]

**A2 — the `sym == Symmetry::asymmetric` branch in `FaceCoeffFields::zero()`
(`faceCoeffMatrix.hpp:226-231`).** When symmetric, `lower[d]` *is* `upper[d]`
(`:163`), so `lower[d]->setVal(0.0)` unconditionally is correct in both regimes —
merely a redundant second `setVal` on an already-zeroed fab. Deleting the branch
removes 5 lines and one of the three `sym` tests. Behaviour-identical. [V for the
aliasing; I for "no test observes the redundant write" — `zero()` is observed only
through values, `test_la_linear_system.py::test_system_zero_clears_coefficients_and_rhs:387`
and `_probe`'s `op_rebuilt_after_zero`.]

**A3 — the default member initialisers that no path can observe.**
`Coefficients::exec {NeoN::SerialExecutor {}}` (`coefficients.hpp:103`): the sole
constructor (`:111-113`) always sets `exec`, and it is private with one friend, so
no default-constructed `Coefficients` exists. Same for
`FaceCoeffFields::exec/bc/sym` (`faceCoeffMatrix.hpp:113,124,125`) — `make()`
assigns all three (`:139-141`) and is the only producer (`:271,:279,:457,:465`).
Cosmetic; group with A1 rather than as its own change. [V]

### (B) Deletable after a decision

**B1 — `CsrMatrix` and its bindings.**
*Decision needed:* do we still want a second `IsMatrix` implementation *at all*?
The erasure (`matrix.hpp`), the `IsMatrix` concept, `name()`, `isAssembled()` and
the null-means-decline protocol are all justified in the comments **by there being
two formats**; with one, `Matrix` is a one-implementation abstraction.

Full reachability set:
* C++: `ginkgoSolve.cpp:670-671` (`makeLaMatrix`, `format=="csr"`),
  `:1481`, `:1488` (`Matrix.csr_symmetric/_asymmetric`), `:1260-1269`
  (`report_structure`'s `dynamic_pointer_cast` to `gko::matrix::Csr`),
  `coefficientsConcepts.cpp:22` (includes the header; its `static_assert`s are on
  the stubs, not on `CsrMatrix`), `faceCoeffMatrix.hpp:575` (`static_assert`).
* nanobind: `Matrix.csr_symmetric`, `Matrix.csr_asymmetric`; plus the `format`
  string `"csr"` accepted by `_la_matrix_solve`, `_la_matrix_probe`,
  `_la_system_solve`, `_la_system_probe`.
* Python API: `linear_algebra.CsrMatrix` (`linear_algebra.py:121-151`), exported in
  `__all__` (`:71`) and documented in the module docstring (`:8,:34`) and in
  `solver_config.py:136`.
* Tests: **4 functions would be deleted** —
  `test_la_boundary_conditions.py::test_the_two_formats_agree_through_the_laplacian`
  (`:412`, pure cross-check),
  `::test_csr_boundary_rows_drop_the_wraparound_column` (`:578`),
  `test_la_matrix_formats.py::test_csr_assembles_once_until_written` (`:346`),
  `test_la_python_api.py::test_csr_declines_gmg_naming_the_format_and_the_precond`
  (`:326`); **12 functions would lose a parametrisation** —
  `test_la_matrix_formats.py:191,222,242,315`,
  `test_la_linear_system.py:251,282,406`,
  `test_la_boundary_conditions.py:358(one of three cases),447,509`,
  `test_la_python_api.py:176,190`.
* Benchmarks: **none** (see Verdict 1).

What would actually be lost, separated by role:
1. *Testing role (cheap to lose):* the cross-format agreement check
   (`test_la_boundary_conditions.py:412`) and the 12 parametrisations. The file's
   own comment already says the agreement test "sees nothing" for the BC
   arrangement it sits next to (`faceCoeffMatrix.hpp:71-75`). [V]
2. *Real capability (must be replaced):* the **S6a variable-row-length guard**
   (`test_la_boundary_conditions.py:578-634`). It is the only assertion on
   `csr.cpp`'s `side()` dropping the boundary column, it is invisible to every
   solve- and coefficient-level check (docstring `:581-587` records that removing
   `bc` from `CsrMatrix` once left the whole suite green), and it reaches the row
   structure only through `CsrMatrix::op()` +
   `_la_system_probe(report_structure=True)` (`ginkgoSolve.cpp:1252-1279`). The
   legacy `FaceCoeffCsrSolver` builds the same CSR (`persistent.cpp:1010`) but
   exposes no row pointers — `grep -rn "csr_row_ptrs"` finds only the probe. So
   deleting `CsrMatrix` requires a small new binding on the legacy path first, or
   the guard dies with it. [V]
3. *Protocol coupling (decide explicitly):* `CsrMatrix` is the **only** format
   that ever returns null from `makePrecond` as a *decline* (`:518-523`);
   `MFFaceCoeffs` never declines (`:373-392`, and `precond.cpp:226-228` shows it
   *throws* instead). Remove it and `solver.hpp:73-80`'s raise and
   `IsMatrix::name()` (`coefficients.hpp:166`) become unreachable, together with
   the ~25 lines of `coefficients.hpp:118-138` that specify the protocol, and
   `test_la_python_api.py:326`'s error-message contract. [V]

**B2 — the whole stored-diagonal machinery** (`faceCoeffMatrix.hpp:316-352`,
`:399-404` `dirty`, `:406-410`, `:420-425`, `:428-434`, `:312`, plus
`computeFaceCoeffDiag` at `faceCoeffOp.hpp:40-46` / `faceCoeffOp.cpp:21-59`, the
`diag` ctor parameter `faceCoeffOp.hpp:108`, and the `diagOwned_` member
`faceCoeffOp.hpp:173` / `faceCoeffOp.cpp:218`).
*Decision needed:* **is PROTOTYPE (C1) the destination or a detour?** These two
answers give opposite work:
* If C1 stays (stencils recompute inline), the machinery is dead weight: delete it
  all, plus the `_la_stored_diagonal` binding (`ginkgoSolve.cpp:1314-1413`), its
  stub (`ginkgoSolveStub.cpp:67`) and `test_la_stored_diagonal.py` (258 lines,
  4 tests) — and drop the `MultiFab` allocated per matrix at `:432-434`.
* If C1 is reverted, everything stays exactly as written and the *only* change
  needed is re-adding `diagonal()`'s refresh call at the top of `op()`
  (`:292-293`), which HEAD had.
*Affects:* whoever owns C1 (uncommitted, so a person, not a merged decision), plus
`test_la_stored_diagonal.py`'s author. Note the tests still **pass** under C1
because the binding calls `diagonal()` directly — the machinery currently tests
itself and nothing else. Do not delete the tests without deleting the feature. [V]

**B3 — `IsMatrix::isAssembled()` (`coefficients.hpp:145`) and
`IsMatrix::symmetry()` (`:154`), and with them `enum class Symmetry`'s appearance
on the erasure.** No production consumer:
`grep -rn "isAssembled" src include` → concept `:145`, the two format
implementations (`faceCoeffMatrix.hpp:396,529`), the `Matrix` forwarders
(`matrix.hpp:63,96,114`), the four concept stubs, and five *binding* reads
(`ginkgoSolve.cpp:1011,1068,1165,1247,1493`). `symmetry()` identical
(`ginkgoSolve.cpp:1013,1069,1167,1249,1498`). Both are pure introspection for
pytest (`is_assembled`, `symmetric`, `is_symmetric`). *Decision:* is a
format-introspection surface part of the contract, or an artifact of "blockAmr has
no C++ test target" (`ginkgoSolve.cpp:600-606`)? `isAssembled()` in particular is
meaningless once B1 lands — it would be `false` for every matrix that exists.
`FaceCoeffFields::sym` stays regardless: it drives allocation (`:155-164`). [V]

**B4 — `Coefficients`' duplicated field list.** Fold `MatrixCoefficients` in
rather than restating it: see S1 below. *Decision:* whether the readability of a
flat `c.diag` / `c.upper` at the operator call site (`laplacian.cpp:258,271,301`)
is worth four duplicated declarations. If yes, keep and delete only `:105` (A1).

### (C) Keep — the hypothesis does not hold here

**C1 — `sparse/csr.cpp` + `sparse/csr.hpp`.** Second caller
`FaceCoeffCsrSolver` (`persistent.cpp:946-1010`), bound at `ginkgoSolve.cpp:956`
and covered by six test modules (list in Verdict 1). Not legacy in the removable
sense; only its *`la::` wrapper* is a candidate. [V]

**C2 — `MatrixCoefficients::mesh` / `Coefficients::mesh`
(`coefficients.hpp:72,93`).** Newly added in this working tree and genuinely
consumed: `laplacian.cpp:240` (`c.mesh.fillHalo`), `:242` (`geom.Domain()`),
`:243` (`dx()`). Guarded at `coefficientsConcepts.cpp:176-177`. [V]

**C3 — `Coefficients::exec` (`:103`).** Consumed at `laplacian.cpp:304,321,360`. [V]

**C4 — `friend class LinearSystem` + private ctor (`:109,:111`).** The friend is
used: `linearSystem.hpp:45` is the only construction of a `Coefficients`, and
`Operator::assemble` is private with the same friend (`operator.hpp:63-65`), so the
gate is real. **But the assertion that guards it is vacuous** — see S4. [V]

**C5 — `MatrixCoefficients::symmetric()` (`:77`) and the
interface-vs-storage `lower` split (`:75` / `faceCoeffMatrix.hpp:218`).** One live
caller (`ginkgoSolve.cpp:1081`) and one pinning test
(`test_la_matrix_formats.py:333-343`); the split is load-bearing for
`op()`/`makePrecond`, which need the *aliased* low side
(`faceCoeffMatrix.hpp:305,376`) while operators must not see one
(`laplacian.cpp:301`). Asymmetric storage itself is exercised only by tests today
(no convection operator exists), but `MFFaceCoeffs::asymmetric` is public Python
API (`linear_algebra.py:112-118`) and the honest read is "capability awaiting its
operator", not dead code. [I]

**C6 — `IsOperator` (`:172-177`) despite having one implementation.** It is what
keeps the dependency direction `ops -> la` (comment `:170-171`) and it is
*negatively* asserted in three places to prove the `system += op` gate
(`coefficientsConcepts.cpp:123,127,140`). Deleting it would delete the gate's
proof. [V]

---

## Simplifications that remove no feature

**S1 — `Coefficients` holds a `MatrixCoefficients` instead of restating it.**
Replaces four declarations + one predicate + a five-member ctor
(`coefficients.hpp:93-96,105,111-113`) with one member. Costs: call sites become
`c.m.diag` (or the type gains public inheritance from `MatrixCoefficients`, which
keeps `c.diag` working and makes A1 automatic since the predicate is inherited);
`coefficientsConcepts.cpp:164-167,177` would need retargeting. **Note the risk:**
public inheritance makes `Coefficients` convertible to `MatrixCoefficients&`, which
would break the discrimination `coefficientsConcepts.cpp:115-117,123` relies on
(`StubOperatorWrongArgument` must *not* satisfy `IsOperator`). Prefer composition,
or leave B4 undecided. [V for the line counts; I for the inheritance hazard]

**S2 — one `dirty` flag instead of two shapes.** `MFFaceCoeffs` and `CsrMatrix`
carry byte-identical bookkeeping under different struct names —
`Diagonal{ptr,dirty}` (`:421-425`) and `Assembly{ptr,dirty}` (`:560-564`), each
with the same `coefficients()`/`zero()` invalidation pair (`:400-410` vs
`:536-546`) and the same shared_ptr-sharing rationale stated twice (`:420`,
`:556-559`). If B1 or B2 lands, one of the two disappears and this evaporates; if
neither lands, a single `detail::Cached<T>` would remove ~20 duplicated lines.
Flagged, not recommended — the repo's CLAUDE.md forbids abstraction for
single-use code, and after B1/B2 this *is* single-use. [V]

**S3 — `MFFaceCoeffs::op()`'s dead argument.** `:312` passes
`CellFieldLevel {state_->diag}` to a constructor that discards it
(`faceCoeffOp.cpp:177`). Whichever way B2 goes, this line is currently a lie.
`FaceCoeffOp`'s `diag` parameter already defaults to `{}`
(`faceCoeffOp.hpp:108`), so dropping the argument is a one-line change. [V]

**S4 — the privacy guard at `coefficientsConcepts.cpp:147` is vacuous and should
be fixed, not deleted.**
`static_assert(!std::is_constructible_v<Coefficients, MatrixCoefficients, CellFieldLevel>)`
tests a **two**-argument construction, but the constructor takes **three**
(`coefficients.hpp:111`, `const NeoN::Executor&`, no default). So the assertion
holds for the wrong reason — the two-arg form does not exist regardless of access
— and would still pass if the constructor were made public. It went stale when
`exec` was added. Correct form adds the third argument. This is the one place in
the audit where a *guard* is broken rather than a feature missing. [V for the
signature mismatch; I for "it would pass if made public" — sound but not compiled]

**S5 — no template parameter is over-generalised, and two are under-exercised.**
`Matrix::Model<M>` is instantiated at exactly two types today
(`faceCoeffMatrix.hpp:574-575`) and would be **one** after B1;
`Operator::Model<T>` is already instantiated at exactly one
(`ops::Laplacian`, `ginkgoSolve.cpp:1152,1238,1528`). Both erasures would then be
one-implementation abstractions. I am *not* recommending their removal — the
`Operator` erasure earns its keep as the privacy gate (C6) — but the maintainer
should know that B1 turns `Matrix` into an erasure over a single type. [V]

**S6 — `FaceCoeffFields::make()` could be aggregate initialisation.** It
default-constructs then assigns five members (`:137-141`) before the loop; with
A3's defaults gone it reads as one initialiser plus the allocation loop. Cosmetic,
same instruction count. [I]

**S7 — nothing in either file wants to become a local.** I looked for the pattern:
`FaceCoeffFields::geom` must be stored (not derivable from a `MultiFab`, `:118-122`
— verified true), `bc` is read per `op()` (`:310`, `:492`), `sym` drives allocation
(`:155`), and `makePrecond`'s `upper`/`lower` are *already* locals for the
`&upper[0]` address-of (`:375-376`). The only member that could stop existing is
`Diagonal::diag` — under B2. [V]

---

## Open questions for the maintainer

1. **Is PROTOTYPE (C1) the destination?** Everything in B2 hinges on it, it is
   uncommitted, and the answer flips the work from "delete ~90 lines + a test
   module" to "restore one call at `faceCoeffMatrix.hpp:292`".
2. **Do we keep a second `IsMatrix` implementation?** If not, `Matrix`,
   `IsMatrix`, `name()`, `isAssembled()` and the decline protocol are an erasure
   over one type. Deleting `CsrMatrix` while keeping them is coherent (a slot for
   a future format); deleting both is coherent; keeping `CsrMatrix` purely as the
   cross-check twin is also coherent — but the code should say which, since
   `coefficients.hpp:22-40` and `faceCoeffMatrix.hpp:32-45` currently argue at
   length for two formats.
3. **Before deleting `CsrMatrix`, where does the S6a row-structure guard go?** It
   needs a row-pointer binding on `FaceCoeffCsrSolver`, or it is lost. That is the
   only capability in the whole audit that `CsrMatrix` uniquely provides.
4. **`FaceCoeffOp`'s `diagOwned_`/`diag` parameter and the "empty handle means
   compute it here" contract** are inert in the working tree. Is
   `matrixFree/faceCoeffOp.hpp` in scope for the same cleanup, or is it owned by
   the C1 experiment?
5. **`MFFaceCoeffs::makePrecond` throws with the wording `"FaceCoeffSolver: …"`**
   (via `precond.cpp:228`) — a legacy name leaking out of a format that is not
   that solver. Worth a message fix, or is the wording a pinned contract? (I did
   not find a test asserting that exact prefix for the `la::` path.)
