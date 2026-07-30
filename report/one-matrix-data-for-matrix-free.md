<!--
SPDX-FileCopyrightText: 2026 NeoN authors
SPDX-License-Identifier: MIT
-->

# One matrix data structure for the matrix-free path

Line numbers are as of writing; two other agents are editing comments in the same
tree, so they drift.

Status: steps 1–3 landed. Every touched translation unit was compiled
`-fsyntax-only`-style (single-TU, object to `/tmp`, no build tree written); see
"Verification" for the list. No test was changed — nothing in `test/` names the C++
vocabulary that moved.

## 1. What changed

### `detail::FaceCoeffFields` now IS a `MatrixCoefficients`

`include/NeoN/blockAmr/linearAlgebra/faceCoeffMatrix.hpp`

Before: `amrex::Geometry geom`, `Symmetry sym`, `shared_ptr<MultiFab> alpha`,
`array<shared_ptr<MultiFab>,3> upper, lower` — the seven-field matrix in a second
vocabulary, bypassing `CellFieldLevel`/`FaceFieldLevel`/`MeshLevel` entirely, and
`make()` accepting a `MeshLevel` only to discard it into a bare `geom`.

After:

```cpp
struct FaceCoeffFields
{
    NeoN::Executor exec {NeoN::SerialExecutor {}};
    la::BcArray bc {};
    MatrixCoefficients mc;   // mesh + diag + upper + optional lower
    ...
};
```

So step 2 landed as well: the struct *holds* a `MatrixCoefficients` rather than
restating its fields, and `coefficients()` is `return mc;`.

Consequences, all behaviour-neutral:

- **`Symmetry sym` is gone.** `mc.lower == nullopt` *is* the symmetry.
  `symmetry()` derives it (`mc.symmetric() ? symmetric : asymmetric`), so the two
  can no longer disagree. The `Symmetry` enum itself stays — it is `IsMatrix`'s
  return type and five binding sites compare against `Symmetry::symmetric`.
- **A symmetric matrix no longer allocates an aliased `lower` array.** It used to
  store `lower[d] = upper[d]`; now it stores nothing and `storedLower()` returns
  `mc.lower.value_or(mc.upper)`. The `shared_ptr`s handed to `FaceCoeffOp`,
  `assembleFaceCoeffCsr` and the GMG hierarchy are the *same objects* as before, so
  every consumer sees an identical aliasing.
- **`mesh()` returns `const MeshLevel&`** off `mc` instead of rebuilding
  `MeshLevel{alpha->boxArray(), alpha->DistributionMap(), geom}` per call. The
  header's old argument against storing `ba`/`dm` ("a second copy nothing keeps in
  step with alpha's") no longer applies: `make()` allocates every field *from*
  `mc.mesh` and nothing repoints them, so it is the single source, not a copy.
- `globalRows()` reads `mc.mesh.ba.numPts()` instead of
  `alpha->boxArray().numPts()` — the same `BoxArray`, `alpha` having been allocated
  from it.

### The `alpha` + six-loose-`MultiFab` signatures

Three of the four converted. Each now takes
`(… const CellFieldLevel& alpha, const FaceFieldLevel& upper, const FaceFieldLevel& lower, const MeshLevel& mesh, …)`,
with `mesh` replacing the bare `geom` and (in `buildGmgHierarchy`) the
`alpha->boxArray()/DistributionMap()` re-derivation:

| signature | file:line | call sites updated |
| --- | --- | --- |
| `buildGmgHierarchy` | `include/.../linearAlgebra/precond.hpp:45` | 3 — `precond.cpp:121`, `solve/persistent.cpp:425`, `solve/persistent.cpp:688` |
| `makeFaceCoeffPrecond` | `include/.../linearAlgebra/precond.hpp:70` | 2 — `faceCoeffMatrix.hpp:336`, `solve/persistent.cpp:709` |
| `assembleFaceCoeffCsr` | `include/.../linearAlgebra/sparse/csr.hpp:35` | 2 — `faceCoeffMatrix.hpp:429`, `solve/persistent.cpp:1012` |

`precond.hpp` and `sparse/csr.hpp` gained `core/fieldLevel.hpp` +
`core/meshLevel.hpp` includes (neither had them; `precond.cpp` failed to compile
until they were added).

Two legacy constructors were widened from `const amrex::MultiFab*` to
`amrex::MultiFab*` so `nonOwning()` can build the handles. Both are already
*called* with non-const pointers, so this widens a parameter and nothing else; both
keep their `const amrex::MultiFab*` members:

- `GmgStationarySolver` (file-local, `solve/persistent.cpp:282/356`) — 1 caller,
  `makeFaceCoeffSolver` at `:857`.
- `FaceCoeffCsrSolver` (`solve/persistent.hpp:133`, `solve/persistent.cpp:947`) —
  1 caller, `bindPersistent`'s `__init__` lambda at `ginkgoSolve.cpp:217`, which
  passes `&alpha` from a non-const `amrex::MultiFab&`. **The Python signature is
  unchanged.**

In `FaceCoeffKrylovSolver` the handle triples were hoisted to four locals
(`mesh`, `alphaLevel`, `upper`, `lower`) and reused by the four consumers that
previously each spelled `nonOwning(*ux), nonOwning(*uy), nonOwning(*uz)` inline —
`FaceCoeffOp::create`, `buildGmgHierarchy`, `makeFaceCoeffPrecond`,
`FaceCoeffOp32::create`. That is where the ux/lx transposition hazard actually
lived on this path.

### Smaller fixes

- `src/blockAmr/linearAlgebra/matrixFree/faceCoeffOp.cpp:197` — the host-path
  comment claimed "alpha is NOT staged" while `pinnedCopy(*diagField)` with
  `diagField = &(*alpha)` stages exactly that. Now says alpha *is* staged and why
  (PROTOTYPE C1 reads it as the diagonal source).
- `src/blockAmr/linearAlgebra/coefficientsConcepts.cpp:151` — the privacy guard
  was `!std::is_constructible_v<Coefficients, MatrixCoefficients, CellFieldLevel>`:
  a 2-argument probe at a 3-argument private constructor, so it passed vacuously.
  Now probes the real list
  `(MatrixCoefficients, CellFieldLevel, NeoN::Executor)`, plus a `StubPrivateCtor`
  with the same list declared *public* asserted **constructible** — that positive
  control is what keeps the negative from going vacuous again.
- `coefficients.hpp` — `Coefficients::symmetric()` **deleted** (see corrections).
  The storage-vs-interface note was rewritten: a symmetric format no longer
  "genuinely stores lower[d] aliased to upper[d]".

## 2. The two corrections

**(a) `symmetric()`.** There are exactly two `.symmetric()` call expressions in the
whole tree. `ginkgoSolve.cpp:1081` (`d["reports_symmetric"] = c.symmetric()`) is on
a `MatrixCoefficients` — `c` comes from `matrix.coefficients()`
(`matrix.hpp:67`) — so **`MatrixCoefficients::symmetric()` is live** and was kept;
`FaceCoeffFields::symmetry()` is now its second caller.
`report/coefficients-vs-faceCoeffMatrix-cleanup.md` was **right** that
`Coefficients::symmetric()` has zero callers, and it is deleted. Operators spell
`c.lower.has_value()` directly (`operators/laplacian.cpp:210`).

Worth recording: two types carrying one accessor name, one a memberwise copy of the
other, is the same duplication this task removed from `FaceCoeffFields`.
`Coefficients` is `MatrixCoefficients` + `rhs` + `exec` and copies all four members
in its constructor (`coefficients.hpp:112`). It could *hold* a `MatrixCoefficients`
the way `FaceCoeffFields` now does — see remaining work, item 1.

**(b) the privacy guard.** Fixed as above, strengthened rather than deleted.

## 3. Remaining work, ordered

1. **`Coefficients` holds a `MatrixCoefficients` instead of copying its four
   members.** `coefficients.hpp:85`. Blast radius: every operator reads
   `c.mesh` / `c.diag` / `c.upper` / `c.lower` unqualified —
   `operators/laplacian.cpp:200–220`, `bindings/ginkgoSolve.cpp:706–771` (three
   helpers), `linearSystem.hpp`. Either the members become `c.mc.diag` (~25 call
   sites, mechanical but noisy) or `Coefficients` grows forwarding accessors, which
   re-introduces a second spelling. **Blocked on a decision:** whether operators
   should say `c.mc.diag`. Also note `coefficientsConcepts.cpp:161–167` asserts
   `decltype(Coefficients::diag)` etc. and would have to follow.
2. **`GmgPrecondT<T>::create`** (`linearAlgebra/gmg/gmgPrecond.hpp:90`) still takes
   `ba, dm, geom, n, alpha, ux, lx, uy, ly, uz, lz` — 3 layout + 7 field
   parameters. 1 caller (`precond.cpp:50`), which now already holds the grouped
   handles, so the conversion is a one-call-site change plus the class body's own
   member initialisation. Left alone only because `gmg/` is being edited
   concurrently.
3. **`makeKokkosGmgApply`** (`linearAlgebra/gmgKokkos/apply.hpp:73`), same shape,
   1 caller (`precond.cpp:179`) which holds the handles. Same reason.
4. **`solveFaceCoeffs`** (`linearAlgebra/solve/oneshot.hpp:51`) and the
   `FaceCoeffSolver` / `makeFaceCoeffSolver` chain
   (`solve/persistent.hpp:113`, `persistent.cpp:797`) — seven loose
   `amrex::MultiFab*`/`&`. **Deliberately not converted:** these are the parameter
   lists nanobind binds (`ginkgoSolve.cpp:174` `bindPersistent`, and
   `ginkgo_solve_face_coeffs`), and the `nb::arg` names are a contract pytest
   asserts on. Converting means an adapter layer, not a rename.
5. **`bench/kokkosBench.hpp:86` `GmgArgs`** carries `alpha, ux, lx, uy, ly, uz, lz`
   as `const amrex::MultiFab*` — a *third* spelling of the seven-field matrix. Its
   members are name-tagged, so there is no positional hazard, but it is the same
   data. Converting needs a read-only grouping (see "Face fluxes", reason (b)) and
   the file is being edited concurrently.
6. **`bindings/ginkgoSolve.cpp` `writeCoefficients` / `readCoefficients` /
   `_la_matrix_probe`** (`:695`, `:749`, `:1050`) take six loose `MultiFab&`.
   These are the Python boundary — the `MultiFab`s arrive one per `nb::arg`. Their
   *bodies* already read through `MatrixCoefficients`. Nothing to do.

## 4. Blocked unifications, and on what decision

- **`Coefficients` vs `MatrixCoefficients`** — item 1. Decision: `c.mc.diag` at
  ~25 operator/binding call sites, or forwarding accessors (a second spelling).
- **A read-only field handle.** `CellFieldLevel`/`FaceFieldLevel` hold
  `shared_ptr<amrex::MultiFab>` and their accessor pair *deliberately* grants write
  access through a `const` handle (`core/fieldLevel.hpp:22-26`). Everything that
  reads coefficients without writing them — `assembleFaceCoeffCsr`,
  `buildGmgHierarchy`, `GmgArgs`, every flux site — therefore either has to be
  handed a write handle or cannot use the grouping at all. Two legacy constructors
  were widened to non-const to get past this; a third family (`GmgArgs`, and
  `stencilKernels.cpp`'s `const MultiFab& fx`) cannot be. Decision needed:
  add a const-qualified sibling (e.g. an `array<const amrex::MultiFab*, 3>`-shaped
  read handle), or accept that the grouping types are write handles and read-only
  consumers keep taking loose `const&`. **Do not** solve it with a const-stripping
  `nonOwning` overload — that silently defeats the accessor pair.
- **Collapsing the `Symmetry` enum entirely.** `IsMatrix::symmetry()` returns it and
  five binding sites compare against `Symmetry::symmetric`
  (`ginkgoSolve.cpp:1013, 1069, 1167, 1249, 1498`). Replacing the enum with a
  `bool`/`optional`-derived predicate is a Python-visible dict-key change
  (`d["symmetric"]`) — not a data-representation refactor. The *duplicate* half (a
  stored `sym` beside a stored `lower`) is what this task removed.

## 5. Face fluxes

Task: "create structs to group the face fluxes". **Decision: no struct, no
conversion.** The prior was to reuse `FaceFieldLevel`; the code overturns it on
three concrete grounds.

**(a) The flux triples do not TRAVEL.** `FaceFieldLevel` earns its keep in `la`
because *one* triple is built once and handed to five consumers — `FaceCoeffOp`,
`makeFaceCoeffPrecond`, `assembleFaceCoeffCsr`, `computeFaceCoeffDiag`,
`GmgPrecondT`. Every host-level flux signature in `stencilKernels.cpp` has exactly
**one** caller, and that caller is a nanobind lambda whose parameter list is frozen
(`nb::arg("fx")`, …). Verified by count for all twelve: `divUpwindAcc`,
`divLinearAcc`, `divVanLeerAcc`, `divQuickAcc`, `eulerStepVanLeerLap`,
`eulerStepLinearLap`, `eulerStepUpwindLap`, `eulerStepQuickLap`, `divUpwind`,
`divLinear`, `divVanLeer`, `divQuick` — one definition, one call each, plus
`buildStencilOffsets` (`:1040`) and `upwind_div_ncomp`'s inline body (`:1362`). So
grouping moves the three-adjacent-identical-parameters spelling from the callee's
parameter list to the caller's aggregate initialiser. The hazard relocates; it does
not shrink. That is churn against a frozen boundary.

**(b) Const-correctness, and it is decisive.** Every flux there is
`const amrex::MultiFab&`. `FaceFieldLevel` grants write access through a const
handle by design, so reusing it would silently widen access *and* need a
`const_cast` at each of ~20 binding lambdas. Avoiding that means a read-only twin —
a second structurally-similar struct, which is exactly the duplication this whole
task removed. Adding it for the flux family alone, while `GmgArgs` and
`assembleFaceCoeffCsr`'s old signature had the same need, would be solving the
narrow case. It belongs in the "read-only field handle" decision above, once, for
both families.

**(c) Half the sites are not `MultiFab`s at all.** `bench/cells.hpp:45`
(`divVanLeerCell`) and the per-cell helpers in `stencilKernels.cpp:37, 74, 111, 165`
take `Face const& fx` = `amrex::Array4<const amrex::Real>` inside
`AMREX_GPU_HOST_DEVICE` kernels. A `shared_ptr<MultiFab>`-backed handle cannot cross
that boundary. The bench path already has its own device-copyable bundle for it:
`Fields<Acc, N>` plus `fieldList()` in `bench/operators.cpp:62`, with the order
fixed and documented (`0 = out, 1 = in, 2..4 = fx, fy, fz`).

Two further findings that support leaving this alone:

- **The grouping already exists where the triple does travel.**
  `bench/kokkosBench.hpp:36` `OpArgs` holds `fx/fy/fz` as members, assigned
  *by name* in `benchBindings.cpp:41-45` — no positional hazard. Its fluxes are
  **nullable** (`MultiFab* fx = nullptr`, "may be null for operators that do not use
  them"), the opposite invariant to `FaceFieldLevel`'s non-nullable-by-construction,
  so absence would have to become `std::optional<FaceFieldLevel>` or a
  pointer-shaped variant — a third shape.
  The one loose spelling left is `makeArgs`'s parameter list
  (`benchBindings.cpp:29`), whose two callers are frozen Python lambdas (`:91`,
  `:120`).
- `linop.cpp:416` (`get_fluxes`) already groups into AMReX's own
  `amrex::Array<MultiFab*, AMREX_SPACEDIM>` inside a four-line Python-visible
  lambda. `multifab.cpp:1158` is an `nb::arg` name only.

**If the maintainer still wants a distinct type**, the justification to build it on
is reason (c) from the brief — fluxes and matrix face coefficients were nearly
conflated in an earlier analysis, and a distinct `FaceFluxLevel` would make the
mixup a compile error. The shape to give it is the read-only handle from §4
(`std::array<const amrex::MultiFab*, 3>` + a single `const` accessor), *not* a copy
of `FaceFieldLevel`, and it should land together with the `GmgArgs` conversion so
one read-only handle serves both families. The place it would pay for itself is
`stencilKernels.cpp`'s internals only if the binding lambdas are also reshaped to
take a Python-side flux object — a Python API change, out of scope here.

Deliberately untouched, listed as asked: `bench/cells.hpp`, `bench/kokkosBench.hpp`,
`bench/operators.cpp`, `bindings/stencilKernels.cpp`, `bindings/benchBindings.cpp`,
`bindings/linop.cpp`, `bindings/multifab.cpp`. No code line in any of them changed.

## 6. Verification

Not a build: each TU compiled individually from `_skbuild/compile_commands.json`
with the object written to `/tmp`, so nothing in the build tree was touched. All
eight TUs that include any header I changed, plus every TU containing an updated
call site, returned `rc=0`:

`coefficientsConcepts.cpp`, `linearAlgebra/precond.cpp`,
`linearAlgebra/solve/persistent.cpp`, `linearAlgebra/solve/oneshot.cpp`,
`linearAlgebra/sparse/csr.cpp`, `linearAlgebra/matrixFree/faceCoeffOp.cpp`,
`operators/laplacian.cpp`, `bindings/ginkgoSolve.cpp`.

`coefficientsConcepts.cpp` is the important one: it instantiates
`static_assert(IsMatrix<MFFaceCoeffs>)` and `IsMatrix<CsrMatrix>`, so the whole
`FaceCoeffFields` rewrite is checked there.

**Compiled but not run.** No test executed. Behaviour-neutrality is argued, not
measured — the claims to re-check when the suite runs are: (i) a symmetric matrix
still hands `FaceCoeffOp`/`assembleFaceCoeffCsr`/`makeFaceCoeffPrecond` the *same*
`MultiFab` for `upper[d]` and `lower[d]`; (ii) `_la_matrix_probe`'s `symmetric`,
`reports_symmetric` and `lower_empty` keys are unchanged; (iii) GMG iteration counts
are identical, since `buildGmgHierarchy` now passes `mesh.ba/dm/geom` where it
passed `alpha->boxArray()/DistributionMap()/geom`.

**Not verified:** `precond.cpp`'s two multi-line call expressions
(`:121`, `:179`) are in a formatting state that `clang-format` v17 (the pinned
pre-commit version) will reflow. I reflowed them once; a concurrent agent reverted
the reflow to the form it had read, and I left it, because the code is correct
either way and re-applying risks a lost edit. Expect a formatting-only diff from
`pre-commit`.
