# blockAMR linear-algebra notes

Rationale that used to sit as long comment blocks under `src/NeoN/`. The code keeps a
one-line pointer to the heading here. Precision and reduced-precision measurements live
in [`report/blockamr-precision-measurements.md`](blockamr-precision-measurements.md)
instead.

## Coefficient handles

*(from `include/NeoN/blockAmr/linearAlgebra/coefficients.hpp`)*

The handles are deliberately narrow: a `CellFieldLevel` is one cell-centred
`amrex::MultiFab`, a `FaceFieldLevel` the three direction fields
(`core/fieldLevel.hpp`). They stand for **both** formats, assembled and matrix-free, and
neither needs a type erasure or a variant inside it.

The reason is that `assembleFaceCoeffCsr` already takes seven `amrex::MultiFab`s and
reads them host-side to build its arrays (`sparse/csr.cpp`) — MultiFabs are the assembled
path's *input*, not a staging buffer added for this interface's convenience. So
`CsrMatrix` holds exactly what `MFFaceCoeffs` holds and hands out these same handles
(`faceCoeffMatrix.hpp`); what it needs on top is only an assembly-freshness flag.

Do not generalise these types speculatively. Revisit only for a format whose storage is
genuinely not MultiFab-shaped — an ELL/banded matrix owning its own arrays, or a
device-assembled CSR that never round-trips through a MultiFab — and then the erasure
goes **in the handles**, not in `Matrix`.

Absence has exactly one spelling, and it is `std::optional` on the one member that
varies. There is no `empty()` sentinel: `diag`, `upper` and `rhs` are non-nullable by
construction, so a format cannot express handing back a missing one and the operators do
not check for it. `coefficientsConcepts.cpp` asserts that.

`lower == nullopt` is the **interface** saying "there is no low side to write", because an
operator writing both sides would double every coefficient. Storage is a separate
reading: a symmetric format still hands a low side to the machinery underneath — `upper`
itself, the alias `FaceCoeffOp`'s convention wants — through
`detail::FaceCoeffFields::storedLower()`, whose return type (`FaceFieldLevel`, not
`std::optional<FaceFieldLevel>`) is what keeps the two from being confused at a call
site.

`mesh` is first and lives on the coefficients rather than beside them at a call site
because they are not self-describing without it: every operator that writes a face
coefficient needs `dx` to scale it and the periodicity to fill the halo the face average
reads across. Passing it separately made a mismatch representable — `ops::Laplacian` used
to carry its own `amrex::Geometry`, handed in at construction, and nothing checked it
against the matrix's.

## IsMatrix

*(from `include/NeoN/blockAmr/linearAlgebra/coefficients.hpp`)*

There is no base class: satisfy the concept and you are a matrix format.

`makePrecond` is the **format's** job, which is the whole point of it being on the matrix
rather than on the solver. A geometric-multigrid hierarchy is built by rediscretising the
*coefficients* on a sequence of coarser levels; a solver holds a `gko::LinOp`, by which
point the erasure has thrown the coefficients away, so the only object that still can
build one is the format that owns them.

`config` is the whole `SolverConfig` the solve was asked for, reused rather than
re-spelled as a narrower precond-only struct: it already carries `precondKind`, the
`GmgConfig` V-cycle knobs, `precondCycles` and `precondMlmg`.

Returning **null** means the format *declines*: it has no way to build that
preconditioner and says so by handing back nothing, leaving the caller — which is the
only party that knows what it was going to do with it — to raise. A format still throws
for a combination it must refuse on its own terms, and `precond="none"` returns null
without either party treating it as a refusal. `name()` exists for that message and for
nothing else: an erased `Matrix` cannot otherwise say what it is holding, and "declines
precond 'gmg'" is not actionable without it.

## Matrix formats

*(from `include/NeoN/blockAmr/linearAlgebra/faceCoeffMatrix.hpp`)*

The matrix formats' half of the boundary-condition contract; the operator's half is
`operators/laplacian.hpp` and `plans/blockamr-linear-algebra-notes.md#laplacian-bcs`.

Both formats carry a `BcArray`, and so does `ops::Laplacian`. It looks like a double fold
and is not.

**These formats are the only place the homogeneous domain BC is applied.** `op()` hands
`bc` to the machinery underneath: `FaceCoeffOp` reflects the domain ghosts per apply,
`assembleFaceCoeffCsr` folds the same reflection onto the diagonal per assembly and drops
the off-diagonal. `ops::Laplacian` carries `bc` for two other reasons only — it must know
which domain faces have no second cell to average `gamma` over, and which sides read an
inhomogeneous datum — and it deliberately leaves the boundary **face coefficient** live.
Every fold here is multiplicative in that coefficient, so a live one is exactly what they
need.

The dependency that makes it safe, spelled out because it is load-bearing and invisible:
an operator that *also* folded — zeroing the boundary coefficient and putting
`(sign-1)*aF` on the diagonal source — would leave the folds in the formats inert and the
**fine** matrix identical, which is why the arrangement can be got wrong and still pass a
solve. It is wrong on the **coarse** levels: the GMG hierarchy built by `makePrecond`
coarsens `alpha` with `gmgRestrict`, an eight-child volume average correct only for a
dx-*independent* density, while `(sign-1)*aF` is `2*gamma/dx^2`. On the face,
`gmgCoarsenFace`'s 1/4 is the right law. Measured: fully-Dirichlet CG+GMG took 12/13/14
iterations at 64/128/256^3 with the operator folding and 8/8/8 without — 1.7x slower and
mesh-*dependent*.

The guard is the **bitwise** coefficient assertion in `test_la_boundary_conditions.py`
(`test_laplacian_writes_the_boundary_face_coefficient`); do not replace it with a
solve-level comparison. `test_the_two_formats_agree_through_the_laplacian` sees nothing —
both formats fold whatever they are handed the same way, so they agree with each other
under either convention.

Keeping `bc` on the format is also what preserves the variable row length: with an
all-zero `BcArray`, `csr.cpp`'s else-branch emits an explicit `0.0` at the
modular-wraparound column, so a non-periodic row would carry seven entries including a
periodic coupling that does not exist.

The legacy `blockamr::la::FaceCoeffSolver` path folds BCs at apply time in exactly the
same way, and always has: it shares its coefficient fields with the GMG hierarchy, which
applies its own ghost reflection per level, so folding into them would apply every BC
twice per level. The `la::` path now agrees with it on the stored coefficients too.

## AmrexLinOpBase

*(from `include/NeoN/blockAmr/linearAlgebra/matrixFree/linOpBase.hpp`)*

A derived class `D` derives as `public AmrexLinOpBase<D>`, forwards to
`AmrexLinOpBase<D>(exec[, size])` in its constructors, and implements only
`apply_impl(b, x)`, preceded by `using AmrexLinOpBase<D>::apply_impl;` so that
declaration does not hide the advanced overload (nvcc warning 611 /
`-Woverloaded-virtual`; cosmetic, the code is correct either way). The exec-only
constructor is required by `create_default` / `clear`, which do `new D(exec)`.

`V` is the value type of the `Dense` vectors only. `gko::EnableLinOp` carries no value
type, so a derived operator is a plain `gko::LinOp` and `Cg<float>` accepts it.

Before this base existed, every derived operator carried a byte-identical copy of the
advanced `apply_impl`.

## Composite AMR operator

*(from `include/NeoN/blockAmr/linearAlgebra/matrixFree/mlmgOps.hpp`)*

`CompositeAmrexOp` is the multi-level `AmrexOp`: the Ginkgo vector concatenates all
levels' valid cells (coarsest first, in the gather/scatter per-box order) and the mat-vec
is the **composite** multi-level `MLMG::apply` — fine coarse/fine ghosts interpolated from
the coarse `in`, the coarse residual refluxed at the interface (cancelling any dependence
on covered coarse cells), the covered coarse output overwritten by `average_down` of the
fine one.

Hence, on the concatenated vector:

- Covered coarse columns are **zero** (index-1 singular; the nullspace is
  covered-cell perturbations, disjoint from the range), so a consistent rhs — covered
  coarse rhs = `average_down` of the fine rhs, enforced by the caller — is solvable and
  the covered solution entries are fixed by a final `average_down`.
- The operator is **not symmetric**: the c/f interpolation is not the adjoint of the
  reflux, so `bicgstab`/`gmres` are the safe solvers. CG may still work in practice —
  that is for the caller and its tests to measure.

The affine offset `c0 = L_inhom(0)` is recorded per level, as in `AmrexOp`.

## Laplacian BCs

*(from `include/NeoN/blockAmr/operators/laplacian.hpp` and
`src/blockAmr/operators/laplacian.cpp`)*

### The split

A non-periodic domain face carries its **real** face coefficient, with `gamma` taken
from the boundary cell **itself** — the ghost beyond it is never filled, so reading it
would read recycled arena memory.

The **diagonal** half of the homogeneous boundary condition belongs to the
**consumer**, and is applied **per level**. `core/bc.hpp` fills
`ghost = sign*interior + scale*g`:

| kind      | sign | scale    | `g`                                |
| --------- | ---- | -------- | ---------------------------------- |
| Dirichlet | -1   | 2        | `u` on the FACE                    |
| Neumann   | +1   | `dx[d]`  | `du/dn`, the OUTWARD normal deriv. |

All three consumers apply that half themselves:

- `FaceCoeffOp` reflects the ghost on every apply (`matrixFree/faceCoeffOp.cpp`),
- `assembleFaceCoeffCsr` folds `diag += sign*aFace` (`sparse/csr.cpp`),
- the GMG hierarchy reflects on every level it builds (`gmg/gmgPrecond.hpp`).

All three are **multiplicative** in the face coefficient, so it must stay live —
which is what makes `bc` on the matrix formats load-bearing arithmetic rather than
metadata. `faceCoeffMatrix.hpp` holds the other half.

### Why that direction, when folding gives the same FINE matrix

The folded `(sign-1)*aF` is dx-**dependent** (`2*gamma/dx^2` for Dirichlet) yet sat in
`alpha`, where `gmgRestrict` coarsens it by a plain eight-child volume average that is
correct only for a dx-**independent** density. On the face it coarsens by the correct
1/4 law instead (`gmgCoarsenFace`). Folded, every coarse level inherited a boundary
diagonal that was too strong, and the V-cycle degraded as the mesh refined.

Hence: **do not re-zero the face coefficient** to make room for an operator-side fold.
Such a fold can only ever be right on the finest level.

### Measured

Same rhs, tolerance and cycle shape:

| configuration    | 64^3  | 128^3 | 256^3 | note                        |
| ---------------- | ----- | ----- | ----- | --------------------------- |
| periodic, either | 8     | 8     | 8     | solutions bitwise equal     |
| Dirichlet, face  | 8     | 8     | 8     | the convention in the code  |
| Dirichlet, folded| 12    | 13    | 14    | 1.69x / 1.72x / 1.74x slower|

Neumann has `(sign-1) == 0`, so nothing was ever folded and both conventions agree.

### Tripwire

`test_la_boundary_conditions.py::test_laplacian_writes_the_boundary_face_coefficient`
is load-bearing: it is the only test that reaches **Neumann**, and solve-level tests
stay green either way. `test_the_two_formats_agree_through_the_laplacian` catches
nothing here — both formats fold whatever they are handed, identically.

## Norms

*(from `include/NeoN/blockAmr/linearAlgebra/krylov/stopNormInf.hpp`)*

The convergence norm is a **choice**. Ginkgo's criteria measure the residual in the
2-norm; AMReX's MLMG measures it in the **infinity** norm, relative to
`max(||b||_inf, ||r0||_inf)` (`AMReX_MLMG.H`: `MLResNormInf` / `MLRhsNormInf`,
`MLMGNormType::greater`, `res_target = max(atol, max(rtol,1e-16) * max_norm)`).

Two solvers stopping on different norms answer different questions, so their iteration
counts are not directly comparable — which matters because the interesting comparisons
are close: **mlmg at 9 iterations against mf-gmgk at 10**.

Which criterion is stricter follows from each vector's max/rms ratio `C`:

```
||r||_inf / ||b||_inf = (C_r / C_b) * ||r||_2 / ||b||_2
```

Neither dominates a priori. The point is only that a comparison should be able to hold
the norm fixed, which is why `ResidualNormInf` exists.

Ginkgo has no inf-norm reduction (`Dense` offers `compute_norm2`/`compute_norm1`
only), so the criterion reduces `max|r_i|` itself with `amrex::Reduce` over Ginkgo's
own device pointer — one cross-runtime synchronisation per iteration, the same one
MLMG's `ResNormInf` pays.

## The nvcc multi-TU trap

*(from `src/blockAmr/core/deviceKernels.cpp`, `src/blockAmr/linearAlgebra/gmg/gmgKernels.cpp`
and `src/blockAmr/linearAlgebra/gmgKokkos/kernels.cpp`)*

A kernel-launching template that opens an extended `__host__ __device__` lambda
(`AMREX_GPU_DEVICE`, `BLOCKAMR_LAMBDA`, `KOKKOS_LAMBDA`) and is reached from more than one
`.cpp` must be **declaration-only in the header**, with its definition and an explicit
instantiation per needed argument list in exactly one `.cpp`.

Left as an ordinary header template it is instantiated once per including CUDA translation
unit, and several of those units land in the *same* final `_blockamr.so` (`blockamr_solvers`:
`solve/persistent.cpp`, `matrixFree/faceCoeffOp.cpp`, `matrixFree/mlmgOps.cpp`;
`blockamr_kokkos`: `gmgKokkos/apply.cpp`, `bench/gmgVcycleBench.cpp` — both are OBJECT
libraries linked into one shared object, not separate `.so`s). Weak/COMDAT folding then keeps
one TU's host-side stub while the TUs' device-side kernel registrations are not guaranteed
consistent with it.

The observed failure mode is **a null device function pointer called at runtime, not a compile
or link diagnostic**: 43 of 102 gate tests SIGSEGV inside `launchKokkosTeamNamed`, for exactly
the `kokkos_fused`/`kokkos_opt` cases, i.e. exactly the callers of the five launchers
`gmgKokkos/kernels.cpp` now defines. The same trap is why a *missed* explicit instantiation is
not caught by the linker either.

`core/deviceKernels.cpp` holds the `bc.hpp`/`transfer.hpp` kernels of this class and lives in
`blockamr_kokkos` rather than beside `core/bc.cpp` in `blockamr_solvers`, because
`gmgKokkos/apply.cpp` and `bench/gmgVcycleBench.cpp` need those symbols in a build **without
Ginkgo**, where `blockamr_solvers` does not exist at all (`blockAmr/CMakeLists.txt`).

## GMG V-cycle bench backends

*(from `src/blockAmr/bench/gmgVcycleBench.cpp`)*

The bench runs the native V-cycle of `gmgPrecond.hpp` through four launchers over the same
hierarchy, sweep counts, control flow and order of operations — only the launcher differs, so a
row is read against the row above it:

| backend        | change                                                                |
| -------------- | --------------------------------------------------------------------- |
| `amrex`        | the production per-box path — the orientation point                   |
| `kokkos`       | its per-box Kokkos twin                                               |
| `kokkos_fused` | the same kernels under one `TeamPolicy` launch per level              |
| `kokkos_opt`   | halo exchange, zero fill and agglomeration transfers on Kokkos too    |

`kokkos_opt` (`gmgKokkos/halo.hpp`) leaves no AMReX operation inside the timed cycle, so there
is no reason to fence between kernels at all and the whole cycle becomes one stream the host
can run ahead of.

Only the Kokkos side is optimised, deliberately: the `amrex` column has to stay the shipped
V-cycle for the comparison to mean anything, and `kokkos`/`kokkos_fused` stay put as the
intermediate baselines. That is also why reduced precision, `share_coeffs` and
`agg_level0_size` are **refused** on every backend but `kokkos_opt` instead of being ignored —
an ignored knob would report a baseline timing under another label. Orthogonally,
`GmgArgs::agglomerate` switches the hierarchy from production's in-place `BoxArray` coarsening
to a re-decomposed coarse grid; red-black smoothing is decomposition-independent, so at equal
depth that changes cost without changing a single arithmetic result.

Port scope, relative to the DEVICE path of `GmgPrecondT<double>`:

- **kept**: hierarchy construction by in-place `BoxArray` coarsening (box COUNT preserved down
  the levels, as in production, unless agglomeration is asked for), RB-SOR smoothing with the
  reversed post-sweep, fused residual+restriction, piecewise-constant prolongation, the ghost
  fill per colour, and the recursive V-cycle with warm-started `sol`.
- **dropped**: Ginkgo (no `LinOp`, no `Dense` pack/unpack), the `ReferenceExecutor` host path,
  the Chebyshev smoother and its lambda-max power iteration, and physical boundary conditions —
  the bench is triply periodic, so `bc.hpp` stays out of this translation unit.

The `amrex` column calls the PRODUCTION kernels (`gmg/gmgKernels.hpp`) rather than a copy, and
is recompiled here in the non-RDC object library, which is what makes the compile flags
identical for both columns (production's `_blockamr` is non-RDC too). `blockamr_kokkos` is a
separate library by history, not because of an RDC split.
