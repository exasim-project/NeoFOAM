# blockAMR GMG notes

Rationale and measurements that used to live as long comment blocks in the headers
under `src/NeoN/include/NeoN/blockAmr/linearAlgebra/gmg/` and `.../gmgKokkos/`.
Each header now carries a one-line pointer to the section here.

Precision/reduced-storage measurements are NOT here — they live in
[`report/blockamr-precision-measurements.md`](../report/blockamr-precision-measurements.md).

## fusion

*(from `gmg/gmgKernels.hpp`, `faceCoeffResidScatterNorm`)*

`faceCoeffResidScatterNorm` is a fused residual + convert-scatter (heavy stencil
kernel, arithmetic in double, result stored cast to `T` straight into the L0 rhs)
followed by a SECOND, light kernel that reduces the just-written `out`.

It is deliberately not one kernel: folding the reduction into the heavy stencil
kernel measured ~1.0 ms/iter at 256^3 — it spills the register-bound kernel —
against the 0.34 + 0.54 ms it saves. The separate reduction over the freshly
written `out` costs ~0.20 ms/iter and reuses the cached data.

One reduction yields BOTH norms (`sum r^2` and `max|r|`), so the solver can stop in
`norm="l2"` or `"linf"` without a second pass; the extra `ReduceOpMax` is
register-only work.

Precision of the reduced norm: see
[`report/blockamr-precision-measurements.md`](../report/blockamr-precision-measurements.md).

The residual write honours `Gpu::LaunchSafeGuard` like every other twin, but the
norm is an EXPLICIT `if (onDevice)` branch: `amrex::ParReduce` picks its
`ReduceOps::eval()` at COMPILE time via `#ifdef AMREX_USE_GPU` with no runtime
`inLaunchRegion()` check, so `LaunchSafeGuard` cannot send it to the host (unlike
`HostDeviceParallelFor` or `amrex::ReduceSum`, which is `gmgNorm2`'s pattern).

## rank-reduction

*(from `gmg/gmgKernels.hpp`, `reduceResidNorms` and `gmgNorm2`)*

`amrex::ParReduce`/`ReduceSum` are rank-LOCAL. The residual norms must therefore be
combined here and on the SAME communicator as the `||rhs||` `gmgSolve` compares
against (`MultiFab::norm2`/`norminf`, which do reduce); otherwise the residual is
understated and the solve stops early and silently — measured at 2 ranks,
`||r||_2` came out 0.711x and `||r||_inf` 0.871x of the true value. Two collectives
because l2 needs a Sum and linf a Max: one latency each per V-cycle, not per kernel.

`gmgNorm2` (setup power iteration only) needs its cross-rank sum for a different
reason: the power iteration renormalises by this norm, so a per-rank norm would give
each rank a DIFFERENT `lambda_max` and different Chebyshev coefficients — the
smoother would stop being one global linear operator. Every level shares level 0's
`DistributionMapping`, so all ranks reach the collective equally often.

`vcycle.hpp`'s `sameField` has the same shape: its `ParReduce` sees this rank's boxes
only, and the answer decides how many face fabs a level keeps, so ranks that
disagreed would build different hierarchies.

## smoother

*(from `gmg/gmgPrecond.hpp`)*

RB-GS is the default because it was measured, not assumed: 9/9 CG iterations at
N=32/64, against 16/16 for omega=6/7 damped Jacobi and 20/22 for omega=2/3.

Chebyshev smooths eigenvalues in `[lambdaMax / kChebEigRatio, lambdaMax]` and leaves
the lower modes to the coarse grid. 4-8 is the usual band; a sweep over
{2,3,4,6,8,15,30} put the minimum at 6 (degree 2: 11 CG iterations at N=32/64,
against 9 for rbgs).

`omega = 1.1` is the measured optimum for a symmetric operator, but it breaks the
colour sweep's self-adjointness, so it is refused once the caller declares the
operator asymmetric — as is Chebyshev, whose polynomial is built on the REAL interval
`[lambda_max/6, lambda_max]` and has no contraction guarantee on a complex spectrum.
Symmetry is DECLARED by the caller, never sniffed: a coefficient set that happens to
be symmetric on this call may not be on the next, and switching algorithm on that
would change the answer without changing the configuration.

## bottom

*(from `gmg/gmgBottom.hpp`)*

Why a Krylov bottom exists at all: a smoother cannot touch the coarse grid's
near-null modes — a consistent polynomial smoother has `p(0) = 1`, so the constant
mode survives every sweep. MLMG solves its bottom with a Krylov method for the same
reason.

Why Ginkgo and not a hand-rolled CG: CG is valid only for a SYMMETRIC operator, so as
soon as the operator can be asymmetric (convection) the bottom also needs BiCGStab or
GMRES — three hand-rolled implementations instead of one dispatch. `GmgBottomOp` reads
its own upper and lower coefficient array per direction, so it represents an
asymmetric operator exactly; which SOLVER is legal on it is the caller's explicit
`symmetric` flag, never inferred. A CG bottom on an asymmetric operator does not fail
loudly — it converges to the wrong correction or stalls, and the caller sees only a
worse outer iteration count — hence refused, not warned.

STATIONARITY. A residual-tested Krylov bottom takes a different iteration count per
right-hand side, making the V-cycle a DIFFERENT linear operator on each apply —
outside the theory of an outer CG, which assumes a fixed preconditioner. Two ways to
stay inside it: a tight `rtol` so the variation is below what the outer solver can see
(cheap, the bottom is a handful of cells — recommended), or a FLEXIBLE outer method
(`solver='gcr'` or `'fcg'`). The DEFAULT bottom is `'smoother'` (fixed sweeps),
stationary by construction. The parent zeroes the coarse sol before recursing, so the
bottom always starts from x = 0, which also makes its iteration count reproducible for
a given rhs.

## halo-plans

*(from `gmgKokkos/halo.hpp` and `gmgKokkos/vcycle.hpp`)*

`gmgKokkos/kernels.hpp` leaves the data movements (FillBoundary, the agglomerated
level's ParallelCopy, setVal) to AMReX because they are not cell loops — and that
costs a synchronisation point per operation, the two runtimes' streams being
unordered: a colour sweep becomes `fence -> FillBoundary -> streamSynchronizeAll ->
kernel`, so the host waits on the device twice per colour. Dropping the Kokkos fence
alone buys nothing; the FillBoundary needs the same ordering. The halo exchange is
therefore the pivot: with it on Kokkos the whole timed cycle is one correctly ordered
Kokkos stream with no host fence, and the host can run a coarse level's launches ahead
of the arithmetic above it — which is what the launch-bound coarse levels need.

A plan is AMReX's decomposition resolved ONCE, at setup, untimed, built from the same
primitives FillBoundary uses internally (`boxDiff` for the ghost shell,
`BoxArray::intersections` for who covers it, `Periodicity::shiftIntVect` for the
images). The timed cycle sees a flat device table of rectangular copies and one launch.

Ghosts are enumerated as the shell `boxDiff(grow(valid, ng), valid)`, which excludes
the valid region, so no task copies a box onto itself; the covering sources come from
the BoxArray's own hash, so this is the same partition of the shell FillBoundary uses
— valid regions and their images tile space, every ghost is covered exactly once, and
task order cannot matter. Ghosts outside a non-periodic domain get no task, as in
FillBoundary, which leaves those to the boundary-condition code.

Fixed-size work blocks (`kCopyBlock`) rather than one team per region, because region
sizes differ by orders of magnitude: a single-box 256^3 level has 26 regions, six of
them 65k cells, and one team per region would hand each of those to 128 threads. Not a
corner case — every level of a hierarchy coarsened in place from one box looks like
that. A block short of a full `kCopyBlock` leaves lanes idle, which is cheaper than
any mapping that would even it out.

ONE RANK ONLY, and that is a limit of the PLANS, not of the kernels: a `CopyTask` names
two LOCAL box indices, so a ghost covered by a box on another rank has no address to
copy from. Nothing in `halo.hpp` consults the rank count — on >1 rank these builders
are simply not called and the same three movements are routed through AMReX
(`Vcycle::amrexFree_`), i.e. exactly what the `kokkos_fused` backend already does.

`makeCopyPlan` is cell-centred only: face BoxArrays share their internal faces, so a
face cell can have several sources and the result would depend on task order (harmless
for the coefficients, which are copied once at setup and stay with AMReX).

`gmgZeroKokkos` clears VALID cells only where `setVal` also clears the ghosts —
equivalent only because the sole reader of a coarse solution's ghosts is a colour sweep
and `smooth()` runs a full ghost fill first, while prolongation reads valid cells only.
Anything that read those ghosts unfilled would have to clear them here.

## kokkos-twins

*(from `gmgKokkos/kernels.hpp`)*

The Kokkos kernels are twins of the three `gmgKernels.hpp` V-cycle kernels the timed
cycle runs (`gmgGsColor`, `gmgResidRestrict`, `gmgProlongAdd`) — same order, same
signatures, same cell arithmetic, so the correspondence stays reviewable side by side.

Two launch forms per kernel, same signature, one shared body: `*Kokkos` is one
`MDRangePolicy` per box (the shape production is written in), `*KokkosFused` one
`TeamPolicy` for all boxes of the level, so per-box launch cost cannot appear. The cell
arithmetic is factored into `*Cell` structs precisely so the two cannot drift apart.
`arrays()`/`const_arrays()` is AMReX's cached device Array4 table, so the fused
launch's per-launch cost is a pointer copy.

The fused loop is driven by the fab whose valid boxes define the iteration space — rhs
for the smoother, the COARSE rhs for the restriction, the FINE solution for the
prolongation — and every other field is addressed at the same local box index. Exact
whenever the fabs share a `DistributionMapping` and box order, which `vcycle.hpp` keeps
true by routing an agglomerated level through a transfer fab on the fine layout.

Kokkos writes the MultiFab memory on its own default execution space, which costs ONE
fence per kernel: the V-cycle interleaves these with AMReX and the two runtimes'
streams are unordered. Not a handicap — production's default-`MFItInfo` MFIter
stream-synchronizes in its destructor too, so both backends sync once per kernel. The
twins pass `DisableDeviceSync`, dropping the AMReX sync that has no AMReX kernel left
to wait for rather than paying both. The `fence` argument of the fused launchers orders
against whatever runs next: true whenever that is AMReX (the default, and what every
backend but `kokkos_opt` needs), false when it is another Kokkos kernel on the same
execution space.

Accessors are `amrex::Array4` for both backends, not an unmanaged Kokkos View: the
operator bench showed the accessor choice sits within the noise floor, so the LAUNCHER
stays the only difference between the two columns.

## agglomeration

*(from `gmgKokkos/vcycle.hpp`)*

Without agglomeration the fine BoxArray is coarsened in place and the
`DistributionMapping` reused, so only the box SIZE shrinks — production behaviour, and
why the coarsest level is launch-bound.

Agglomeration takes a fresh `aggGridSize`-capped decomposition of the coarse domain,
only when it has strictly fewer boxes than coarsening in place. Same mechanism as
MLMG's `LPInfo::do_agglomeration` but a different trigger: MLMG reduces the number of
MPI ranks with work (default `agg_grid_size` 8 in 3D, which on one GPU would leave the
box count untouched), while the cost here is per-box launches — so the trigger is the
box count itself. It cannot change the result at equal depth (red-black smoothing is
decomposition-independent), it keeps the coarse levels from being one launch per tiny
box, and the extra depth it unlocks helps the iteration count.

An agglomerated level's BoxArray no longer matches the fine level box for box, so the
inter-level kernels — which address fine and coarse at the SAME local box index —
cannot reach it. `xferRhs`/`xferSol` hold the restriction output and the prolongation
input on `coarsen(fine BoxArray, 2)` with the FINE `DistributionMapping`, and a copy
bridges to the level's own fields.

Level-0 agglomeration (`aggLevel0Size`) is the same idea for the level the caller
addresses, so bigger boxes cost an interface: a caller-layout fab at each end of an
apply plus a plan-driven copy, once per APPLY, so `precond_cycles > 1` amortises it.
What it buys is halo traffic: 32^3 boxes carry 19% ghost overhead against 9.4% for
64^3, and level 0 is 7/8 of the cells in the hierarchy. Only available where the plans
are — on >1 rank the caller's layout is kept, because the trade was measured against a
local copy, not against an MPI ParallelCopy.

## share-coeffs

*(from `gmgKokkos/vcycle.hpp` and `gmgKokkos/apply.hpp`)*

`ux == lx` pointwise IS symmetry of the operator (cell i's east coefficient and cell
i+1's west one are the entries `A[i][i+1]` and `A[i+1][i]`, both stored at face i+1),
and so the exact condition for one array to stand in for both. That removes three of
the nine arrays a colour sweep streams. The test is BITWISE on purpose: a tolerance
would silently symmetrise a near-symmetric operator. The common case (the same fab
passed twice) is settled by the pointer test; the reduction runs at setup only.

A shared level rediscretises three faces instead of six: averaging the same fine values
twice gives the same coarse numbers, so symmetry survives the hierarchy and the pair is
never re-formed. An operator that quietly lost its lower coefficients would solve a
different system at full speed, hence the verification.

## kokkos-handle

*(from `gmgKokkos/apply.hpp` and `gmgKokkos/precond.hpp`)*

`KokkosGmgApply` mentions neither Kokkos nor Ginkgo because the two stacks compile in
different object libraries: the Kokkos kernels in `blockamr_kokkos`, the Ginkgo stack
in `blockamr_solvers`, both with `CUDA_SEPARABLE_COMPILATION OFF` (`blockamr_kokkos`
is separate by history, not because of an RDC fence — see CMakeLists.txt). So the cycle
is an opaque handle over flat device vectors and `precond.hpp` wraps it in a
`gko::LinOp` on the other side, with nothing shared but that header and two double
pointers.

Why a preconditioner interface and not just a cycle benchmark: a V-cycle measured in
isolation is not a preconditioner. What a caller sees is the SOLVE, where the cycle is
one term next to the matrix-vector product, the Krylov algebra and the iteration count
— and this is the only way to compare it with MLMG and the native GMG on equal terms
(`bench_solvers.py` runs all of them over the same operator).

`GmgKokkosPrecondT` sits beside `GmgPrecondT` rather than inside it, because
`GmgPrecondT` is the shipped preconditioner and the baseline every measurement is read
against: `precond="gmg"` and `precond="gmg_kokkos"` are independent objects, so
`bench_solvers.py` can run both in one process and compare them. What the Kokkos one
does not carry, because the ported V-cycle does not: the Chebyshev smoother and the
host (ReferenceExecutor) path — both rejected explicitly rather than ignored.
