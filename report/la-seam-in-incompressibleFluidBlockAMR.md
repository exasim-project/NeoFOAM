# The `blockamr::la` seam and `incompressibleFluidBlockAMR`

Investigation, read-only. Paths are relative to the repo root
(`/home/henning/libsAndApps/NeoFOAM/.claude/worktrees/feat+blockamr_ginkgo`).
`src/NeoN/` is a submodule with an in-flight working tree; where a claim depends on an
uncommitted change this document says so.

## Summary

**No.** `incompressibleFluidBlockAMR` does not touch the type-erased linear-algebra
seam anywhere. Its pressure Poisson solve goes through
`blockamr.dsl.solve._solve_implicit`, which builds an AMReX `MLNodeLaplacian` + `MLMG`
(`src/NeoN/src/blockAmr/python/blockamr/dsl/solve.py:290-336`), and its MAC projection
goes through a second, independent `MLABecLaplacian` + `MLMG`
(`src/NeoN/src/blockAmr/python/blockamr/operators/mac_project.py:165-190`). Neither
`blockamr.linear_algebra` nor `blockamr.FaceCoeffSolver` appears anywhere under
`src/neofoam/` — a grep for `linear_algebra|LinearSystem|MFFaceCoeffs|FaceCoeffSolver|GmgConfig`
over `src/neofoam/` and `test/` returns only unrelated `SolverConfig` classes belonging
to the framework's own IO decorator (`src/neofoam/io/decorator.py:28`) and to
`test/framework/integration/dummy_solver/`. The single biggest blocker is not
performance and not statelessness: **the solver's pressure projection is NODAL and the
seam is strictly cell-centred.** The unknown is `p` on the *nodes* of a nodal
`BoxArray` (`solve.py:253-258`), the rhs is AMReX's `comp_divergence`
(`solve.py:328-330`), and the velocity correction consumes MLMG's `get_fluxes` output
directly as the pressure gradient (`solve.py:347-354` feeding
`blockamr/dsl/exp.py:46` → `PressureGradient.evaluate`). Nothing in
`src/NeoN/include/NeoN/blockAmr/linearAlgebra/` mentions "nodal" at all (grep for
`nodal|Nodal` over that directory and `src/blockAmr/linearAlgebra/`: no files match).
Routing the pressure step through `la::Solver` is therefore a re-derivation of the
projection scheme, not a solver swap — with multi-level AMR reachability
(§ Blocker 2) as the close second.

## Current state

### Call chain, pressure Poisson

1. `src/neofoam/solver/incompressibleFluidBlockAMR/create_fields.py:58` — `ProjectionAlgorithm.detect_and_create`.
2. `.../models/projection/base.py:26,36` — only member is `chorinProjection`; no
   alternative is selectable (`create()` raises for any other `algorithm_type`,
   `base.py:41-45`).
3. `.../models/projection/chorinProjection.py:70-73` — the model registers
   `BlockAMRSolutionConfig`, `FvSchemesConfig`, `USolutionConfig`, `PSolutionConfig`.
   `sol_p = config.p_solution_config.resolve()` (`chorinProjection.py:97`) is the
   `system/fvSolution` `solvers.p` block, carried to the solves as a plain dict.
4. `.../models/projection/chorinProjection.py:249-253` — the pressure equation:
   `Equation(imp.laplacian(dt, p) == exp.div(U))`, then `pEqn.solve(dt=dt, t=t, solution=sol_p)`.
5. `src/NeoN/src/blockAmr/python/blockamr/dsl/equation.py:97-100` — `Equation.solve`
   delegates to the free `dsl.solve.solve`.
6. `src/NeoN/src/blockAmr/python/blockamr/dsl/solve.py:49-51` — an equation with an
   `implicit_lhs` dispatches to `_solve_implicit`.
7. `dsl/solve.py:213-215` — **the solver name is validated and only `"MLMG"` is
   accepted**: `if solver_name != "MLMG": raise ValueError(...)`. The default is
   `"MLMG"` (`solve.py:213`, and `FieldSolutionConfig.solver: str = "MLMG"` at
   `src/neofoam/solver/incompressibleFluidBlockAMR/configs.py:203`).
8. `dsl/solve.py:238-306` — cached build: nodal `BoxArray` via `convert_ba(..., node_type())`
   (`:253`), nodal `phi_mf`/`rhs_mf` (`:255-259`), per-side `LinOpBCType` from
   `p.pressure_bc` (`:268-276`), agglomeration/consolidation when a Dirichlet side is
   present (`:284-288`), then `MLNodeLaplacian` — the single-level overload for
   `n_levels == 1` and the **list overload for `n_levels > 1`** (`:289-292`) — and
   `MLMG(lp)` (`:299`).
9. `dsl/solve.py:328-336` — `lp.comp_divergence(rhs, vel3)` then
   `mlmg.solve(phi, rhs, rtol, atol)`, warm-started from the previous step's `phi`
   (the cache survives across time steps; the rebuild key is
   `(n_levels, sigma, bottom_solver)`, `:234-236`).
10. `dsl/solve.py:346-354` — `mlmg.get_fluxes(...)` and `p_field.grad = [...]`.
11. Back in `chorinProjection.py:255` — `correct(U, -dt * exp.grad(p))`.
    `exp.grad` returns a `PressureGradient` because `p.grad` is set
    (`blockamr/dsl/exp.py:46-48`), whose `evaluate(lev)` is *literally* the stored
    MLMG fluxes (`exp.py:105`). The central-difference `Grad` operator
    (`blockamr/operators/grad.py`) is never used for `p`.

Native entry point: `blockamr.MLNodeLaplacian` / `blockamr.MLMG`, bound in
`src/NeoN/src/blockAmr/bindings/linop.cpp:215-274`. Not `ginkgoSolve.cpp`.

Verified side observation: `p_field.mf` is **never written** by `_solve_implicit` — the
solution lives in `cache.phi_mfs` (nodal) and only `p.grad` is published. So the `p`
handed to `PlotfileWriteHook`
(`src/neofoam/solver/incompressibleFluidBlockAMR/models/blockamr_backend.py:62`) is
whatever `CellField(..., ngrow=0, name="p")` was allocated with
(`chorinProjection.py:133`).

### Call chain, MAC projection (the second implicit solve per step)

`chorinProjection.py:220` → `blockamr/operators/mac_project.py:34` → per level
`_mac_project_level` (`:57`) → `MLABecLaplacian(geom, ba, dm, LPInfo())` with
`alpha=0, beta=1` (`:165,177`), `MLMG` (`:190`), solved at `sol_p`'s `rtol`/`atol`
(`:85-90`), corrected from `get_fluxes` (`:93-115`). Cached on the `phi` field
(`:150-155`) — note the cache key is only `lev`, so on a multi-level mesh it is rebuilt
every level every step.

### Configurability

Partially, and not toward the seam. `FieldSolutionConfig`
(`configs.py:188-219`) exposes `solver`, `rtol`, `atol`, `maxIter`, `bottomSolver`,
`ibm`, `backend`, `verbose`, `bottomVerbose`. `solver` is a free string but the only
value `_solve_implicit` accepts is `"MLMG"` (`dsl/solve.py:213-215`); there is no
`precond` field, no `gmg` block, no way to name a Krylov method. `backend`
(`"jax"|"cpp"`) selects the *explicit* kernel implementation only — it is read in
`dsl/solve.py:88` on the Forward-Euler path, not on the implicit one.

Path used, of the three: **AMReX `MLNodeLaplacian`/`MLMG` (pressure) and
`MLABecLaplacian`/`MLMG` (MAC).** The persistent `blockamr.FaceCoeffSolver` is not
used either. `la::Solver` is not used.

## Gap analysis

What would have to exist to route the pressure solve through `la::Solver`:

**Discretisation.** The seam's only operator is `ops::Laplacian`
(`src/NeoN/include/NeoN/blockAmr/operators/laplacian.hpp:91`), which writes
`upper[d](face) += -gammaFace/dx[d]^2` on a **cell-centred** 7-point stencil
(`laplacian.hpp:25-27`), with `gammaFace` the two-cell arithmetic mean. The
module docstring says outright that only one operator exists and that there is no
`ddt` — the cell-centred diagonal source is written directly with
`Matrix.diagonal_source(alpha)`
(`src/NeoN/src/blockAmr/python/blockamr/linear_algebra.py:47-49`; binding at
`ginkgoSolve.cpp:1502-1509`). So a cell-centred `sigma*p - div(grad p) = div(U)` is
expressible; the nodal projection is not. **A nodal operator does not exist** — grep
for `nodal|Nodal` across `include/NeoN/blockAmr/linearAlgebra/` and
`src/blockAmr/linearAlgebra/` matches no file.

**Hand-built coefficients.** The solver does **not** hand-build `-1.0/dx**2` face
coefficients anywhere; `grep -rn "1.0/dx\|1/dx\*\*2"` over `src/neofoam/` finds
nothing, and it does not need to — AMReX's `MLNodeLaplacian` takes a scalar `sigma`
(`dsl/solve.py:290`). The hand-built `-gamma/dx**2` convention lives only in
`src/NeoN/test/` and `src/NeoN/benchmarks/` (documented as such at
`laplacian.hpp:80-83`). On the in-flight change: the working tree's
`src/blockAmr/operators/laplacian.cpp` *unfolds the boundary* — a non-periodic domain
face now keeps its real `-gamma/dx^2` coefficient and the diagonal half of the BC moves
to the consumers (per level), because the folded term is `dx`-dependent and
`gmgRestrict` coarsened it with the wrong law. The face field still stores
`-gamma/dx^2`, not physical gamma; the measured payoff quoted in the new comment is
Dirichlet 8/8/8 iterations unfolded against 12/13/14 folded at 64/128/256^3. Either
way this does not change what a *caller* has to supply: a `gamma` MultiFab.

**Boundary conditions.** `models/bc_mapping.py:26-46` maps OpenFOAM patch specs to
`blockamr` velocity BC objects (`fixedValue`/`noSlip`/`NeumannBC`/`slip`), and
`blockamr.bc.pressure_domain_bc` (`blockamr/bc.py:197-228`) derives the *pressure*
per-side types as `(lo_bc, hi_bc)` — two length-3 lists of `LinOpBCType`
(`Periodic`/`Dirichlet`/`Neumann`), stashed on `p`/`phi`
(`chorinProjection.py:134,141`). The seam's `bc` is a 6-string list in
`xlo,xhi,ylo,yhi,zlo,zhi` order, parsed by `parseBc` into
`la::BcArray = std::array<int,6>` with exactly three values: 0 periodic, 1 homogeneous
Dirichlet, 2 homogeneous Neumann (`src/NeoN/include/NeoN/blockAmr/core/bc.hpp:22-25`).
The mapping is mechanical and lossless *for the types the solver actually produces* —
a ~10-line translation from the `LinOpBCType` pair to the 6-string list.

**Inhomogeneous data.** `laplacian(gamma, geom, bc=..., bc_data=...)`
(`linear_algebra.py:154-179`) takes a cell-centred MultiFab whose ghost layer carries
the datum — `MLMG`'s `setLevelBC` contract, the same carrier `FaceCoeffSolver` takes
(`laplacian.hpp:95-98`). The solver's pressure BCs are all homogeneous today
(`pressure_domain_bc` emits types only, no values), so `bc_data=None` would do.

**rhs / solution MultiFabs.** `la::LinearSystem(Matrix&, amrex::MultiFab& rhs)` is
non-owning (`linearSystem.hpp:42`) and `la::Solver::solve(system, amrex::MultiFab& sol)`
writes in place, seeding from `sol`'s incoming values (`solver.hpp:49`, binding at
`ginkgoSolve.cpp:1644-1647`). Both are **one MultiFab, cell-centred, one level**. The
solver would have to compute the rhs itself — `exp.div(U)` — instead of using
`lp.comp_divergence`, and compute the pressure gradient itself instead of using
`mlmg.get_fluxes`. That second item is where discrete divergence-freeness lives:
`get_fluxes` is the exact adjoint of AMReX's divergence, which is the property the
projection relies on. `mac_project.py:40-44` states the same for the MAC solve.

**Config schema.** `FieldSolutionConfig` would need at least `precond` and a nested
GMG block. `blockamr.solver_config.SolverConfig`/`GmgConfig` already exist as validated
pydantic models (`src/NeoN/src/blockAmr/python/blockamr/solver_config.py:22,108`) and
`SolverConfig.kwargs()` splats straight into the binding, so the neofoam-side work is a
mapping from the `fvSolution` dict, not a new schema. Note `solver`/`precond` are
deliberately plain strings there (parsed once in C++,
`solver_config.py:122-127`), which matches `FieldSolutionConfig`'s existing style.

## Blockers & open questions

**1. Nodal vs cell-centred discretisation — RESOLVED, blocking.**
Evidence: nodal `BoxArray`/MultiFabs at `dsl/solve.py:253-259`; `MLNodeLaplacian` at
`:290-292`; `comp_divergence` at `:328-330`; `get_fluxes` → `p.grad` at `:346-354`;
consumed as the gradient at `blockamr/dsl/exp.py:46,105`. Against:
`MatrixCoefficients { MeshLevel mesh; CellFieldLevel diag; FaceFieldLevel upper; ... }`
(`coefficients.hpp:70-78`) and `ops::Laplacian`'s cell-centred 7-point stencil
(`laplacian.hpp:25-27`). No nodal operator or nodal handle exists in
`linearAlgebra/` (grep: no matches for `nodal`). Consequence: this is a change of
*scheme*, and the acceptance oracles that would move are the physical ones
(`test/solver/incompressibleFluidBlockAMR/test_cavity_ghia.py`,
`test_cylinder_validation.py`, `test_verification_projection.py`).

**2. Multi-level AMR — RESOLVED: the seam is single-level by construction.**
`blockamr::MeshLevel` is "the layout of ONE AMR level, which is the granularity
everything in the linear algebra works at", and the header says explicitly that
Python's `Mesh`/`AmrMesh` are the multi-level containers
(`src/NeoN/include/NeoN/blockAmr/core/meshLevel.hpp:16-21`). The Python surface repeats
it: "It is NOT `blockamr.Mesh`, which is the multi-level container"
(`linear_algebra.py:17-19`). `MatrixCoefficients` holds exactly one `MeshLevel`
(`coefficients.hpp:72`); `FaceCoeffFields` holds one alpha and six face fields over one
layout (`faceCoeffMatrix.hpp:87-120`); `LinearSystem` holds one `Matrix*` and one
`MultiFab*` (`linearSystem.hpp:83-84`); `Solver::solve` takes one `MultiFab&`
(`solver.hpp:49`). `bench_solvers2.py:86-88` states the measurement scope as "Single
box, single GPU, single AMR level ... Nothing here says anything about multi-box, MPI
or composite hierarchies". Multi-box *is* supported by `MFFaceCoeffs` (`localRows()`
uses `localCount(*alpha)` over local boxes, `faceCoeffMatrix.hpp:234-237`); only
`CsrMatrix` is single-box (`linear_algebra.py:121-127`). Multi-*level* is not.

Yes, MLMG is being relied on precisely because it is multi-level: `dsl/solve.py:289-292`
picks the `MLNodeLaplacian` list overload for `n_levels > 1`, and every subsequent call
branches on `n_levels == 1` (`:327-330`, `:333-336`, `:346-349`). A composite
multi-level Ginkgo path *does* exist — `CompositeAmrexOp`
(`include/NeoN/blockAmr/linearAlgebra/matrixFree/mlmgOps.hpp:62-93`), `solveComposite`
(`.../solve/oneshot.hpp:37-52`), bound as `ginkgo_solve_composite`
(`ginkgoSolve.cpp:845`) — but it wraps an AMReX `MLLinOp`, i.e. it is Ginkgo *around*
MLMG, not the `la::Matrix` seam, and it documents that the composite operator is not
symmetric so `bicgstab`/`gmres` are the safe solvers. **Mitigation, verified:** every
shipped case is single level — `maxLevel 0` in all five of
`test/solver/incompressibleFluidBlockAMR/cases/{box,box_cpp,cavity,cylinder,cylinder_re20}/system/meshDict`.
So this blocker is real for the general code path and dormant for the tested cases.

**3. Statelessness / per-solve hierarchy rebuild — RESOLVED, quantified, and a caching
hook does not exist.** `la::Solver::solve` calls
`system.matrix().makePrecond(cfg_)` and constructs a fresh `SystemKrylovSolver` — hence
a fresh `gko` solver and fresh device work vectors — on every call
(`solver.hpp:72-87`); `KrylovSolver`'s own doc comment is explicit that these are built
"ONCE" *per instance*, "no per-call operator/solver rebuild"
(`.../solve/persistent.hpp:41-48`) — but the instance is per `solve()`. The Python
binding is one level worse: `PyLaSolver::solve` constructs the C++
`la::Solver` itself per call (`ginkgoSolve.cpp:792-795`). `Matrix::op()` is also rebuilt
per call (pinned by `test_matrix_free_op_is_rebuilt_every_call`,
`src/NeoN/test/blockAmr/test_la_matrix_formats.py:362`). Grep for `cache|reuse|Reuse`
over `solver.hpp`, `matrix.hpp`, `linearSystem.hpp`: no matches — **there is no caching
hook.** `bench_solvers2.py:77-84` documents the consequence and says the `la-cg-gmg`
wall clocks are therefore not comparable to `bench_solvers.py`'s persistent `mf-gmg`.
The parent session's measurement — +1.5 ms @64^3, +4.4 ms @128^3, +19 ms @256^3, i.e.
10-20 % of a single solve — is consistent with that note; I did not re-run it. For a
transient solver this is per *pressure* solve per step, and `chorinProjection` does two
implicit solves per step (MAC at `:220`, Poisson at `:253`), so a 256^3 run would pay
it ~2x per step where the current path pays zero (the MLMG objects are cached on the
field and warm-started, `dsl/solve.py:234-236,332-334`). A reuse hook would be needed:
either a persistent wrapper holding the built `SystemKrylovSolver` keyed on the
matrix identity, or the existing `FaceCoeffSolver`, whose hierarchy *is* built once in
its constructor.

**4. Variable coefficients — RESOLVED, supported.** `ops::Laplacian` takes a
`const amrex::MultiFab& gamma` held by pointer and read at `+=` time
(`laplacian.hpp:106-108,115`), face value = two-cell mean (`laplacian.hpp:25-27`,
pinned by `test_laplacian_face_gamma_is_the_two_cell_average`,
`src/NeoN/test/blockAmr/test_la_linear_system.py:345`). The solver only needs a
constant `sigma = dt` today (`chorinProjection.py:250`), so this is not a constraint.

**5. Non-uniform / inhomogeneous BCs — PARTIALLY RESOLVED.** Supported: one type per
domain *side* plus a per-cell inhomogeneous datum in a ghost layer
(`bc.hpp:22-25`, `laplacian.hpp:95-98`, pinned by
`test_laplacian_writes_the_inhomogeneous_datum_into_the_rhs`,
`src/NeoN/test/blockAmr/test_la_boundary_conditions.py:478`). Not supported: two
different BC *types* on one side, or a `slip`/`symmetry` type — `BcArray` has three
values only. The pressure BCs the solver derives are per-side and
Periodic/Dirichlet/Neumann only (`blockamr/bc.py:216-227`), so this is adequate for
pressure. It would not be adequate for a future implicit *velocity* solve, where
`bc_mapping.py:44` maps `slip`/`symmetry`/`symmetryPlane`.

**6. GMG as a solver, not a preconditioner — RESOLVED, by design.**
`solver="gmg"/"ir"/"mpir"` raise from `la::Solver::solve` before any hierarchy is built
(`solver.hpp:59-67`), with the reason spelled out; the Python doc repeats it
(`linear_algebra.py:41-46`). Preconditioned Krylov (`precond="gmg"/"gmg_kokkos"/"mlmg"`)
*is* reachable, built by the matrix from its own coefficients
(`coefficients.hpp:119-127`, `matrix.hpp:90-93`). For the shipped cases, MLMG-as-solver
in ~5 V-cycles (`dsl/solve.py:311-315`) is what would be given up.

**7. Reuse across time steps / regrid — OPEN.** The current path invalidates its cache
on `(n_levels, sigma, bottom_solver)` (`dsl/solve.py:234-236`); `sigma = dt`, so a
variable time step already forces a rebuild. What would settle it: whether a
`la::LinearSystem` can be held across steps with only its rhs and `gamma` rewritten
(`zero()` + re-`+=`, `linearSystem.hpp:52,64`) while a cached solver object survives —
that requires blocker 3's hook to exist first, so it cannot be answered from the code
as it stands.

**8. In-flight submodule state — OPEN (transient).** The working tree of `src/NeoN`
has uncommitted changes to `meshLevel.hpp`, `coefficients.hpp`, `faceCoeffMatrix.hpp`,
`solverConfig.hpp`, `operators/laplacian.{hpp,cpp}`, `faceCoeffOp.cpp`,
`ginkgoSolve.cpp` and `solver_config.py`. Three matter here: `ops::Laplacian` lost its
`amrex::Geometry` constructor argument (it now reads `dx` off `c.mesh`,
`laplacian.hpp:99-108`; the old 3-arg spelling is asserted un-constructible at
`src/blockAmr/linearAlgebra/coefficientsConcepts.cpp`); the boundary fold moved from
the operator to the consumers; and `gmg_precision` now defaults to `"fp32"` rather than
`"fp64"` (`solverConfig.hpp:104`, `solver_config.py:77`, `ginkgoSolve.cpp:383`). A
`faceCoeffMatrix.hpp:287-290` comment marks the stored-diagonal removal as
`PROTOTYPE (C1)`, which means `test_la_stored_diagonal.py` is presumably mid-flight
too. Any work item below should be re-checked against the committed state.

## Test coverage

Already exercising the seam, all in the submodule (`src/NeoN/test/blockAmr/`):

- `test_la_python_api.py` — the Python surface end to end: same problem through
  `MFFaceCoeffs`/`LinearSystem`/`laplacian` and through the raw binding, asserted
  bitwise (`:177`); GMG and GMG-Kokkos preconditioners beat unpreconditioned and reach
  the same answer (`:235,261,285`); the V-cycle knobs reach the hierarchy (`:305`);
  `CsrMatrix` declines `gmg` naming both (`:326`); `gmg`/`ir`/`mpir` refused (`:348`).
- `test_la_linear_system.py` — `+=` accumulates rather than assigns (`:381`), `zero()`
  clears both halves (`:408`), the system reports the matrix shape and holds the
  caller's rhs (`:428`), stats keys match `FaceCoeffSolver` (`:450`), and the assembled
  system agrees with the hand-built solver (`:313`).
- `test_la_boundary_conditions.py` — the bitwise coefficient assertion that both
  `laplacian.cpp` and `faceCoeffMatrix.hpp` name as the load-bearing guard (`:364`,
  renamed to `test_laplacian_writes_the_boundary_face_coefficient` in the working
  tree), format agreement (`:408`), agreement with the legacy solver folded (`:445`)
  and inhomogeneous (`:507`), CSR wraparound-column drop (`:574`).
- `test_la_matrix_formats.py`, `test_la_stored_diagonal.py` — format mechanics and the
  derived diagonal.
- `src/NeoN/benchmarks/blockAmr/bench_solvers2.py` — the seam vs MLMG vs
  `FaceCoeffSolver` comparison; results in the untracked root-level
  `bench_solvers2.csv`, e.g. `periodic,64,la-cg-gmg,yes,8 iters` against
  `periodic,64,legacy-mf,no,249 iters`, and MLMG as the speedup baseline (`la-cg-gmg`
  at 0.83x of MLMG at 64^3 periodic, 0.43x at 64^3 dirichlet).

Nothing in `test/` (the neofoam suite) touches the seam. The blockAMR solver tests are
`test/solver/incompressibleFluidBlockAMR/*` — `test_verification_poisson.py` is the
closest template: it solves a manufactured `laplacian(phi)=f` with `MLPoisson`+`MLMG`
directly and asserts observed 2nd order (`:101-113`).

**Smallest test that would pin a new path in the solver:** a sibling of
`test_verification_poisson.py` that solves the *same* manufactured problem
(`phi* = sin2pi(x)sin2pi(y)sin2pi(z)`, homogeneous Dirichlet on `[0,1]^3`, so the exact
solution vanishes on every face) through
`MFFaceCoeffs.symmetric(blockamr.MeshLevel(ba, dm, geom), bc=["dirichlet"]*6)` +
`Matrix.diagonal_source(zero)` + `system += laplacian(gamma, geom, bc=["dirichlet"]*6)`
+ `Solver(SolverConfig(solver="cg", precond="gmg")).solve(system, sol)`, and asserts
observed order > 1.8 via the existing `verification_helpers.observed_order`. That test
needs no solver-model change and no config change, and it is the honest gate for
"is the cell-centred seam good enough for a Poisson solve in this repo's terms" —
it deliberately does *not* claim anything about the nodal projection.

## Staged work-list

Smallest first. Each stage is independently useful and independently revertible.

1. **Pin the seam against this repo's own Poisson oracle.** Add the MMS test described
   above under `test/solver/incompressibleFluidBlockAMR/`. No `src/` change. Verifies:
   observed order > 1.8, and the `precond="gmg"` iteration count is mesh-independent
   (the `GmgConfig` docstring claims 8 at 64/128/256^3, `solver_config.py:26-31`).
2. **Map the pressure BCs to the seam's 6-string form.** A small pure function next to
   `models/bc_mapping.py` turning `pressure_domain_bc`'s `(lo_bc, hi_bc)` into
   `["periodic"|"dirichlet"|"neumann"] * 6` in `xlo,xhi,...` order, with a unit test
   in `test/solver/incompressibleFluidBlockAMR/test_bc_mapping.py`. No solve change.
   Verify: round-trips every case's `meshDict.boundary`.
3. **Route the MAC projection through the seam, behind the existing `solver` key.**
   MAC is cell-centred already (`MLABecLaplacian`, `alpha=0, beta=1` — exactly
   `Matrix.diagonal_source(0)` plus one `laplacian`), single level per call, and its
   correction needs a *face* gradient, which the seam does not produce — so this stage
   is only honest if the face gradient is computed in Python from the cell-centred
   solution, and the acceptance criterion is that `test_verification_projection.py` and
   `test_cavity_ghia.py` stay green. **Do this before touching the nodal Poisson**: it
   is the smaller of the two and it answers the adjointness question on the cheaper
   solve. Extend `FieldSolutionConfig` with `precond` only when this stage needs it.
4. **The nodal pressure Poisson.** Blocked on stage 3's answer and on a nodal
   operator that does not exist. Do not start this without a decision on whether the
   projection becomes cell-centred (approximate projection) — that is a physics choice,
   not a refactor, and it moves `test_cavity_ghia.py` / `test_cylinder_validation.py`.

Risky / unknown, called out separately rather than folded into a stage:

- **Solver reuse across calls.** Blocker 3. Needs a hook in `la::` that does not
  exist; belongs in the submodule, not in `src/neofoam/`. Until it exists, any stage
  above is a *correctness* experiment, not a performance one — and stage 3 in
  particular would regress wall clock against the cached MLMG it replaces.
- **Discrete adjointness of the pressure gradient.** The current path gets it free from
  `get_fluxes`. Replacing it means writing a gradient that is the exact adjoint of the
  divergence used for the rhs, or accepting an approximate projection. Unresolved from
  code alone.
- **Multi-level AMR.** Blocker 2. Dormant (`maxLevel 0` everywhere) but the code path
  exists and branches on `n_levels`; anything routed through the seam silently loses
  it. If a stage lands, it should raise for `n_levels > 1` rather than solve level 0
  and pretend.
- **In-flight submodule.** Blocker 8. Re-read `laplacian.cpp` /
  `faceCoeffMatrix.hpp` after the concurrent work settles; the `PROTOTYPE (C1)` marker
  and the fold move both touch what a caller must supply.
