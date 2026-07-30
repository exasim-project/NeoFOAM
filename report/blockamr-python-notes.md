# blockAMR Python package — relocated notes

Rationale that used to live as long comment blocks in
`src/NeoN/src/blockAmr/python/blockamr/`. Each site now carries a one-line
conclusion and points here.

## Nodal pressure projection: BCs, agglomeration and the bottom solver

*(from `blockamr/dsl/solve.py`, `_solve_implicit`)*

The nodal pressure solve is `imp.laplacian(sigma, p) == exp.div(U)`, driven by
AMReX `MLNodeLaplacian` + `MLMG` (`solution["solver"]` accepts only `"MLMG"`).
The RHS is NODAL (`compDivergence` of a cell-centred, ghost-filled ncomp=3
velocity MultiFab) and `getFluxes` returns the CELL-CENTRED gradient stored on
`p.grad` for `correct()`.

**Per-face pressure BC.** The solver-derived spec stashed on the pressure field
by `bc.pressure_domain_bc` is used when present (outflow face → Dirichlet,
inlet/wall → Neumann); otherwise the periodic/all-Neumann default applies.

**Agglomeration + consolidation.** A lone outflow-Dirichlet face anchoring an
otherwise-Neumann domain is badly conditioned for plain nodal multigrid — the
coarse-grid correction is ineffective and convergence stalls. Agglomeration plus
consolidation let AMReX coarsen far enough for an effective bottom solve; this is
the standard incflo nodal-projection setup. It is enabled *only* when a Dirichlet
face is present, so the periodic/closed (all-Neumann) path is untouched.

**Bottom solver.** `solution["bottomSolver"]` defaults to `None`, which lets
AMReX pick its Krylov default; with the agglomeration above that converges the
nodal projection in ~5 V-cycles. Do NOT force `"smoother"`: measured at ~600
iterations per solve, ~100x the Krylov default, and it dominated runtime.
`set_bottom_solver` is sticky across calls that omit it, which is why
`bottomSolver` is part of the cache rebuild key.
