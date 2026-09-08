# Version 0.3.0 (unreleased)
- Add stretchedVortex neoIcoFoam tutorial: a strained (Burgers) vortex initialised and driven from the analytic solution via setExprFields/setExprBoundaryFields, plus a coreHistory.py post-processing script that checks the core contraction and swirl amplification against the analytic viscous equilibrium [#XXX](https://github.com/exasim-project/NeoFOAM/pull/XXX)
- Add neoSimpleFoam steady-state incompressible SIMPLE/SIMPLEC solver with motorBike tutorial, SIMPLEC rAtU/flux corrections, inletOutlet BC parsing, bounded-scheme prefix stripping, and CI smoke test [#362](https://github.com/exasim-project/NeoFOAM/pull/362)
- Add kEpsilon and kOmegaSST turbulence models [#337](https://github.com/exasim-project/NeoFOAM/pull/337)
- Add neoPimpleFoam solver with PIMPLE outer-loop control and SA-DDES turbulence support [#337](https://github.com/exasim-project/NeoFOAM/pull/337)
- Wire equation and field under-relaxation (URF) into PDESolver [#337](https://github.com/exasim-project/NeoFOAM/pull/337)
- Enable mempool via controlDict [#246](https://github.com/exasim-project/NeoFOAM/pull/246)
- Add SpalartAllmarasDDES turbulence model and integration test [#233](https://github.com/exasim-project/NeoFOAM/pull/233)
- Added continuity error calculation [#306](https://github.com/exasim-project/NeoFOAM/pull/306)
- Add distributed (processor-boundary) support for neoIcoFoam: processor-face geometry, non-orthogonal snGrad and viscous-stress correction with halo exchange, split-storage SurfaceField read/write with BC-preserving restart, and a tiltedCube distributed test case [#310](https://github.com/exasim-project/NeoFOAM/pull/310)
- Added forceCoeffs functionObject and required infrastructure e.g. IO helpers [#265](https://github.com/exasim-project/NeoFOAM/pull/265)

## Development
- Update submodule regularly by dependabot [#209](https://github.com/exasim-project/NeoFOAM/pull/209)
- Allow auto grabbing version from submodule without initialization and update the documentation [#210](https://github.com/exasim-project/NeoFOAM/pull/210)

## Fixes
- Fix spurious bad_any_cast errors when reading fixedValue boundaries [#194](https://github.com/exasim-project/NeoFOAM/pull/194)
- Distributed/restart robustness: preserve OpenFOAM BC types and promote vector BC components on read for restart; default the smoothSolver preconditioner to diagonal (avoids a ParIc FPE on the non-symmetric momentum matrix) [#310](https://github.com/exasim-project/NeoFOAM/pull/310)

# Version 0.2.0 (2025.12.01)
- Use NeoN logging functionality [#144](https://github.com/exasim-project/NeoFOAM/pull/144)
- Add support for PDEs on vector fields [#119](https://github.com/exasim-project/NeoFOAM/pull/119)[#134](https://github.com/exasim-project/NeoFOAM/pull/134)
- Improve solver interface with neon [#114](https://github.com/exasim-project/NeoFOAM/pull/114)
- Add basic runtime class [#113](https://github.com/exasim-project/NeoFOAM/pull/113)
- Time integrator: integrates the newest dsl version 0.1 into NeoFOAM #41 [#14](https://github.com/exasim-project/NeoFOAM/pull/14)
- Convert foam dictionary to neofoam dictionary #13  [#13](https://github.com/exasim-project/NeoFOAM/pull/13)
