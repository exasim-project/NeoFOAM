# Version 0.3.0 (unreleased)
- Enable mempool via controlDict [#246](https://github.com/exasim-project/NeoFOAM/pull/246)
- Add SpalartAllmarasDDES turbulence model and integration test [#233](https://github.com/exasim-project/NeoFOAM/pull/233)
- Added continuity error calculation [#306](https://github.com/exasim-project/NeoFOAM/pull/306)
- Add distributed (processor-boundary) support for neoIcoFoam: processor-face geometry, non-orthogonal snGrad and viscous-stress correction with halo exchange, split-storage SurfaceField read/write with BC-preserving restart, and a tiltedCube distributed test case [#310](https://github.com/exasim-project/NeoFOAM/pull/310)


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
