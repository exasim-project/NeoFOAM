<!-- refreshed: 2026-05-10 -->
# Codebase Structure

**Analysis Date:** 2026-05-10

## Directory Layout

```
NeoFOAM_distributed/
├── src/
│   ├── NeoN/                         # NeoN compute backend (git submodule)
│   ├── algorithms/
│   │   └── pressureVelocityCoupling.cpp  # PISO/SIMPLE helper free functions
│   ├── auxiliary/
│   │   ├── comparison.cpp            # OF ↔ NeoN field equality (test-only)
│   │   ├── convert.cpp               # Scalar/vector/dictionary type conversion
│   │   ├── foamDictionary.cpp        # Foam::dictionary → NeoN::Dictionary
│   │   ├── setup.cpp                 # createAdapterRunTime, createExecutor, createMesh
│   │   └── writers.cpp               # NeoN field → OF file I/O
│   ├── compatibility/
│   │   ├── fvSchemes.cpp             # OF scheme names → NeoN names (Euler→BDF1)
│   │   └── fvSolution.cpp            # OF solver/preconditioner → Ginkgo names
│   ├── datastructures/
│   │   ├── foamMesh.cpp              # (legacy mesh helpers)
│   │   ├── meshAdapter.cpp           # readOpenFOAMMesh, MeshAdapter constructors
│   │   └── pdeSolver.cpp             # (placeholder; PDESolver is header-only)
│   └── fvcc/
│       └── surfaceInterpolation/
│           └── surfaceInterpolations.cpp  # OF RTST registration for "linear" scheme
├── include/
│   └── NeoFOAM/
│       ├── NeoFOAM.hpp.in            # Configured umbrella header (generated)
│       ├── algorithms/
│       │   └── pressureVelocityCoupling.hpp
│       ├── auxiliary/
│       │   ├── comparison.hpp        # operator== NeoN ↔ OF field
│       │   ├── convert.hpp           # primitive type conversion
│       │   ├── fieldTraits.hpp       # isVolumeField / isSurfaceField concepts
│       │   ├── procFaceCheck.hpp     # processor face index helpers
│       │   ├── readers.hpp           # constructFrom, constructAndRegister templates
│       │   ├── setup.hpp             # createAdapterRunTime / createExecutor decls
│       │   ├── typeConversion.hpp    # TypeMap<FoamType> trait specializations
│       │   └── writers.hpp           # write() overloads for NeoN→OF I/O
│       ├── compatibility/
│       │   ├── fvSchemes.hpp         # mapFvSchemes, updateDdtSchemes
│       │   └── fvSolution.hpp        # mapFvSolution, updateSolver, updatePreconditioner
│       └── datastructures/
│           ├── meshAdapter.hpp       # MeshAdapter : Foam::fvMesh; readOpenFOAMMesh()
│           ├── pdeSolver.hpp         # PDESolver<ValueType> template class
│           └── runTime.hpp           # RunTime struct; readOrCreate<T>()
├── examples/
│   └── neoIcoFoam/
│       ├── neoIcoFoam.cpp            # Main solver application (PISO loop)
│       └── createFields.H            # OF field reads (ofP, ofU, ofPhi)
├── test/
│   ├── common.hpp                    # Catch2 helpers, compare(), randomXxxField()
│   ├── catch2/                       # Catch2 MPI reporter, serialization, test_main
│   ├── test_advection.cpp
│   ├── test_backwardDdtScheme.cpp
│   ├── test_compatibility.cpp
│   ├── test_ddtFluxCorr.cpp
│   ├── test_distributedMomentum.cpp
│   ├── test_distributedPressureVelocityCoupling.cpp
│   ├── test_distributedUnstructuredMesh.cpp
│   ├── test_geometricFields.cpp
│   ├── test_implicitOperators.cpp
│   ├── test_momentum.cpp
│   ├── test_operators.cpp
│   ├── test_pressureVelocityCoupling.cpp
│   ├── test_readDict.cpp
│   ├── test_stencils.cpp
│   ├── test_unstructuredMesh.cpp
│   └── setup_*/                      # Per-test OpenFOAM case directories
├── tutorials/
│   ├── cylinder2D/                   # 2D cylinder tutorial case
│   └── cylinder3D/                   # 3D cylinder tutorial case
├── benchmarks/
│   └── benchmarkSuite/
│       └── templates/                # 2DSquare, 3DCube benchmark cases
├── cmake/                            # CMake helper modules (OpenFOAM.cmake, etc.)
├── ci/                               # CI pipeline configs (GitHub, LRZ GitLab)
├── doc/                              # Sphinx documentation source
├── CMakeLists.txt                    # Top-level build config
├── CMakePresets.json                 # develop / production / profiling presets
├── CLAUDE.md                         # Project instructions for Claude Code
└── environment.yml                   # Conda environment spec
```

## Key Modules

### `src/datastructures/` + `include/NeoFOAM/datastructures/`

The foundational layer. All other modules depend on it.

- `meshAdapter.hpp` / `meshAdapter.cpp` — `MeshAdapter : Foam::fvMesh` inherits the full OpenFOAM mesh and stores a `NeoN::UnstructuredMesh nfMesh_` built by `readOpenFOAMMesh()`. The conversion flattens patch data into contiguous arrays with the invariant: non-processor patches first, processor patches appended at the tail. Provides `nfMesh()` and `exec()` accessors.
- `runTime.hpp` — `RunTime` struct is the top-level context object passed through the adapter layer. Fields: `db`, `meshPtr`, `mesh`, `nfMesh`, `exec`, `t`, `dt`, `fvSchemesDict`, `fvSolutionDict`, `mpiEnvironment`. `readOrCreate<T>()` provides lazy object registration.
- `pdeSolver.hpp` — `PDESolver<ValueType>` template wraps `dsl::Expression` + `la::LinearSystem`. Key methods: `assemble()`, `solve()`, `solve(rhsOp)`. Inner functor `SetReference` pins the pressure reference cell on rank 0.

### `src/auxiliary/` + `include/NeoFOAM/auxiliary/`

Conversion utilities. No solver logic.

- `typeConversion.hpp` — `TypeMap<FoamType>` trait: maps `Foam::volScalarField` → `fvcc::VolumeField<NeoN::scalar>`, etc. Used by all template readers/writers.
- `readers.hpp` — `constructFrom<FoamFieldType>()` builds a NeoN field from an OF field. `constructAndRegister()` also registers it in a `VectorCollection`. `fromFoamField()` does the raw data copy via `reinterpret_cast`.
- `writers.hpp` / `writers.cpp` — `write()` overloads for `VolumeField<scalar>`, `VolumeField<Vec3>`, `SurfaceField<scalar>`, `SurfaceField<Vec3>`. Copies NeoN device data to host, patches boundary data into a temporary `Foam::GeometricField`, calls `field.write()`.
- `convert.hpp` / `convert.cpp` — Primitive conversions: `NeoN::Vec3 ↔ Foam::vector`, `NeoN::Dictionary ↔ Foam::dictionary`, `NeoN::TokenList ↔ Foam::ITstream`. The `insert<T>()` template handles type-safe dictionary entry insertion.
- `setup.hpp` / `setup.cpp` — `createAdapterRunTime()` assembles the `RunTime` struct; `createExecutor()` maps `"Serial"/"CPU"/"GPU"/"default"` strings to `NeoN::Executor` variants (with optional Umpire allocator).
- `comparison.hpp` — `operator==` for NeoN ↔ OF field comparison; intended for test assertions only.
- `procFaceCheck.hpp` — Helpers for processor face index validation.

### `src/compatibility/` + `include/NeoFOAM/compatibility/`

Configuration translation. Depends only on `NeoN::Dictionary`.

- `fvSchemes.cpp` — `mapFvSchemes()` calls `updateDdtSchemes()` which renames `Euler`→`BDF1` and `backward`→`BDF2` in the `ddtSchemes` subdictionary.
- `fvSolution.cpp` — `mapFvSolution()` calls `updateSolver()` (maps `PCG`→`solver::Cg`, etc.), `updatePreconditioner()` (maps `DIC`→`preconditioner::Ic`, wraps in `preconditioner::Schwarz` if MPI distributed), and `updateCriteria()` (maps `tolerance`/`relTol`/`maxIter` to Ginkgo `criteria` subdictionary).

### `src/algorithms/` + `include/NeoFOAM/algorithms/`

PISO/SIMPLE free functions. Depends on `PDESolver` and NeoN finite-volume types.

- `pressureVelocityCoupling.hpp` / `pressureVelocityCoupling.cpp` — `constrainHbyA()`, `computeRAU()`, `computeRAUandHByA()`, `updateFaceVelocity()`, `updateVelocity()`, `flux()`. All use `NeoN::parallelFor` kernels and operate on `NeoN::finiteVolume::cellCentred::VolumeField<T>` and `SurfaceField<T>`.

### `src/fvcc/surfaceInterpolation/`

- `surfaceInterpolations.cpp` — Registers `linear` in OpenFOAM's RunTimeSelectionTable via `addNamedToRunTimeSelectionTable`. This is the only use of OF's RTST in NeoFOAM.

### `examples/neoIcoFoam/`

- `neoIcoFoam.cpp` — Complete solver application. Calls `createAdapterRunTime()`, reads fields, builds `PDESolver` expressions using NeoN DSL, runs PISO loop, writes results. Serves as the canonical usage example.
- `createFields.H` — Included into `neoIcoFoam.cpp`; reads `ofP`, `ofU`, `ofPhi` from the OF case directory using `Foam::IOobject`.

### `test/`

- `common.hpp` — Shared test utilities: `randomizeField()`, `randomScalarField()`, `randomVectorField()`, `randomSurfaceScalarField()`, and a template `compare()` function that validates NeoN field data against OF reference data including boundary values.
- `catch2/` — MPI-aware Catch2 extensions: `mpiGlobals`, `mpiReporter`, `mpiSerialization`, dual `test_main` entries for serial and MPI runs.
- Per-test case directories (`setup_advection/`, `setup_operator/`, `setup_pressureVelocityCoupling/`, etc.) — Minimal OpenFOAM cases providing mesh and initial conditions for integration tests. `setup_pressureVelocityCoupling/` includes pre-decomposed `processor0/` – `processor2/` directories for MPI tests.

### `src/NeoN/` (git submodule)

NeoN provides the entire portable compute layer. Key namespaces used in NeoFOAM:
- `NeoN::finiteVolume::cellCentred` (alias `fvcc`) — `VolumeField<T>`, `SurfaceField<T>`, `VolumeBoundary<T>`, `SurfaceBoundary<T>`, `VectorCollection`, `GaussGreenGrad`, `SurfaceInterpolation<T>`
- `NeoN::dsl` — `Expression<T>`, `imp::ddt`, `imp::div`, `imp::laplacian`, `exp::grad`, `exp::div`
- `NeoN::la` — `LinearSystem`, `CSRMatrix`, `Solver`, `SolverStats`
- `NeoN` (core) — `Executor` variants, `Database`, `Dictionary`, `mpi::Environment`, `CommunicationPattern`, `parallelFor`

## Entry Points

**Main Solver:**
- `examples/neoIcoFoam/neoIcoFoam.cpp` — `int main()`. Requires a sourced OpenFOAM environment and a valid case directory. Build target produced by `examples/neoIcoFoam/CMakeLists.txt`.

**Test Binaries:**
- `build/develop/bin/tests/neofoam_test_<name>` — one binary per `test/test_*.cpp`. Run individually or via `ctest --preset develop`.

## Configuration Files

**Build:**
- `CMakeLists.txt` — Top-level; resolves NeoN (submodule → CPM fallback), configures options, requires `FOAM_SRC` env var.
- `CMakePresets.json` — Defines `develop` (Debug + tests), `production` (Release), `profiling` (RelWithDebInfo + benchmarks) presets. Build dir is `build/<presetName>/`.
- `cmake/OpenFOAM.cmake` — Locates OpenFOAM headers/libraries from `FOAM_SRC`.

**OpenFOAM Case (per test/tutorial):**
- `system/controlDict` — Must include `executor` and `allocator` keys (NeoFOAM extensions) in addition to standard OF entries.
- `system/fvSchemes` — Standard OF format; NeoFOAM translates at runtime via `mapFvSchemes()`.
- `system/fvSolution` — Standard OF format; NeoFOAM translates at runtime via `mapFvSolution()`. Alternatively, solver entries may include a `configFile` key pointing to a Ginkgo JSON config for fine-grained control (bypasses mapping).
- `system/decomposeParDict` — Required for decomposed MPI test setups.

## Naming Conventions

**Files:**
- Headers mirror source: `include/NeoFOAM/<module>/<name>.hpp` paired with `src/<module>/<name>.cpp`.
- Test binaries named `test_<feature>.cpp` for integration tests and `test_distributed<Feature>.cpp` for MPI variants.
- Test case directories named `setup_<featureName>/`.

**Types:**
- NeoFOAM adapter classes use PascalCase: `MeshAdapter`, `PDESolver`, `RunTime`.
- Free functions use camelCase: `readOpenFOAMMesh`, `constructFrom`, `createAdapterRunTime`.
- NeoN types accessed via namespace aliases: `fvcc::VolumeField<T>`, `dsl::Expression<T>`, `la::LinearSystem<T>`.

## Where to Add New Code

**New OF↔NeoN field type conversion:**
- Add `TypeMap<>` specialization: `include/NeoFOAM/auxiliary/typeConversion.hpp`
- Add any new primitive conversion: `include/NeoFOAM/auxiliary/convert.hpp` + `src/auxiliary/convert.cpp`
- Reuse `constructFrom` / `constructAndRegister` templates in `include/NeoFOAM/auxiliary/readers.hpp` if the field type fits the existing `TypeMap` pattern.

**New solver algorithm (e.g. SIMPLE loop):**
- Implement as free functions in `src/algorithms/<name>.cpp` and declare in `include/NeoFOAM/algorithms/<name>.hpp`.
- Follow the pattern in `pressureVelocityCoupling.cpp`: accept `PDESolver<T>` and NeoN field references; use `NeoN::parallelFor` for kernels; handle processor-boundary faces explicitly in a third `parallelFor` over `[nInt+nBnd, nTotal)`.

**New scheme/solver name mapping:**
- Add entry to the static maps in `src/compatibility/fvSchemes.cpp` or `src/compatibility/fvSolution.cpp`.
- For distributed preconditioners, add a corresponding entry to `distributedPreconditionerMap` in `fvSolution.cpp`.

**New test:**
- Add `test/test_<feature>.cpp` with Catch2 `TEST_CASE`.
- Create `test/setup_<feature>/` with a minimal OF case (mesh + initial fields + `system/` dicts).
- Register the binary in `test/CMakeLists.txt`.

**New tutorial:**
- Add `tutorials/<caseName>/` following standard OpenFOAM directory layout.
- Include `executor` and `allocator` keys in `system/controlDict`.

---

*Structure analysis: 2026-05-10*
