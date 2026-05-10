# Testing

**Analysis Date:** 2026-05-10

## Test Strategy

Tests verify that NeoFOAM and standard OpenFOAM produce numerically identical results for the same input case. Each test:

1. Constructs an OpenFOAM field or operator result using standard OF APIs
2. Constructs the equivalent NeoFOAM/NeoN field or operator using the adapter layer
3. Compares internal values and boundary data element-by-element with an absolute tolerance

This is a black-box correctness approach — the tests do not mock internal behaviour; they exercise the full stack (OF mesh → NeoFOAM adapter → NeoN field → comparison). Randomized field initialization is used to avoid false passes from zero-initialized fields.

MPI distributed tests additionally verify correct processor-boundary data exchange under `MPI_Alltoallv`, including rank-routing correctness and interpolation weight symmetry.

## Test Structure

**Location:** `test/`

**Test binaries:** one binary per `test_<name>.cpp` file; binary named `neofoam_test_<name>`

**Setup directories:** each test runs with its working directory set to a `test/setup_<name>/` subdirectory containing a complete OpenFOAM case (mesh, initial fields, `fvSchemes`, `fvSolution`, `controlDict`). The CMake `neofoam_unit_test()` function wires this up automatically.

**Test categories:**

| Binary | Setup Dir | MPI | What is tested |
|--------|-----------|-----|----------------|
| `neofoam_test_readDict` | `setup_operator` | No | OF↔NeoN dictionary conversion |
| `neofoam_test_unstructuredMesh` | `setup_compatibility` | No | Mesh adapter: cell counts, face centres, boundary mesh data |
| `neofoam_test_geometricFields` | `setup_operator` | No | VolumeField/SurfaceField construction and boundary conditions |
| `neofoam_test_operators` | `setup_operator` | No | Surface interpolation (linear, upwind), GaussGreen grad, div |
| `neofoam_test_stencils` | `setup_backwardDdt` | No | Finite-volume stencil construction |
| `neofoam_test_implicitOperators` | `setup_backwardDdt` | No | Implicit operator assembly |
| `neofoam_test_backwardDdtScheme` | `setup_backwardDdt` | No | Backward Euler ddt scheme |
| `neofoam_test_ddtFluxCorr` | `setup_ddtCorr` | No | ddt flux correction |
| `neofoam_test_pressureVelocityCoupling` | `setup_pressureVelocityCoupling` | No | PISO/SIMPLE loop, rAU, HbyA, pressure correction |
| `neofoam_test_advection` | `setup_advection` | No | Advection operator |
| `neofoam_test_compatibility` | `setup_compatibility` | No | fvSchemes/fvSolution dict mapping |
| `neofoam_test_momentum` | `setup_pressureVelocityCoupling` | No | Momentum equation assembly |
| `neofoam_test_distributedUnstructuredMesh` | `setup_pressureVelocityCoupling` | 3 ranks | Distributed mesh boundary layout, proc-patch weight symmetry, `communicateBoundaryData` routing |
| `neofoam_test_distributedPressureVelocityCoupling` | `setup_pressureVelocityCoupling` | 3 ranks | PISO coupling correctness on decomposed mesh |
| `neofoam_test_distributedMomentum` | `setup_pressureVelocityCoupling` | 3 ranks | Momentum equation on decomposed mesh |

## Running Tests

```bash
# Build tests first (develop preset required)
cmake --preset develop
cmake --build --preset develop -- -j4

# Run all tests
ctest --preset develop

# Run from build directory
cd /home/hendrice4work/NeoFOAM_distributed/build/develop
ctest

# Run a single test by name
ctest --preset develop -R neofoam_test_unstructuredMesh

# Run a single binary directly (serial)
./build/develop/bin/tests/neofoam_test_operators

# Run a single MPI test directly (must specify -parallel for OF)
mpirun -np 3 ./build/develop/bin/tests/neofoam_test_distributedUnstructuredMesh -parallel

# MPI tests have a 10-second timeout (set via CMake TIMEOUT property)
```

**Prerequisites:** OpenFOAM must be sourced before building or running tests:
```bash
source /path/to/OpenFOAM-2406/etc/bashrc
```

## Test Fixtures / Helpers

**`test/common.hpp`** — primary test utility header, included by all test `.cpp` files. Provides:

- `NeoFOAM::randomizeField(field)` — fills any OF field with random values and calls `correctBoundaryConditions()`
- `NeoFOAM::createRandomField<FieldType>(runTime, mesh, name, rand)` — template factory for typed random OF fields; handles `volScalarField`, `volVectorField`, `surfaceScalarField`, `surfaceVectorField`
- `NeoFOAM::randomScalarField(runTime, mesh, name)` — convenience wrapper returning random `Foam::volScalarField`
- `NeoFOAM::randomVectorField(runTime, mesh, name)` — convenience wrapper for random `Foam::volVectorField`
- `NeoFOAM::randomSurfaceScalarField(runTime, mesh, name)` — convenience wrapper for random `Foam::surfaceScalarField`
- `NeoFOAM::randDimField<FieldType>(mesh, dimensionSet, name)` — creates a dimensioned field with random entries (no `runTime` needed)
- `NeoFOAM::compare(nfField, ofField, comparator, withBoundaries)` — template comparison between NeoN fields and OF fields; iterates internal data then boundary data; processor patches are placed last in the comparison to match NeoN boundary layout
- `SECTION_IF(COND, ...)` — macro that enters a Catch2 `SECTION` only if `COND` is true; used to write rank-specific assertions in MPI tests without `#ifdef` noise

**`test/catch2/common.hpp`** — lower-level comparison helpers:

- `ApproxScalar{margin}` — functor for element-wise scalar comparison with absolute margin; used with `Catch::Matchers::RangeEquals`
- `ApproxVector{margin}` — functor for element-wise `NeoN::Vec3` vs `Foam::vector` comparison; accepts scalar margin (applied to all 3 components) or `NeoN::Vec3` margin (per-component)

**`test/catch2/executorGenerator.hpp`** — Catch2 generator for executor parametrization:

- `allAvailableExecutor()` — returns a `GeneratorWrapper` over all compiled-in executors: always `SerialExecutor`, plus `CPUExecutor` if OpenMP/threads are enabled, plus `GPUExecutor` if CUDA/HIP/SYCL is enabled
- Usage in tests:
  ```cpp
  auto [execName, exec] = GENERATE(allAvailableExecutor());
  SECTION("Some test on " + execName) { ... }
  ```

**`test/catch2/mpiGlobals.hpp`** — exposes `COMM`, `ROOT`, `RANK`, `COMM_SIZE`, `IS_ROOT` as `extern` globals for MPI tests (populated by `test_main_mpi.cpp`)

**`test/catch2/mpiReporter.hpp` / `.cpp`** — custom Catch2 streaming reporter (`MpiReporter`) that serializes assertion output across ranks; registered as `"mpi"` reporter; used automatically by `test_main_mpi.cpp`

**`test/catch2/mpiSerialization.hpp` / `.cpp`** — background IO serialization thread (`serializeIO`) used in `test_main_mpi.cpp` to prevent interleaved output from multiple MPI ranks

**`test/catch2/test_main.cpp`** — custom `main()` for serial tests; splits `argv` on `"---"` separator so Catch2 flags and OpenFOAM flags can coexist; initialises NeoN and OF before running Catch2 session; exposes `timePtr`, `argsPtr`, `meshPtr` as globals

**`test/catch2/test_main_mpi.cpp`** — custom `main()` for MPI tests; adds `MpiReporter` registration; calls `MPI_Allreduce` on result so all ranks agree on pass/fail; exposes same globals

## Test Patterns

**Executor-parametrized serial test:**
```cpp
TEST_CASE("UnstructuredMesh")
{
    Foam::Time& runTime = *timePtr;
    auto [execName, exec] = GENERATE(allAvailableExecutor());
    auto meshPtr = createMesh(exec, runTime);
    // ...
    SECTION("Internal mesh data members on " + execName)
    {
        REQUIRE(nfMesh.nCells() == ofMesh.nCells());
    }
}
```

**Rank-conditional MPI section:**
```cpp
SECTION_IF(rt.mpiEnvironment.rank() == 1, "Correct boundary Mesh on rank 1")
{
    REQUIRE(rt.nfMesh.boundaryMesh().nBoundaries() == 5);
}
```

**Field comparison pattern:**
```cpp
auto ofField = randomScalarField(runTime, mesh, "p");
auto nfField = NeoFOAM::constructFrom(rt.exec, rt.nfMesh, ofField);
// ... operate on both ...
NeoFOAM::compare(nfField, ofField, ApproxScalar(1e-15));
```

**Tolerance values used in tests:**
- Mesh geometry: `1e-16` (essentially exact)
- Linear algebra / operators: `1e-15` to `1e-13`
- Pressure-velocity coupling convergence: `1e-13` (distributed, `epsilonII`)
- Near-exact copy operations: `1e-32` (`epsilon` in coupling tests)

## Coverage Gaps

**No unit tests for individual conversion functions** (`src/auxiliary/convert.cpp`) — covered only incidentally through higher-level tests

**No tests for `writers.cpp`** — the `NeoFOAM::write()` functions that copy NeoN fields back to OF and write to disk have no dedicated test

**No tests for `foamDictionary.cpp`** — the `NeoN::Dictionary` wrapping of `Foam::IOdictionary` is exercised only through `test_readDict` and `test_compatibility`

**MPI tests pinned to 3 ranks and `CPUExecutor`** — `test_distributedUnstructuredMesh.cpp`, `test_distributedPressureVelocityCoupling.cpp`, and `test_distributedMomentum.cpp` hard-code `CPUExecutor` and require exactly 3 ranks; non-ascending neighbour-rank decompositions (e.g. scotch with 4+ ranks) are noted in comments as untested paths

**No GPU executor tests in CI** — `ExecutorGenerator` adds GPU executors only when CUDA/HIP/SYCL are compiled in; standard CI runs serial/CPU only

**`comparison.hpp` in production headers** — `include/NeoFOAM/auxiliary/comparison.hpp` is in the public include tree despite being noted as test-only (there is a TODO to move it into `test/`)
