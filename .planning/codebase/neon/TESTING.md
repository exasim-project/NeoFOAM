# NeoN Testing

**Analysis Date:** 2026-05-10

---

## Test Strategy

Tests are unit and integration level, verifying individual containers, operators, and assembled PDE workflows. There are no separate end-to-end runner cases — integration is handled by the `NeoFOAM` layer above.

Each test binary is compiled from a single `.cpp` file. Tests cover:
- **Core containers:** `Vector`, `Array`, `View`, `SegmentedVector` — constructors, arithmetic, copy semantics
- **Primitives:** `scalar`, `Vec3`, `Tensor`, `SymmTensor`
- **Infrastructure:** `Dictionary`, `Input`, `RuntimeSelectionFactory`, executor types, `Database`/`Collection`
- **DSL:** `SpatialOperator`, `TemporalOperator`, `Expression`, `Coeff` — composition and evaluation
- **Finite Volume operators:** `GaussGreenDiv`, `LaplacianOperator`, `DdtOperator`, `SourceTerm`
- **Interpolation schemes:** `linear`, `upwind`, `surfaceInterpolation`
- **Boundary conditions:** `fixedValue`, `fixedGradient`, `symmetry`, `calculated` for both volume and surface fields
- **Linear algebra:** `CSRMatrix`, `LinearSystem`, `SparsityPattern`, `FaceToMatrixAddress`, Ginkgo solver, PETSc solver
- **Distributed (MPI):** partitioning, operator assembly and solve on distributed meshes
- **Time integration:** explicit, implicit, Runge-Kutta, ddt flux correction
- **Mesh:** `UnstructuredMesh` construction, `communicator`

---

## Test Framework

**Runner:** Catch2 (header-only, fetched by CPM at configure time)

**Custom main files:**
- `test/catch2/test_main.cpp` — for non-MPI tests; initializes `NeoN::initialize` + `NeoN::finalize` around `Catch::Session`
- `test/catch2/test_main_mpi.cpp` — for MPI tests; wraps `NeoN::mpi::Init`, spawns IO-serialization thread, calls `MPI_Allreduce` on result to propagate failures from all ranks

All test executables are linked against either `NeoN_catch_main` or `NeoN_catch_main_mpi` (CMake targets from `test/catch2/CMakeLists.txt`). A test source must **not** define its own `main`.

---

## Running Tests

```bash
# Build with tests enabled (develop preset includes NeoN_BUILD_TESTS=ON)
cmake --preset develop
cmake --build --preset develop -j4

# Run all tests via ctest
ctest --preset develop

# Run from build directory
cd build/develop
ctest

# Run a single test binary directly
./build/develop/bin/tests/neon_test_gaussGreenDiv

# Run with a Catch2 tag filter
./build/develop/bin/tests/neon_test_gaussGreenDiv "[DivOperator]"

# Run MPI tests manually
mpirun -n 3 ./build/develop/bin/tests/neon_test_operator
mpirun -n 3 ./build/develop/bin/tests/neon_test_partitioning
```

Test binary naming pattern: `neon_test_<filename-without-extension>`.

The working directory for test execution is `build/<preset>/bin/tests/`.

---

## Test File Organization

```
test/
├── catch2/                          # Shared Catch2 infrastructure
│   ├── catch2_common.hpp            # All test includes: Catch2 headers + custom matchers
│   ├── executorGenerator.hpp        # Catch2 generator for all compiled backends
│   ├── mpiGlobals.hpp / .cpp        # Global MPI rank/comm/size for MPI reporter
│   ├── mpiReporter.hpp / .cpp       # Catch2 reporter that serializes output per-rank
│   ├── mpiSerialization.hpp / .cpp  # IO serialization thread for MPI tests
│   ├── test_main.cpp                # main() for non-MPI tests
│   └── test_main_mpi.cpp            # main() for MPI tests
├── core/                            # Array, Vector, View, Dictionary, Database, executor
│   ├── vector/vector.cpp
│   ├── mpi/                         # MPI buffers and operators (non-distributed mesh)
│   └── ...
├── distributed/                     # MPI distributed operator and partitioning tests
│   ├── operator.cpp                 # Full operator assembly + solve on 3 ranks
│   └── partitioning.cpp
├── dsl/                             # DSL expression, operator, coeff tests
│   ├── common.hpp                   # Shared DSL test helpers (CreateVector, Dummy operator)
│   ├── expression.cpp
│   ├── spatialOperator.cpp
│   └── ...
├── fields/                          # Field<T> tests
├── finiteVolume/cellCentred/        # FV operators, BCs, interpolation, fields
│   ├── operator/                    # gaussGreenDiv, laplacianOperator, ddtOperator, sourceTerm
│   ├── boundary/volume/             # fixedValue, fixedGradient, symmetry
│   ├── boundary/surface/            # surfFixedValue, surfSymmetry
│   ├── interpolation/               # linear, upwind, surfaceInterpolation
│   └── fields/                      # volumeField, surfaceField
├── linearAlgebra/                   # CSRMatrix, LinearSystem, Ginkgo, PETSc, sparsity
├── mesh/unstructured/               # UnstructuredMesh, communicator
├── timeIntegration/                 # Explicit, implicit, Runge-Kutta, ddtFluxCorr
└── bindings/                        # Python binding tests
```

---

## Test Fixtures and Helpers

### `catch2_common.hpp` — `test/catch2/catch2_common.hpp`

Include this in every test file instead of individual Catch2 headers. It provides:

- All standard Catch2 includes (`catch_test_macros.hpp`, `catch_approx.hpp`, `catch_generators_all.hpp`, `catch_matchers_all.hpp`)
- `ExecutorGenerator` via `executorGenerator.hpp`
- `allAvailableExecutor()` free function
- `I<T>` alias for `std::initializer_list<T>`
- `SECTION_IF(COND, ...)` macro — enters a SECTION only when condition is true (used for rank-conditional assertions in MPI tests)
- Custom matchers and predicates (see below)

### `allAvailableExecutor()` — `test/catch2/executorGenerator.hpp`

Catch2 generator that yields all executor backends compiled into the current build. Use in every test that should run on all backends:

```cpp
auto [execName, exec] = GENERATE(allAvailableExecutor());
```

The generator always yields `SerialExecutor`, then `CPUExecutor` if OpenMP/Threads is compiled, then `GPUExecutor` if CUDA/HIP/SYCL is compiled. The `execName` string is appended to `SECTION` names to identify which backend failed.

### Custom Matchers — `test/catch2/catch2_common.hpp`

**`EqualsMatcher<Expected, Predicate>`** — element-wise range comparison that calls `copyToHost()` on device fields before comparing:

```cpp
// Example: compare mesh cell centres
REQUIRE_THAT(mesh.cellCentres(), Equals(expectedVec3s, ApproxVec3{1e-12}));

// Example: compare a scalar Vector
REQUIRE_THAT(result, Equals(I({1.0, 2.0, 3.0})));
```

**Predicates for `Equals`:**

| Predicate | Used for |
|-----------|----------|
| `ApproxScalar{margin}` | Floating-point scalar comparison with absolute tolerance (default: `1e-32`) |
| `ApproxVec3{margin}` | Component-wise Vec3 comparison |
| `EqualInt{}` | Exact equality for integers and label types |

### DSL Common Helpers — `test/dsl/common.hpp`

Provides:
- `CreateVector` functor — inserts a `VolumeField<scalar>` into a `Database`
- `CreateVolumeVector<T>` functor — generic version for any `ValueType`
- `Dummy<T>` operator — minimal `SpatialOperator` implementation for DSL expression tests
- `randomizeVector(field)` — fills a field with random values (seeded `srand(42)`)

### Mesh Factories

Test meshes are created using free functions (defined in `include/NeoN/`):
- `NeoN::createSingleCellMesh(exec)` — minimal 1-cell mesh with 4 boundaries
- `NeoN::create1DUniformMesh(exec, nCells)` — 1D slab of `nCells` hex cells with left/right patches

---

## Distributed Testing

### MPI Test Infrastructure

Distributed tests link against `NeoN_catch_main_mpi` which:
1. Calls `NeoN::mpi::Init mpi(argc, argv)` before Catch2 session
2. Stores `RANK`, `COMM_SIZE`, `IS_ROOT` in globals (`test/catch2/mpiGlobals.hpp`)
3. Spawns an IO-serialization thread on root to prevent interleaved output
4. Calls `MPI_Allreduce(MPI_IN_PLACE, &result, 1, MPI_INT, MPI_MAX, COMM)` so any rank failure propagates to the exit code
5. Uses custom `MpiReporter` (`test/catch2/mpiReporter.hpp`) registered as `"mpi"` reporter

### Declaring MPI Tests in CMake

```cmake
# test/distributed/CMakeLists.txt
if(NeoN_WITH_MPI)
  neon_unit_test(operator MPI_SIZE 3)
  neon_unit_test(partitioning MPI_SIZE 3)
endif()
```

The `NeoN_unit_test` CMake function (`test/CMakeLists.txt`) detects `MPI_SIZE` and wraps the test command with `mpirun -n <N>`. It also sets `PROCESSORS <N>` so CTest reserves sufficient resources and `TIMEOUT 10` for MPI tests.

### Rank-Conditional Assertions — `SECTION_IF`

MPI tests use `SECTION_IF` to assert rank-specific results without entering Catch2 sections on wrong ranks:

```cpp
SECTION_IF(mpiEnviron.rank() == 0, "Correct result on rank 0")
{
    REQUIRE_THAT(xPart, Equals(take(x, 0, 4)));
}
SECTION_IF(mpiEnviron.rank() == 1, "Correct result on rank 1")
{
    REQUIRE_THAT(xPart, Equals(take(x, 4, 8)));
}
```

### MPI Environment Object

For tests that need rank information but run via `NeoN_catch_main_mpi`, instantiate `NeoN::mpi::Environment mpiEnviron;` at the top of the test case to access `.rank()`, `.size()`:

```cpp
TEST_CASE("Distributed Operator")
{
    NeoN::mpi::Environment mpiEnviron;
    auto [execName, exec] = GENERATE(allAvailableExecutor());
    // ...
    SECTION_IF(mpiEnviron.rank() == 0, "Verify rank 0 partition") { ... }
}
```

### Conditional Backend Guards

Tests requiring Ginkgo use `#if NF_WITH_GINKGO` / `#endif` to skip the solver section when the backend is not compiled:

```cpp
#if NF_WITH_GINKGO
    auto solver = NeoN::la::Solver(exec, solverDict);
    // ...
#endif
```

---

## Test Registration Pattern

Every test source file must:

1. Define `CATCH_CONFIG_RUNNER` before including any Catch2 header (this is present in the `catch2_common.hpp` guard — do NOT add a `main()` function yourself)
2. Include `"catch2_common.hpp"` as the first project include
3. Include `"NeoN/NeoN.hpp"` for library access

```cpp
// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER
#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"

TEST_CASE("Feature name")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());
    // ...
}
```

For template test cases (testing `scalar` and `Vec3` simultaneously):

```cpp
TEMPLATE_TEST_CASE("OperatorName", "[template]", NeoN::scalar, NeoN::Vec3)
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());
    // TestType is NeoN::scalar or NeoN::Vec3
}
```

---

## Adding a New Test

1. Create `test/<module>/<feature>.cpp`
2. Add to the module's `CMakeLists.txt`:
   ```cmake
   neon_unit_test(<feature>)                        # non-MPI
   neon_unit_test(<feature> MPI_SIZE 3)             # MPI, 3 ranks
   ```
3. The `neon_unit_test` function sets the binary name to `neon_test_<feature>`, links Catch2 main, links `NeoN`, and registers the test with CTest

---

*Testing analysis: 2026-05-10*
