# NeoN Integrations

**Analysis Date:** 2026-05-10

## Compute Backends (Kokkos)

**Version:** 5.0.2

**Integration mechanism:** CPM.cmake auto-fetch (`cmake/CxxThirdParty.cmake`) or system install. CMake first tries `find_package(Kokkos 5.0.2 QUIET)`; if not found, fetches from `https://github.com/kokkos/kokkos.git`.

**How Kokkos is used:**
- `GPUExecutor::exec = Kokkos::DefaultExecutionSpace` (`include/NeoN/core/executor/GPUExecutor.hpp`)
- `CPUExecutor::exec = Kokkos::DefaultHostExecutionSpace` (OpenMP or Threads)
- `SerialExecutor::exec = Kokkos::Serial`
- `NEON_LAMBDA` macro wraps lambdas with `KOKKOS_LAMBDA` for device compatibility
- `parallelFor(exec, range, NEON_LAMBDA(...){})` dispatches Kokkos parallel loops
- Memory views use `Kokkos::View`; `Kokkos::deep_copy` handles host↔device transfers

**Ginkgo↔Kokkos bridge:** `ginkgo/extensions/kokkos.hpp` — Ginkgo executor is created from NeoN's Kokkos executor context (`src/linearAlgebra/ginkgo/ginkgo.cpp`, function `getGkoExecutor`)

**Backend detection:** `cmake/AutoEnableDevice.cmake`
- CUDA detected via `check_language(CUDA)` → `Kokkos_ENABLE_CUDA=ON`
- HIP detected via `check_language(HIP)` → `Kokkos_ENABLE_HIP=ON`
- SYCL: passed through via `Kokkos_ENABLE_SYCL` (from Umpire options), not auto-detected
- OpenMP/Threads: selected via `NeoN_WITH_OMP` / `NeoN_WITH_THREADS` (default: Threads)

**Compile guard:** `NN_WITH_KOKKOS=1` on `NeoN_public_api`

---

## Linear Solvers

### Ginkgo (default)

**Version:** 2.0.0 (git tag `6a3abf8c920228006f3b28bc3bf04fc7a5f6aee0`)

**Enabled by:** `NeoN_WITH_GINKGO=ON` (default ON)

**Compile guard:** `NF_WITH_GINKGO` preprocessor macro

**Headers:**
- `include/NeoN/linearAlgebra/ginkgo.hpp` — `GinkgoSolver` class, `getGkoExecutor()`, `parse()`
- Uses `ginkgo/ginkgo.hpp`, `ginkgo/extensions/kokkos.hpp`, `ginkgo/extensions/config/json_config.hpp`

**Solver class:** `NeoN::la::ginkgo::GinkgoSolver` — registered in `SolverFactory` via `RuntimeSelectionFactory` pattern. Solver type and parameters are configured via `Dictionary` (nlohmann_json pnode under the hood).

**Distributed Ginkgo:** `src/linearAlgebra/ginkgo/ginkgoDistributed.cpp` — MPI-distributed solve support. Built with `GINKGO_BUILD_MPI=${NeoN_WITH_MPI}` and `GINKGO_BUILD_CUDA/HIP` matching Kokkos backends.

**L1 stopping criterion:** `src/linearAlgebra/ginkgo/ginkgoL1Stop.cpp` — custom convergence criterion.

**Why nlohmann_json is linked:** Ginkgo's JSON config extension requires it; linked as `nlohmann_json::nlohmann_json` on `NeoN_public_api` when Ginkgo is enabled.

### PETSc (optional)

**Enabled by:** `NeoN_WITH_PETSC=ON` (default OFF)

**Detection:** `find_package(PkgConfig)` + `pkg_search_module(PETSc REQUIRED IMPORTED_TARGET PETSc)` — must be system-installed

**Compile guard:** `NF_WITH_PETSC`

**Headers:**
- `include/NeoN/linearAlgebra/petsc.hpp` — `petscSolver` class
- `include/NeoN/linearAlgebra/petscSolverContext.hpp`
- Uses `petscvec_kokkos.hpp`, `petscmat.h`, `petscksp.h`

**Note:** When PETSc is used, it provides its own Kokkos build — `Kokkos_ROOT` is set to `PETSc_PREFIX` to force consistency.

---

## MPI / Distributed

**MPI standard:** 3.1 minimum

**Enabled by:** `NeoN_WITH_MPI=ON` (default ON); disabled on Windows

**Compile guard:** `NF_WITH_MPI_SUPPORT` — all MPI code is `#ifdef`-guarded

**Thread support:** Optional `NeoN_ENABLE_MPI_WITH_THREAD_SUPPORT=ON` → uses `MPI_Init_thread(..., MPI_THREAD_MULTIPLE, ...)` instead of `MPI_Init`

**Core MPI abstractions** (`include/NeoN/core/mpi/`):
- `environment.hpp` — `NeoN::mpi::Environment`: wraps `MPI_Comm`, provides `rank()`, `sizeRank()`, init/finalize RAII
- `halfDuplexCommBuffer.hpp` / `fullDuplexCommBuffer.hpp` — non-blocking point-to-point buffers for halo exchange
- `operators.hpp` — `allReduce()` for scalars and `Vec3`; `isend/irecv` wrappers

**MPI sources compiled into NeoN** (when `NeoN_WITH_MPI=ON`):
- `src/core/mpi/halfDuplexCommBuffer.cpp`
- `src/mesh/unstructured/communicator.cpp`

**Distributed mesh support:**
- `include/NeoN/mesh/unstructured/communicator.hpp` — processor-boundary face communication
- `include/NeoN/distributed/communicationPattern.hpp` — maps halo cells to MPI ranks
- `include/NeoN/distributed/partitioning.hpp` — simple uniform 1D partitioning helper (`partitionVolField`)

**Ginkgo MPI:** Ginkgo is built with `GINKGO_BUILD_MPI=ON` when `NeoN_WITH_MPI=ON`, enabling distributed linear solves.

**Test MPI support:** Tests using MPI are registered with `NeoN_unit_test(... MPI_SIZE N)` in `test/CMakeLists.txt` and link against `NeoN_catch_main_mpi` (`test/catch2/test_main_mpi.cpp`). CI sets `MPIEXEC_MAX_NUMPROCS=3`.

---

## Python Bindings

**Library:** nanobind 2.9.2

**Package name:** `neon` (module) / `neon_pde` (PyPI package)

**Enabled by:** `NeoN_BUILD_PYTHON_BINDINGS=ON` (default OFF; auto-enabled when built via scikit-build-core)

**Python build system:** scikit-build-core ≥0.11.0 (`pyproject.toml`)

**Installation:**
```bash
pip install scikit-build-core nanobind
pip install -e . --no-build-isolation  # editable mode
```

**Exposed surface** (`src/bindings/`):
| Binding file | What is exposed |
|---|---|
| `executors.cpp` | `SerialExecutor`, `CPUExecutor`, `GPUExecutor`, `Executor` variant helpers |
| `vectors.cpp` | `Vector<scalar>`, `Vector<Vec3>` |
| `volumeField.cpp` | `VolumeField<scalar>`, `VolumeField<Vec3>` |
| `surfaceField.cpp` | `SurfaceField<scalar>`, `SurfaceField<Vec3>` |
| `unstructuredMesh.cpp` | `UnstructuredMesh` |
| `boundaryMesh.cpp` | `BoundaryMesh` |
| `dsl.cpp` | DSL solve entry point |
| `linearAlgebra.cpp` | `LinearSystem`, solver interface |
| `inputs.cpp` | `Dictionary`, `Input`, `TokenList` |
| `database/` | `Database`, `Document`, `Collection` |
| `vec3.cpp` | `Vec3` primitive |
| `scalar.cpp` | scalar type |
| `coNum.cpp` | Courant number utility |
| `surfaceInterpolation.cpp` | surface interpolation schemes |
| `initialization.cpp` | Kokkos init/finalize wrappers |
| `containerFreeFunctions.cpp` | `fill`, `map`, reduce operations |

**Python tests** (`test/bindings/`): pytest-based, covering executors, vectors, fields, mesh, DSL, database. Run via `ctest` alongside C++ tests.

**numpy dependency:** `numpy>=1.19.0` — used in Python tests for array comparison

---

## Testing Infrastructure

**Framework:** Catch2 3.4.0

**Enabled by:** `NeoN_BUILD_TESTS=ON` (included in `develop` preset)

**Test binary naming:** `neon_test_<filename>` (built to `build/<preset>/bin/tests/`)

**Run commands:**
```bash
ctest --preset develop          # all tests
ctest --preset develop -E bench # skip benchmarks
./build/develop/bin/tests/neon_test_<name> "[tag]"  # single test with filter
mpirun -n N ./build/develop/bin/tests/neon_test_<mpi-test>  # MPI tests
```

**MPI test harness** (`test/catch2/`):
- `test_main.cpp` — serial Catch2 main
- `test_main_mpi.cpp` — MPI-aware Catch2 main; initialises MPI environment
- `mpiReporter.cpp` — serializes test results across ranks
- `mpiSerialization.cpp` — output serialization for rank 0 reporting
- `mpiGlobals.cpp` — shared MPI state for tests

**Test structure** (`test/`):
- `core/` — array, vector, dictionary, database, executor, memory
- `dsl/` — expression, spatial/temporal operators, coefficient
- `finiteVolume/` — boundary conditions, operators (div, grad, laplacian, ddt), interpolation
- `linearAlgebra/` — matrix, CSR, sparsity pattern, Ginkgo solver, PETSc solver
- `mesh/` — unstructured mesh, boundary mesh
- `distributed/` — partitioning, distributed operators
- `fields/` — field types
- `timeIntegration/` — Runge-Kutta
- `bindings/` — Python binding tests (pytest)

**Benchmarks:** `benchmarks/` — Catch2-based, enabled by `NeoN_BUILD_BENCHMARKS=ON` (`profiling` preset)

---

## Optional I/O Integration (ADIOS2)

**Version:** 2.10.2

**Enabled by:** `NeoN_WITH_ADIOS2=ON` (default OFF)

**Purpose:** Parallel I/O for field checkpoint/restart; built with Kokkos support (patch applied at fetch time via `cmake/patches/adios2_kokkos.patch`)

**Status:** Optional/experimental — no headers currently reference ADIOS2 in `include/NeoN/`

---

## Optional Time Integration (SUNDIALS)

**Version:** 7.5.0

**Enabled by:** `NeoN_WITH_SUNDIALS=ON` (default OFF)

**Components built:** ARKODE only (explicit/implicit Runge-Kutta); CVODE, IDA, KINSOL all disabled

**Kokkos NVector:** `SUNDIALS_BUILD_KOKKOS=ON` when enabled

**Status:** Experimental; `src/timeIntegration/rungeKutta.cpp` is the implementation entry point

---

*Integration audit: 2026-05-10*
