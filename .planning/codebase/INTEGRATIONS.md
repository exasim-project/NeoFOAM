# External Integrations

**Analysis Date:** 2026-05-10

## Primary Framework Integration: OpenFOAM

**Purpose:** Source of mesh topology, field data, dictionary I/O, and parallel decomposition infrastructure. NeoFOAM is fundamentally an adapter between OpenFOAM and NeoN — all simulation cases run inside the OpenFOAM runtime.

**Integration mechanism:**
- Detected at configure time via `FOAM_SRC` env var; `cmake/OpenFOAM.cmake` imports shared libraries using `importOFLibrary()` as CMake IMPORTED targets
- Compile definitions injected: `WM_LABEL_SIZE`, `WM_DP`/`WM_SP`, `OPENFOAM=<api>`, `namespaceFoam=1`, `NoRepository`
- OF headers accessed directly from `$FOAM_SRC/<module>/lnInclude`
- Never vendored; must be system-installed and sourced at build time

**OpenFOAM libraries linked:**
- `OpenFOAM::OpenFOAM` — core OF runtime
- `OpenFOAM::finiteVolume` — `fvMesh`, `volFields`, `surfaceFields`, finite-volume operators
- `OpenFOAM::meshTools` — mesh utilities
- `OpenFOAM::incompressibleTransportModels` — transport model interfaces
- `OpenFOAM::turbulenceModels`, `OpenFOAM::incompressibleTurbulenceModels` — turbulence
- `OpenFOAM::Pstream` (from `$FOAM_LIBBIN/$FOAM_MPI/`) — MPI communication layer; also links `MPI::MPI_CXX`

**Key OF types used in NeoFOAM headers:**
- `fvMesh` (`include/NeoFOAM/datastructures/meshAdapter.hpp`)
- `volScalarField`, `volVectorField`, `surfaceScalarField` (`include/NeoFOAM/auxiliary/convert.hpp`)
- `IOdictionary`, `fvSolution`, `fvSchemes` (`include/NeoFOAM/compatibility/`)
- `processorFvPatch` (`include/NeoFOAM/auxiliary/procFaceCheck.hpp`)

**Version requirement:** OpenFOAM >= 2406; build hard-fails if `FOAM_API < 2406`

---

## Compute Backend Integration: NeoN

**Purpose:** Portable CPU/GPU computation — executors, parallel kernels, fields, DSL, linear algebra. NeoFOAM delegates all numerical work to NeoN.

**Integration mechanism:**
- Git submodule at `src/NeoN/` (tracked in `.gitmodules`)
- Added via `add_subdirectory(src/NeoN)` in `CMakeLists.txt`
- Fallback: CPM.cmake auto-fetch from `github.com/exasim-project/NeoN` if submodule not initialised
- Override: `NEOFOAM_NEON_DIR` env/cmake var for explicit path

**Repository:** `git@github.com:exasim-project/NeoN.git`

**NeoN components used:**
- `NeoN/NeoN.hpp` — main entry point included in most `src/` files
- `NeoN/core/executor/executor.hpp` — `Executor` variant (SerialExecutor/CPUExecutor/GPUExecutor)
- `NeoN/mesh/unstructured/unstructuredMesh.hpp` — NeoN mesh type constructed from OF mesh by `meshAdapter`
- `NeoN/finiteVolume/cellCentred/fields/volumeField.hpp` / `surfaceField.hpp` — NeoN field types
- `NeoN/core/dictionary.hpp` — wraps nlohmann_json for runtime config
- `NeoN/linearAlgebra/linearSystem.hpp`, `sparsityPattern.hpp` — assembled sparse systems
- `NeoN/dsl/spatialOperator.hpp` — PDE DSL operators
- `NeoN/core/mpi/environment.hpp`, `NeoN/core/mpi/operators.hpp` — MPI abstractions

---

## MPI / HPC Infrastructure

**MPI:**
- Version required: MPI 3.1+
- Enabled by default (`NEOFOAM_WITH_MPI=ON`, `NeoN_WITH_MPI=ON`)
- Located via `find_package(MPI 3.1 REQUIRED)` in `cmake/CxxThirdParty.cmake`
- Linked as `MPI::MPI_CXX` for both NeoFOAM and NeoN
- OpenFOAM's Pstream library (MPI implementation) also linked alongside standard MPI
- Test suite runs distributed tests with `mpiexec -n 3` for mesh and solver decomposition tests
- MPI thread support configurable: `NeoN_ENABLE_MPI_WITH_THREAD_SUPPORT` (default OFF)
- Compile guard: `OMPI_SKIP_MPICXX` applied to test targets to avoid OpenMPI C++ bindings conflict

**GPU Backends (via Kokkos):**
- CUDA: `Kokkos_ENABLE_CUDA=ON`, `CMAKE_CUDA_ARCHITECTURES=native`; CI image `chihtaw/cuda-openfoam-ginkgo:12.8.1-v2412-arch_89`
- HIP (AMD ROCm): `Kokkos_ENABLE_HIP=ON`; CI image `chihtaw/rocm-openfoam-ginkgo:6.4.1-v2412-arch_gfx90a`
- SYCL (Intel oneAPI): `Kokkos_ENABLE_SYCL=ON`; CI image `dheerajraghunathan/oneapi-openfoam-ginkgo:2025.3-v2412-arch_pvc`
- Serial CPU: `Kokkos_ENABLE_SERIAL=ON` (always on in all presets)
- CPU threads: `NeoN_WITH_THREADS=ON` (default)

---

## External Libraries (via NeoN)

### Kokkos 5.0.2
- **Purpose:** Platform portability layer; abstracts CUDA/HIP/SYCL/CPU thread execution and memory allocation
- **Integration:** FetchContent from `github.com/kokkos/kokkos.git` if not system-installed; version pinned in `src/NeoN/cmake/Versions.cmake`
- **Usage:** `Kokkos_Core.hpp` included directly in `src/algorithms/pressureVelocityCoupling.cpp` and throughout NeoN; `NeoN_LAMBDA` macro wraps `__host__ __device__` annotations; `parallelFor` drives all kernel launches

### Ginkgo 2.0.0
- **Purpose:** Sparse linear algebra backend; provides iterative solvers (CG, GMRES, etc.) operating on `CSRMatrix`
- **Integration:** CPM from `github.com/ginkgo-project/ginkgo`; built with CUDA/HIP/OMP/MPI flags matching Kokkos configuration
- **Usage:** Accessed through NeoN's `LinearSystem` interface in `include/NeoN/linearAlgebra/`

### nlohmann_json 3.11.3
- **Purpose:** JSON parsing for NeoN's `Dictionary` type, which is used for all runtime configuration (scheme parameters, solver settings)
- **Integration:** CPM from `github.com/nlohmann/json`
- **Usage:** `NeoN::Dictionary` wraps json internally; `include/NeoN/core/dictionary.hpp`

### fmt 12.1.0
- **Purpose:** String formatting; used in NeoN logging and error messages
- **Integration:** Always fetched via CPM from `github.com/fmtlib/fmt`
- **Usage:** `include/NeoN/core/logging.hpp`

### cpptrace 0.7.3
- **Purpose:** Human-readable C++ stack traces in error output
- **Integration:** Always fetched via CPM
- **Usage:** `include/NeoN/core/error.hpp`

### Umpire (tag 18a808d1, fork at `github.com/greole/umpire`)
- **Purpose:** Advanced memory management for GPU/CPU allocations (experimental)
- **Integration:** CPM; enabled by default (`NeoN_WITH_UMPIRE=ON`)
- **Status:** Experimental

### ADIOS2 2.10.2 (optional, default OFF)
- **Purpose:** Parallel I/O for checkpoint/restart and field output
- **Integration:** CPM from `github.com/ornladios/ADIOS2`; Kokkos patch applied at configure time
- **Enable:** `NeoN_WITH_ADIOS2=ON`

### SUNDIALS 7.5.0 (optional, default OFF)
- **Purpose:** ODE/DAE time integration (ARKODE solver)
- **Integration:** CPM from `github.com/LLNL/sundials`
- **Enable:** `NeoN_WITH_SUNDIALS=ON`

### PETSc (optional, default OFF)
- **Purpose:** Alternative sparse linear algebra backend
- **Integration:** `find_package(PkgConfig)` + `pkg_search_module(PETSc)`; system-installed only
- **Enable:** `NeoN_WITH_PETSC=ON`

### spdlog 1.16.0 (optional, default OFF)
- **Purpose:** Structured logging alternative to fmt-based logging
- **Integration:** CPM from `github.com/gabime/spdlog`
- **Enable:** `NeoN_WITH_SPDLOG=ON`

---

## Testing Integrations

### Catch2 v3.4.0
- **Purpose:** C++ unit and integration test framework; also used for benchmarks
- **Integration (NeoFOAM tests):** CPM in `cmake/CxxThirdParty.cmake`; test `catch2/` subdirectory with custom main targets `neofoam_catch_main` (serial) and `neofoam_catch_main_mpi` (parallel)
- **Integration (NeoN tests/benchmarks):** FetchContent in `benchmarks/CMakeLists.txt`
- **Test binary naming:** `neofoam_test_<name>` in `build/develop/bin/tests/`

### Kokkos Tools (optional profiling)
- **Purpose:** Runtime profiling and tracing hooks (sampling, kernel timers, PAPI)
- **Integration:** ExternalProject from `github.com/kokkos/kokkos-tools.git` (tag `33693813`)
- **Enable:** `NEOFOAM_ENABLE_KOKKOS_TOOLS=ON` (included in `develop` preset)
- **Build dir:** `build/<preset>/kokkos_tools_build/`

---

## Python Tooling

### foamlib
- **Purpose:** Python interface for reading/writing OpenFOAM case files in validation scripts
- **Integration:** pip dependency in `pyproject.toml`

### casefoam / pyfoam
- **Purpose:** OpenFOAM case setup and post-processing automation
- **Integration:** pip via `environment.yml` conda environment

### snakemake
- **Purpose:** Workflow management for running benchmark/tutorial suites
- **Integration:** conda dependency in `environment.yml`

### numpy, matplotlib
- **Purpose:** Numerical post-processing and result plotting in tutorial validation
- **Integration:** pip dependencies in `pyproject.toml`

---

## Data Formats

**Input (read by NeoFOAM from OpenFOAM cases):**
- OpenFOAM ASCII/binary field files (`0/`, `constant/`, `system/`) — parsed via OF runtime; `fvSchemes` and `fvSolution` dictionaries interpreted by `src/compatibility/`
- OpenFOAM `polyMesh/` — cell connectivity, face lists, boundary patches; converted to NeoN `UnstructuredMesh` by `src/datastructures/meshAdapter.cpp`

**Output:**
- OpenFOAM field files — NeoN results written back to OF fields via `src/auxiliary/convert.cpp` then output by OF runtime
- Auxiliary writers: `src/auxiliary/writers.cpp` — additional output helpers
- ADIOS2 parallel format (optional, when `NeoN_WITH_ADIOS2=ON`)

**Configuration:**
- OpenFOAM `fvSchemes` / `fvSolution` — mapped to NeoN `Dictionary` by `src/compatibility/`
- nlohmann_json internally within NeoN `Dictionary` — runtime configuration objects

---

## CI/CD

**GitHub Actions** (`.github/workflows/`):
- `build.yaml` — build + ctest on ubuntu-24.04 with GCC and Clang; uses `gerlero/setup-openfoam@v1` action to install OpenFOAM 2406
- `static_checks.yaml` — IWYU + clang-tidy analysis
- `changelog_check.yaml` — enforces CHANGELOG entries on PRs
- `build_doc.yaml` — Sphinx documentation build
- `trigger-lrz-gitlab-ci.yaml` — triggers LRZ HPC cluster pipeline

**GitLab CI** (`.gitlab-ci.yml`):
- Triggered from GitHub via `ci/github/trigger_pipeline.sh`
- Runs on NVIDIA, AMD, and Intel GPU runners
- Docker images provide pre-built OpenFOAM + Ginkgo environments
- Separate testing and benchmarking pipelines

---

*Integration audit: 2026-05-10*
