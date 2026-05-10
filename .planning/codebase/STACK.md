# Technology Stack

**Analysis Date:** 2026-05-10

## Languages

**Primary:**
- C++20 — all NeoFOAM and NeoN source (`src/`, `include/`, `test/`, `benchmarks/`, `examples/`)
- C — required by CMake `LANGUAGES C CXX` for interop with MPI and Kokkos C internals

**Secondary:**
- Python 3.9+ — post-processing tooling (`pyproject.toml`), benchmark orchestration (`benchmarks/benchmarkSuite/`), tutorials validation
- CMake — build configuration language throughout `CMakeLists.txt` and `cmake/` modules

## Runtime

**Environment:**
- Linux (POSIX); macOS partially supported (rpath workaround); Windows not supported with MPI

**C++ Standard:**
- C++20 (`set(CMAKE_CXX_STANDARD 20)` in both `CMakeLists.txt` and `src/NeoN/CMakeLists.txt`)

**OpenFOAM:**
- Required: OpenFOAM >= 2406 (enforced by `FOAM_API >= 2406` check in `CMakeLists.txt`)
- CI uses: OpenFOAM 2406 (`openfoam2406` and `OpenFOAM-v2412`)
- Must be sourced before build: `source /path/to/OpenFOAM-2406/etc/bashrc`
- Env var `FOAM_SRC` must be set — build hard-fails if missing

## Package Manager

**C++ dependencies:**
- CPM.cmake (`cmake/CPM.cmake` and `src/NeoN/cmake/CPM.cmake`) — fetches third-party libs at configure time
- Prefers system-installed packages (`CPM_USE_LOCAL_PACKAGES=ON` by default in NeoN)
- No lockfile; versions pinned in `src/NeoN/cmake/Versions.cmake`

**Python:**
- pip / conda (`environment.yml` uses conda-forge + bioconda + pip extras)
- `pyproject.toml` defines the `neofoam` Python package (setuptools backend)

## Build System

**Generator:** Ninja (all presets use `"generator": "Ninja"`)

**CMake minimum:** 3.22.0

**Build presets** (defined in `CMakePresets.json`):
- `develop` — Debug, tests + examples + benchmarks off, Kokkos debug bounds check, IWYU/tools enabled
- `production` — Release, examples only
- `profiling` — RelWithDebInfo, benchmarks + examples, frame-pointer preserved

**Build output dirs:**
- `build/<presetName>/bin/` — executables
- `build/<presetName>/lib/` — shared libraries

**Build commands:**
```bash
cmake --preset develop
cmake --build --preset develop -- -j4
ctest --preset develop
```

## Frameworks

**Core compute — NeoN (git submodule at `src/NeoN/`):**
- Portable CPU/GPU kernel execution via Kokkos executor abstraction
- DSL for composing PDE equations (`include/NeoN/dsl/`)
- Linear algebra via Ginkgo backend
- Field types: `VolumeField<T>`, `SurfaceField<T>`, `Array<T>`, `Vector<T>`

**CFD framework — OpenFOAM (external, sourced from system):**
- Provides mesh, field, dictionary, and I/O infrastructure
- Linked as shared libraries imported from `$FOAM_SRC` / `$FOAM_LIBBIN`
- Key imported targets: `OpenFOAM::OpenFOAM`, `OpenFOAM::finiteVolume`, `OpenFOAM::meshTools`, `OpenFOAM::Pstream`

**Testing:**
- Catch2 v3.4.0 — unit and integration tests, also benchmarks
- CTest — test runner, MPI tests invoked via `mpiexec`

## Key Dependencies

**Critical (always on):**
- Kokkos 5.0.2 — platform portability layer (CPU serial, CPU threads, CUDA, HIP, SYCL backends); fetched via FetchContent if not system-installed (`src/NeoN/cmake/CxxThirdParty.cmake`)
- Ginkgo 2.0.0 (tag `6a3abf8`) — sparse linear algebra backend; fetched via CPM if not system-installed
- nlohmann_json 3.11.3 — runtime configuration via `Dictionary`; fetched via CPM if not system-installed
- fmt 12.1.0 — string formatting; always fetched via CPM
- cpptrace 0.7.3 — C++ stack traces; always fetched via CPM
- MPI 3.1+ — parallel communication; `find_package(MPI 3.1 REQUIRED)` in both `cmake/CxxThirdParty.cmake` and `src/NeoN/cmake/CxxThirdParty.cmake`; linked as `MPI::MPI_CXX`

**Optional:**
- Umpire (tag `18a808d1`) — advanced memory management (default ON in NeoN, experimental)
- ADIOS2 2.10.2 — parallel I/O (default OFF)
- SUNDIALS 7.5.0 — ODE/DAE solvers (default OFF)
- PETSc — alternative linear algebra backend (default OFF)
- spdlog 1.16.0 — structured logging (default OFF, falls back to fmt-based logging)
- Kokkos Tools (tag `33693813`) — profiling and tracing; fetched as ExternalProject when `NEOFOAM_ENABLE_KOKKOS_TOOLS=ON`
- nanobind 2.9.2 — Python bindings for NeoN (default OFF)

## Configuration

**Environment:**
- `FOAM_SRC` — path to OpenFOAM source (required at configure time)
- `FOAM_API` — OpenFOAM version integer (must be >= 2406)
- `FOAM_LIBBIN` — OpenFOAM binary library path
- `WM_LABEL_SIZE`, `WM_PRECISION_OPTION` — OpenFOAM precision/label compile definitions
- `CMAKE_CUDA_ARCHITECTURES` — set to `native` in CMakePresets.json base

**Build:**
- `cmake/OpenFOAM.cmake` — imports OF shared libraries via custom `importOFLibrary()` function
- `cmake/CxxThirdParty.cmake` — fetches Catch2 and MPI for NeoFOAM-level tests
- `src/NeoN/cmake/Versions.cmake` — all third-party version pins in one file
- `src/NeoN/cmake/CxxThirdParty.cmake` — fetches Kokkos, Ginkgo, nlohmann_json, fmt, cpptrace, Umpire, etc.
- `CMakePresets.json` — developer-facing preset definitions

## Platform Requirements

**Development:**
- Linux with OpenFOAM 2406+ sourced
- C++20-capable compiler: GCC or Clang (both tested in CI on ubuntu-24.04)
- Ninja build system
- MPI implementation (OpenMPI used in CI: `libopenmpi-dev`, `openmpi-bin`)
- Python 3.9+ (for benchmarks and tutorials)

**Production:**
- Same Linux + OpenFOAM environment
- GPU support: NVIDIA (CUDA 12.x, arch 89+), AMD (ROCm 6.4+, arch gfx90a), Intel (oneAPI 2025.3, PVC)
- CI GPU images: `chihtaw/cuda-openfoam-ginkgo:12.8.1-v2412-arch_89`, `chihtaw/rocm-openfoam-ginkgo:6.4.1-v2412-arch_gfx90a`, `dheerajraghunathan/oneapi-openfoam-ginkgo:2025.3-v2412-arch_pvc`

## Development Tools

**Formatting:**
- clang-format v17.0.6 — C++ formatting; config in `.clang-format` (LLVM-based with OpenFOAM style, 4-space indent, 100 column limit)
- cmake-format v0.6.13 — CMake file formatting

**Linting:**
- clang-tidy — static analysis; config in `.clang-tidy`; enabled via `NeoFOAM_ENABLE_CLANG_TIDY` CMake option
- cppcheck — static analysis; enabled via `NeoFOAM_ENABLE_CPP_CHECK` CMake option
- IWYU (include-what-you-use) — include hygiene; run in `static_checks.yaml` CI job; enabled via `NeoN_ENABLE_IWYU`
- typos v1.23.1 — spell checking

**Pre-commit:**
- `.pre-commit-config.yaml` hooks: check-yaml, end-of-file-fixer, trailing-whitespace, pretty-format-json, clang-format, cmake-format, cmake-lint, REUSE license check, typos

**License:**
- REUSE 6.2.0 compliance; `REUSE.toml` and `LICENSES/` directory

**CI:**
- GitHub Actions (`.github/workflows/`): build+test on ubuntu-24.04 with GCC and Clang, static checks (IWYU, clang-tidy), changelog check, doc build
- GitLab CI (`.gitlab-ci.yml`): GPU testing on NVIDIA/AMD/Intel runners via triggered pipelines from GitHub; benchmarking pipeline
- LRZ GitLab CI (`ci/lrz-gitlab/`) for HPC system testing

---

*Stack analysis: 2026-05-10*
