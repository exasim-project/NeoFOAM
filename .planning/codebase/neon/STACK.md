# NeoN Tech Stack

**Analysis Date:** 2026-05-10

## Languages

**Primary:**
- C++20 — all library source and headers (`include/NeoN/`, `src/`)
  - `CMAKE_CXX_STANDARD 20` enforced, extensions OFF
  - Lambda templates, `std::variant`, concepts used throughout

**Secondary:**
- Python 3.9+ — bindings package `neon_pde` (`src/bindings/`, `test/bindings/`)
  - Requires Python >=3.9 per `pyproject.toml`

## Runtime

**Environment:**
- Linux (primary), macOS, Windows (STATIC lib mode)
- MPI 3.1+ required when `NeoN_WITH_MPI=ON` (default ON)

**Package Manager:**
- C++: CPM.cmake (`cmake/CPM.cmake`) — auto-fetches all C++ deps at configure time
  - `CPM_USE_LOCAL_PACKAGES=ON` by default (prefers system packages)
- Python: `uv` (CI uses `astral-sh/setup-uv`) / `pip` with scikit-build-core

## Build System

**Generator:** Ninja (all presets)

**CMake minimum:** 3.22.0

**Presets** (`CMakePresets.json`):
| Preset | Build Type | Key Flags |
|--------|-----------|-----------|
| `develop` | Debug | Tests ON, warnings ON, `Kokkos_ENABLE_DEBUG=ON`, `Kokkos_ENABLE_DEBUG_BOUNDS_CHECK=ON` |
| `production` | Release | Tests OFF, warnings suppressed |
| `profiling` | RelWithDebInfo | Benchmarks ON, `-fno-omit-frame-pointer` |

All presets enable `Kokkos_ENABLE_SERIAL=ON` and disable `NeoN_BUILD_PYTHON_BINDINGS` by default.

**Build directories:** `build/<presetName>/`

**Python build backend:** scikit-build-core (`pyproject.toml`) with `cmake --preset=production`

## Key Dependencies

**Required (always on):**
- **Kokkos** 5.0.2 — platform portability layer; GPU/CPU kernel dispatch; central to the `Executor` abstraction (`include/NeoN/core/executor/`)
- **cpptrace** 0.7.3 — stack trace support in error/exception handling (`include/NeoN/core/error.hpp`, `dictionary.hpp`)
- **fmt** 12.1.0 — string formatting; used in logging (`include/NeoN/core/logging.hpp`)

**Enabled by default (can be disabled):**
- **Ginkgo** 2.0.0 (tag `6a3abf8c`) — sparse linear algebra backend; provides CG, BiCGStab, GMRES etc.; Kokkos extension used for executor bridging (`include/NeoN/linearAlgebra/ginkgo.hpp`)
- **nlohmann_json** 3.11.3 — JSON config for Ginkgo solver configuration (`ginkgo/extensions/config/json_config.hpp`)
- **Umpire** (tag `18a808d1`) — alternative memory manager/allocator (experimental); forked at `greole/umpire`
- **MPI** 3.1 — distributed communication; found via CMake `find_package(MPI)` (`include/NeoN/core/mpi/`)

**Optional / off by default:**
- **PETSc** — alternative linear algebra backend (`include/NeoN/linearAlgebra/petsc.hpp`); requires system install via pkg-config
- **ADIOS2** 2.10.2 — parallel I/O (with Kokkos patch); not wired into headers yet
- **SUNDIALS** 7.5.0 — ODE/time integration (ARKODE only); experimental (`timeIntegration/rungeKutta.cpp`)
- **spdlog** 1.16.0 — structured logging (replaces fmt-based logger when enabled)
- **nanobind** 2.9.2 — Python bindings; only used when `NeoN_BUILD_PYTHON_BINDINGS=ON`
- **Catch2** 3.4.0 — test framework; only fetched when `NeoN_BUILD_TESTS=ON` or `NeoN_BUILD_BENCHMARKS=ON`

## Supported Backends

NeoN maps three executor types to Kokkos execution spaces (`include/NeoN/core/executor/executor.hpp`):

| NeoN Executor | Kokkos Backend | Enable Flag |
|---------------|---------------|-------------|
| `SerialExecutor` | `Kokkos::Serial` | Always available |
| `CPUExecutor` | `Kokkos::OpenMP` or `Kokkos::Threads` | `NeoN_WITH_OMP=ON` or `NeoN_WITH_THREADS=ON` (default: Threads) |
| `GPUExecutor` | `Kokkos::DefaultExecutionSpace` | Auto-detected: CUDA → HIP → SYCL |

Backend auto-detection logic is in `cmake/AutoEnableDevice.cmake`:
- CUDA: `check_language(CUDA)` — sets `Kokkos_ENABLE_CUDA=ON` if `CMAKE_CUDA_COMPILER` found
- HIP: `check_language(HIP)` — sets `Kokkos_ENABLE_HIP=ON` if `CMAKE_HIP_COMPILER` found
- SYCL: supported via `Kokkos_ENABLE_SYCL` (passed through to Umpire), no auto-detect in CMake

`createDefaultExecutor()` (`executor.hpp`) picks the best available backend at runtime: CUDA > HIP > SYCL > OpenMP > Threads > Serial.

**Compile definitions** set on `NeoN_public_api`:
- `NEON_ENABLE_CUDA`, `NEON_ENABLE_HIP`, `NEON_ENABLE_OPENMP`, `NEON_ENABLE_THREADS`
- `NF_WITH_GINKGO`, `NF_WITH_UMPIRE`, `NF_WITH_PETSC`, `NF_WITH_MPI_SUPPORT`, `NF_WITH_SPDLOG`

## Numeric Precision Options

Controlled via CMake options, set as compile definitions on `NeoN_public_api`:
- `NeoN_DEFINE_DP_SCALAR` (ON by default) → `NeoN_DP_SCALAR=1` — double-precision scalar
- `NeoN_DEFINE_DP_LABEL` (OFF) → `NeoN_DP_LABEL=1` — 64-bit integer labels
- `NeoN_DEFINE_US_IDX` (OFF) → `NeoN_US_IDX=1` — unsigned indices

## Development Tools

**Pre-commit hooks** (`.pre-commit-config.yaml`) — enabled via `NeoN_DEVEL_TOOLS=ON` (included in `develop` preset):
- `clang-format` v17.0.6 — C++ formatting
- `cmake-format` + `cmake-lint` v0.6.13 — CMake formatting
- `reuse` v4.0.3 — SPDX license header compliance
- `typos` v1.23.1 — spell checking

**Static analyzers** (optional CMake flags):
- `NeoN_ENABLE_CLANG_TIDY=ON` — clang-tidy
- `NeoN_ENABLE_CPP_CHECK=ON` — cppcheck
- `NeoN_ENABLE_IWYU=ON` — include-what-you-use

**Sanitizers** (optional CMake flags):
- Address, Leak, UndefinedBehavior, Thread, Memory sanitizers — each a separate `NeoN_ENABLE_SANITIZE_*` option

**Python tooling** (`pyproject.toml` `[dev]` extras):
- `pytest` ≥6.0
- `ruff` ≥0.14.0 — Python linting
- `black` — Python formatting (line-length 100)
- `mypy` — type checking

## CI

**GitHub Actions** (`.github/workflows/`):
- `build_on_ubuntu.yaml` — matrix: clang-19 / gcc-10 × develop / profiling / production
- `build_on_macos.yaml` — macOS builds
- `build_on_windows.yaml` — Windows builds
- `build_on_aws.yaml` — AWS GPU builds
- `build_with_san.yaml` — sanitizer builds
- `static_checks.yaml` — pre-commit / static analysis
- `build_doc.yaml` — documentation

**LRZ GitLab CI** (`ci/lrz-gitlab/`) — HPC cluster builds and benchmarks

---

*Stack analysis: 2026-05-10*
