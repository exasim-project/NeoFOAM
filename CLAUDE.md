# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NeoFOAM is an adapter layer that connects [OpenFOAM](https://openfoam.org/) data structures to the [NeoN](https://github.com/exasim-project/NeoN) GPU/CPU compute backend. The goal is to replace standard OpenFOAM solvers with GPU-accelerated NeoN variants, enabling portable HPC CFD on CPU and GPU hardware without changing the OpenFOAM user workflow.

**NeoN** (at `src/NeoN/`) handles portable computation — executors, fields, DSL, linear algebra. **NeoFOAM** wraps NeoN to speak OpenFOAM's language: meshes, dictionaries, boundary conditions, and solver algorithms. See `src/NeoN/CLAUDE.md` for NeoN internals.

## Repository Structure

```
NeoFOAM/
├── src/NeoN/          # NeoN compute backend (git submodule)
├── src/
│   ├── algorithms/    # Pressure-velocity coupling (PISO/SIMPLE) using NeoN solvers
│   ├── auxiliary/     # OF↔NeoN data conversion, field comparison, writers
│   ├── compatibility/ # Maps OF fvSchemes/fvSolution dicts to NeoN configuration
│   ├── datastructures/# foamMesh/meshAdapter — bridges OF and NeoN mesh types
│   └── fvcc/          # Finite-volume cell-centred surface interpolation
├── include/NeoFOAM/   # Public headers (mirrors src/ structure)
├── test/              # Integration tests verifying OF and NeoFOAM give identical results
├── examples/          # Example solver applications
├── tutorials/         # OpenFOAM-style tutorial cases
├── benchmarks/        # Performance benchmarks
└── cmake/             # CMake helper modules
```

NeoN is a git submodule tracked in `.gitmodules`:
```bash
git submodule update --init --recursive   # initialise/update NeoN
```

## Build Prerequisites

OpenFOAM **must** be sourced before configuring or building:
```bash
source /path/to/OpenFOAM-2406/etc/bashrc   # or equivalent
# Requires FOAM_SRC env var and OpenFOAM API >= 2406
```

## Build Commands

```bash
# List available presets
cmake --list-presets

# Development build (Debug, tests + examples enabled, Kokkos debug bounds check)
cmake --preset develop
cmake --build --preset develop -- -j4

# Production build (Release, examples enabled)
cmake --preset production
cmake --build --preset production -- -j4

# Profiling build (RelWithDebInfo, benchmarks + examples enabled)
cmake --preset profiling
cmake --build --preset profiling -- -j4
```

Build directories are placed at `build/<presetName>/`. All presets use Ninja.

### CMake Options

| Option | Default | Purpose |
|---|---|---|
| `NEOFOAM_BUILD_TESTS` | OFF | Enable unit/integration tests |
| `NEOFOAM_BUILD_EXAMPLES` | ON | Build example applications |
| `NEOFOAM_BUILD_BENCHMARKS` | OFF | Enable benchmarks |
| `NEOFOAM_ENABLE_KOKKOS_TOOLS` | OFF | Enable Kokkos profiling tools |

### NeoN Resolution Priority

CMake resolves NeoN in this order:
1. `NEOFOAM_NEON_DIR` (explicit path override)
2. Local `src/NeoN/CMakeLists.txt` (submodule, preferred)
3. CPM.cmake auto-fetch (fallback if submodule not initialised)

## Running Tests

```bash
# Run all tests (from repo root, develop preset must be built)
ctest --preset develop

# Or from the build directory
cd build/develop
ctest

# Run a specific test binary
./build/develop/bin/tests/neofoam_test_<name>
```

Tests in `test/` verify that NeoFOAM and standard OpenFOAM produce identical results for the same cases.

## Architecture: The Adapter Layer

NeoFOAM's src modules are a one-way bridge: OpenFOAM data in → NeoN operations → results back to OpenFOAM.

- **`datastructures/`** — `foamMesh` and `meshAdapter` convert OpenFOAM `fvMesh` into NeoN `UnstructuredMesh`. This is the foundational mapping all other modules depend on.
- **`auxiliary/`** — `convert.cpp` copies field data between OF `volScalarField`/`volVectorField` and NeoN `VolumeField`. `comparison.cpp` diffs OF and NeoN fields for test assertions. `foamDictionary.cpp` wraps OF `IOdictionary` into NeoN `Dictionary`.
- **`compatibility/`** — `fvSchemes.cpp` and `fvSolution.cpp` read OpenFOAM's standard scheme/solution dictionaries and produce NeoN-compatible configuration, so existing OF case setups work without modification.
- **`algorithms/`** — `pressureVelocityCoupling.cpp` implements PISO/SIMPLE loops using NeoN's DSL and linear solvers instead of OF's built-in solvers.
- **`fvcc/surfaceInterpolation/`** — Surface interpolation operators (e.g. linear, upwind) implemented over NeoN `SurfaceField`.

## Key Patterns

- Always check `FOAM_SRC` is set before attempting a build or test run — the build will hard-fail otherwise.
- When adding a new OF↔NeoN conversion, follow the existing pattern in `auxiliary/convert.cpp` — one function per field type, no ownership transfer.
- Scheme names registered in `compatibility/` must match OpenFOAM's canonical spelling so that existing tutorial `fvSchemes` files work unchanged.
- NeoN executor is chosen at runtime; NeoFOAM code must not hard-code `SerialExecutor` — always thread the executor through from the call site.

<!-- GSD:project-start source:PROJECT.md -->
## Project

**NeoFOAM Distributed — Parallel neoIcoFoam**

NeoFOAM is an adapter layer that connects OpenFOAM data structures to the NeoN GPU/CPU compute backend,
replacing standard OpenFOAM solvers with NeoN-accelerated variants. This milestone targets making
`neoIcoFoam` run correctly in parallel (MPI), validated on the cylinder3D case with 2–4 MPI ranks,
producing field results within L∞ tolerance of the single-process case — without changing the
OpenFOAM user workflow.

**Core Value:** neoIcoFoam on cylinder3D with 4 MPI ranks produces field results within L∞ tolerance of the
serial run.

### Constraints

- **Compatibility**: OpenFOAM ≥ 2406 required; `FOAM_SRC` must be set before build
- **Runtime**: NeoN executor must never be hard-coded; always thread through from call site
- **Submodule**: NeoN changes land on `fix/testsRebase` branch in `src/NeoN/`; coordinate commits
- **MPI**: Target 2–4 ranks for development; HPC validation on cluster (ranks TBD)
- **Build preset**: Use `develop` preset for all development (Debug + bounds checks)
- **Solver target**: Ginkgo distributed solver — PETSc is out of scope for this milestone
- **Mesh**: cylinder3D (4-rank hierarchical decomp) is the validation case; cylinder2D is regression
<!-- GSD:project-end -->

<!-- GSD:stack-start source:codebase/STACK.md -->
## Technology Stack

## Languages
- C++20 — all NeoFOAM and NeoN source (`src/`, `include/`, `test/`, `benchmarks/`, `examples/`)
- C — required by CMake `LANGUAGES C CXX` for interop with MPI and Kokkos C internals
- Python 3.9+ — post-processing tooling (`pyproject.toml`), benchmark orchestration (`benchmarks/benchmarkSuite/`), tutorials validation
- CMake — build configuration language throughout `CMakeLists.txt` and `cmake/` modules
## Runtime
- Linux (POSIX); macOS partially supported (rpath workaround); Windows not supported with MPI
- C++20 (`set(CMAKE_CXX_STANDARD 20)` in both `CMakeLists.txt` and `src/NeoN/CMakeLists.txt`)
- Required: OpenFOAM >= 2406 (enforced by `FOAM_API >= 2406` check in `CMakeLists.txt`)
- CI uses: OpenFOAM 2406 (`openfoam2406` and `OpenFOAM-v2412`)
- Must be sourced before build: `source /path/to/OpenFOAM-2406/etc/bashrc`
- Env var `FOAM_SRC` must be set — build hard-fails if missing
## Package Manager
- CPM.cmake (`cmake/CPM.cmake` and `src/NeoN/cmake/CPM.cmake`) — fetches third-party libs at configure time
- Prefers system-installed packages (`CPM_USE_LOCAL_PACKAGES=ON` by default in NeoN)
- No lockfile; versions pinned in `src/NeoN/cmake/Versions.cmake`
- pip / conda (`environment.yml` uses conda-forge + bioconda + pip extras)
- `pyproject.toml` defines the `neofoam` Python package (setuptools backend)
## Build System
- `develop` — Debug, tests + examples + benchmarks off, Kokkos debug bounds check, IWYU/tools enabled
- `production` — Release, examples only
- `profiling` — RelWithDebInfo, benchmarks + examples, frame-pointer preserved
- `build/<presetName>/bin/` — executables
- `build/<presetName>/lib/` — shared libraries
## Frameworks
- Portable CPU/GPU kernel execution via Kokkos executor abstraction
- DSL for composing PDE equations (`include/NeoN/dsl/`)
- Linear algebra via Ginkgo backend
- Field types: `VolumeField<T>`, `SurfaceField<T>`, `Array<T>`, `Vector<T>`
- Provides mesh, field, dictionary, and I/O infrastructure
- Linked as shared libraries imported from `$FOAM_SRC` / `$FOAM_LIBBIN`
- Key imported targets: `OpenFOAM::OpenFOAM`, `OpenFOAM::finiteVolume`, `OpenFOAM::meshTools`, `OpenFOAM::Pstream`
- Catch2 v3.4.0 — unit and integration tests, also benchmarks
- CTest — test runner, MPI tests invoked via `mpiexec`
## Key Dependencies
- Kokkos 5.0.2 — platform portability layer (CPU serial, CPU threads, CUDA, HIP, SYCL backends); fetched via FetchContent if not system-installed (`src/NeoN/cmake/CxxThirdParty.cmake`)
- Ginkgo 2.0.0 (tag `6a3abf8`) — sparse linear algebra backend; fetched via CPM if not system-installed
- nlohmann_json 3.11.3 — runtime configuration via `Dictionary`; fetched via CPM if not system-installed
- fmt 12.1.0 — string formatting; always fetched via CPM
- cpptrace 0.7.3 — C++ stack traces; always fetched via CPM
- MPI 3.1+ — parallel communication; `find_package(MPI 3.1 REQUIRED)` in both `cmake/CxxThirdParty.cmake` and `src/NeoN/cmake/CxxThirdParty.cmake`; linked as `MPI::MPI_CXX`
- Umpire (tag `18a808d1`) — advanced memory management (default ON in NeoN, experimental)
- ADIOS2 2.10.2 — parallel I/O (default OFF)
- SUNDIALS 7.5.0 — ODE/DAE solvers (default OFF)
- PETSc — alternative linear algebra backend (default OFF)
- spdlog 1.16.0 — structured logging (default OFF, falls back to fmt-based logging)
- Kokkos Tools (tag `33693813`) — profiling and tracing; fetched as ExternalProject when `NEOFOAM_ENABLE_KOKKOS_TOOLS=ON`
- nanobind 2.9.2 — Python bindings for NeoN (default OFF)
## Configuration
- `FOAM_SRC` — path to OpenFOAM source (required at configure time)
- `FOAM_API` — OpenFOAM version integer (must be >= 2406)
- `FOAM_LIBBIN` — OpenFOAM binary library path
- `WM_LABEL_SIZE`, `WM_PRECISION_OPTION` — OpenFOAM precision/label compile definitions
- `CMAKE_CUDA_ARCHITECTURES` — set to `native` in CMakePresets.json base
- `cmake/OpenFOAM.cmake` — imports OF shared libraries via custom `importOFLibrary()` function
- `cmake/CxxThirdParty.cmake` — fetches Catch2 and MPI for NeoFOAM-level tests
- `src/NeoN/cmake/Versions.cmake` — all third-party version pins in one file
- `src/NeoN/cmake/CxxThirdParty.cmake` — fetches Kokkos, Ginkgo, nlohmann_json, fmt, cpptrace, Umpire, etc.
- `CMakePresets.json` — developer-facing preset definitions
## Platform Requirements
- Linux with OpenFOAM 2406+ sourced
- C++20-capable compiler: GCC or Clang (both tested in CI on ubuntu-24.04)
- Ninja build system
- MPI implementation (OpenMPI used in CI: `libopenmpi-dev`, `openmpi-bin`)
- Python 3.9+ (for benchmarks and tutorials)
- Same Linux + OpenFOAM environment
- GPU support: NVIDIA (CUDA 12.x, arch 89+), AMD (ROCm 6.4+, arch gfx90a), Intel (oneAPI 2025.3, PVC)
- CI GPU images: `chihtaw/cuda-openfoam-ginkgo:12.8.1-v2412-arch_89`, `chihtaw/rocm-openfoam-ginkgo:6.4.1-v2412-arch_gfx90a`, `dheerajraghunathan/oneapi-openfoam-ginkgo:2025.3-v2412-arch_pvc`
## Development Tools
- clang-format v17.0.6 — C++ formatting; config in `.clang-format` (LLVM-based with OpenFOAM style, 4-space indent, 100 column limit)
- cmake-format v0.6.13 — CMake file formatting
- clang-tidy — static analysis; config in `.clang-tidy`; enabled via `NeoFOAM_ENABLE_CLANG_TIDY` CMake option
- cppcheck — static analysis; enabled via `NeoFOAM_ENABLE_CPP_CHECK` CMake option
- IWYU (include-what-you-use) — include hygiene; run in `static_checks.yaml` CI job; enabled via `NeoN_ENABLE_IWYU`
- typos v1.23.1 — spell checking
- `.pre-commit-config.yaml` hooks: check-yaml, end-of-file-fixer, trailing-whitespace, pretty-format-json, clang-format, cmake-format, cmake-lint, REUSE license check, typos
- REUSE 6.2.0 compliance; `REUSE.toml` and `LICENSES/` directory
- GitHub Actions (`.github/workflows/`): build+test on ubuntu-24.04 with GCC and Clang, static checks (IWYU, clang-tidy), changelog check, doc build
- GitLab CI (`.gitlab-ci.yml`): GPU testing on NVIDIA/AMD/Intel runners via triggered pipelines from GitHub; benchmarking pipeline
- LRZ GitLab CI (`ci/lrz-gitlab/`) for HPC system testing
<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->
## Conventions

## Naming Patterns
- Header files: `lowerCamelCase.hpp` under `include/NeoFOAM/<module>/` (e.g., `meshAdapter.hpp`, `pressureVelocityCoupling.hpp`)
- Source files: `lowerCamelCase.cpp` under `src/<module>/` (e.g., `foamMesh.cpp`, `setup.cpp`)
- Test files: `test_<snake_case>.cpp` under `test/` (e.g., `test_unstructuredMesh.cpp`, `test_distributedPressureVelocityCoupling.cpp`)
- OpenFOAM case setup dirs: `setup_<camelCase>/` under `test/` (e.g., `setup_pressureVelocityCoupling/`, `setup_advection/`)
- `CamelCase` for all class, struct, enum, and union names (enforced by `.clang-tidy`)
- Example: `MeshAdapter`, `EqualsRangeMatcher`, `ApproxScalar`, `ApproxVector`
- `camelBack` (lower first letter) for all functions and methods (enforced by `.clang-tidy`)
- Example: `computeOffset()`, `createMesh()`, `readOpenFOAMMesh()`, `flatBCField()`
- Exception: OpenFOAM-conventional single-letter physics variables (`U`, `T`, `A`, `HbyA`) are exempt from the camelBack rule (listed in `.clang-tidy` `VariableIgnoredRegexp` and `ParameterIgnoredRegexp`)
- `camelBack` (lower first letter), enforced by `.clang-tidy`
- Example: `execName`, `nfMesh`, `runTime`, `fieldName`
- Physics exception: `U`, `T`, `A`, `HbyA` are allowed as-is
- Project namespace: `NeoFOAM` (all NeoFOAM code lives here)
- NeoN library namespace aliases are declared at the top of each file or header:
- Alias declarations belong at file scope, before the `namespace NeoFOAM {` block in headers, or at the top of `.cpp` files
## Code Style
- `IndentWidth: 4` — 4 spaces per indentation level, no tabs (`UseTab: Never`)
- `ColumnLimit: 100` — maximum line length 100 characters
- `PointerAlignment: Left`, `ReferenceAlignment: Left` — `T* ptr`, `T& ref`
- `AlignAfterOpenBracket: BlockIndent` — arguments wrap to next line, all on separate lines
- `BinPackArguments: false`, `BinPackParameters: false` — no bin-packing; each arg on its own line when wrapping
- `BreakBeforeBraces: Custom` (Allman-style with all braces on their own line: after `class`, `function`, `if`, `for`, `namespace`, etc.)
- `AlwaysBreakTemplateDeclarations: Yes` — template declaration always on its own line
- `MaxEmptyLinesToKeep: 2` — up to two blank lines between sections/functions
- `SortIncludes: false` — include order is NOT sorted automatically (critical for OpenFOAM header ordering)
- `SpaceBeforeParens: ControlStatementsExceptForEachMacros` — space in `if ()`, `for ()`, but NOT in `forAll()`
- Only `readability-identifier-naming` check is enabled
- Warnings are NOT treated as errors (`WarningsAsErrors: ''`)
- Filter applies only to `include/NeoFOAM/` headers (`HeaderFilterRegex`)
## License Headers
## Include Organization
- OpenFOAM headers: NO angle-bracket prefix sorting — include order must remain exactly as written, because OpenFOAM uses macro-based includes (`createTime.H`, `setRootCase.H`) and ordering matters
- `SortIncludes: false` is explicitly set to prevent clang-format from reordering
- Typical order in implementation files: project header first, then NeoN headers, then OpenFOAM headers
## Key Patterns
- `NeoN::Executor` is always passed in from the call site — never hard-coded
- The one acceptable use of `SerialExecutor` directly is in `meshAdapter.cpp` for the `fullMeshOnGPU=false` branch (mesh data that does not need to live on GPU)
- In tests, the `allAvailableExecutor()` generator (`test/catch2/executorGenerator.hpp`) returns all compiled-in executors: always includes `SerialExecutor`, adds `CPUExecutor` if OpenMP/threads enabled, adds `GPUExecutor` if CUDA/HIP/SYCL enabled
- MPI distributed tests currently hard-code `CPUExecutor` (pending full distributed GPU support):
- Runtime executor selection from dictionaries: `NeoFOAM::createExecutor(dict)` in `src/auxiliary/setup.cpp`
- All type conversions go through overloaded `NeoFOAM::convert()` functions in `src/auxiliary/convert.cpp`
- One overload per type pair — no ownership transfer
- Supported: `Foam::vector` ↔ `NeoN::Vec3`, `Foam::scalar` ↔ `NeoN::scalar`, `Foam::label` ↔ `NeoN::label`, `Foam::word` → `std::string`, `Foam::ITstream` → `NeoN::TokenList`, `Foam::dictionary` → `NeoN::Dictionary`
- When adding a new conversion, add both directions in `include/NeoFOAM/auxiliary/convert.hpp` (declaration) and `src/auxiliary/convert.cpp` (definition)
- Both `meshAdapter.cpp` and `test/common.hpp` iterate boundary patches in two passes: first all non-processor patches, then all processor patches
- This matches the NeoN `BoundaryMesh` layout where processor patches occupy the trailing tail of the boundary data arrays
- New code iterating over OF boundary patches must follow this two-pass pattern
- `NeoFOAM::RunTime` (declared in `include/NeoFOAM/datastructures/runTime.hpp`) bundles `exec`, `mesh`, `nfMesh`, `db`, `fvSchemesDict`, `fvSolutionDict`, `mpiEnvironment`, and time-step values
- Created via `NeoFOAM::createAdapterRunTime(runTime, exec)` in `src/auxiliary/setup.cpp`
- All algorithmic functions receive `RunTime&` rather than individual components
- `/** @brief ... */` for Doxygen-style doc comments on public API functions
- `/* ... */` for multi-line explanatory comments
- `// ...` for inline comments
- OpenFOAM-style section delimiters (`// * * * * *`) appear only where OpenFOAM integration code requires them (e.g., `defineTypeNameAndDebug`)
## What to Avoid
- **Hard-coding `SerialExecutor`** in algorithmic or field code — always thread the executor through from the call site
- **Reordering OpenFOAM includes** — clang-format is configured with `SortIncludes: false` specifically to prevent this; reordering can break OpenFOAM's macro-based include system
- **Ownership transfer in conversions** — `convert()` functions copy data; they do not transfer ownership or return references into the source object
- **`namespace NeoFOAM` closing comment inconsistency** — some files use `} // namespace NeoFOAM`, others `}; // namespace NeoFOAM` or `} // namespace NeoFoam` (note lowercase `f`). Use `} // namespace NeoFOAM` (no semicolon, uppercase F) for consistency
- **Single-executor test assumptions** — tests must use `allAvailableExecutor()` so they run on all compiled backends; do not `GENERATE` only `SerialExecutor` in serial tests (MPI distributed tests are the exception, currently pinned to `CPUExecutor`)
- **Storing scheme names that differ from OpenFOAM canonical spelling** — `compatibility/fvSchemes.cpp` maps OF scheme names verbatim; new scheme registrations must match OpenFOAM's spelling so existing `system/fvSchemes` files work unchanged
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->
## Architecture

## System Overview
```text
```
## Core Abstractions
| Class / Type | Role | File |
|---|---|---|
| `NeoFOAM::MeshAdapter` | Inherits `Foam::fvMesh`; embeds `NeoN::UnstructuredMesh nfMesh_`; single source of truth for both mesh views | `include/NeoFOAM/datastructures/meshAdapter.hpp` |
| `NeoFOAM::RunTime` | POD struct bundling `MeshAdapter`, `NeoN::Database`, `NeoN::Executor`, time values, and pre-converted dictionaries (`fvSchemesDict`, `fvSolutionDict`) | `include/NeoFOAM/datastructures/runTime.hpp` |
| `NeoFOAM::PDESolver<ValueType>` | Wraps a `NeoN::dsl::Expression` + its assembled `NeoN::la::LinearSystem`; provides `assemble()` and `solve()` | `include/NeoFOAM/datastructures/pdeSolver.hpp` |
| `NeoFOAM::TypeMap<FoamType>` | Compile-time trait mapping OF field types to NeoN container/primitive types (e.g. `volScalarField` → `fvcc::VolumeField<NeoN::scalar>`) | `include/NeoFOAM/auxiliary/typeConversion.hpp` |
| `readOpenFOAMMesh()` | Free function: converts `Foam::fvMesh` → `NeoN::UnstructuredMesh`. Processor patches appended last in boundary arrays | `src/datastructures/meshAdapter.cpp` |
| `constructFrom<FoamFieldType>()` | Template free function: creates a NeoN VolumeField or SurfaceField from an OF geometric field, copying internal data and boundary conditions | `include/NeoFOAM/auxiliary/readers.hpp` |
| `constructAndRegister()` | Registers a NeoN field derived from an OF field into `fvcc::VectorCollection`; handles old-time storage for time-dependent schemes | `include/NeoFOAM/auxiliary/readers.hpp` |
| `CreateFromFoamField<FieldType>` | Functor that produces a `fvcc::VectorDocument` for database registration | `include/NeoFOAM/auxiliary/readers.hpp` |
| `write(fvcc::VolumeField<T>, Foam::fvMesh)` | Copies NeoN field back to an OF field object and calls `field.write()` for OpenFOAM file I/O | `include/NeoFOAM/auxiliary/writers.hpp`, `src/auxiliary/writers.cpp` |
## Data Flow
### Initialization
### Time Loop (neoIcoFoam PISO)
### Write-back
## Design Patterns
## Module Boundaries
## Architectural Constraints
- **Executor threading model:** NeoN uses Kokkos under the hood. The executor (Serial/CPU/GPU) is chosen once at startup from `controlDict["executor"]` and threaded through `RunTime`. No module may hard-code `SerialExecutor`; always read executor from `RunTime::exec` or a passed argument.
- **Processor-boundary ordering:** Both mesh construction (`computeOffset`, `flatBCField`) and field readers (`readVolBoundaryConditions`, `readSurfaceBoundaryConditions`, `constructFrom` for surface fields) enforce the invariant: non-processor patches occupy `[0, nNonProcBnd)`, processor patches occupy `[nNonProcBnd, nBnd)` in boundary arrays. Any kernel iterating boundary ranges must respect this ordering. Violating it causes MPI sync to send stale zeros.
- **Global state:** `fvcc::VectorCollection::instance(db, "VectorCollection")` is a singleton keyed on the NeoN `Database`. One `Database` per `RunTime`; do not share databases across solver instances.
- **Circular imports:** None observed. Dependency direction is: `algorithms` → `datastructures`, `auxiliary` → `datastructures`, `compatibility` → (NeoN only), `fvcc` → (OF RTST only).
- **Reference cell pinning:** `PDESolver::SetReference` executes only on MPI rank 0. The global→(rank, localIdx) mapping is not implemented; `pRefCell` is always treated as local index on rank 0 (see comment in `pdeSolver.hpp`).
- **C++ standard:** C++20 required (see `CMakeLists.txt`).
- **OpenFOAM API:** Requires `FOAM_API >= 2406`; build hard-fails if `FOAM_SRC` is not set.
## Anti-Patterns
### Calling `SerialExecutor` directly
### Using OF full-size face arrays for boundary kernels
## Error Handling
## Cross-Cutting Concerns
<!-- GSD:architecture-end -->

<!-- GSD:skills-start source:skills/ -->
## Project Skills

No project skills found. Add skills to any of: `.claude/skills/`, `.agents/skills/`, `.cursor/skills/`, `.github/skills/`, or `.codex/skills/` with a `SKILL.md` index file.
<!-- GSD:skills-end -->

<!-- GSD:workflow-start source:GSD defaults -->
## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:
- `/gsd-quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd-debug` for investigation and bug fixing
- `/gsd-execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->

<!-- GSD:profile-start -->
## Developer Profile

> Profile not yet configured. Run `/gsd-profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
