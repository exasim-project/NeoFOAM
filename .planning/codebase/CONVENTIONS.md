# Coding Conventions

**Analysis Date:** 2026-05-10

## Naming Patterns

**Files:**
- Header files: `lowerCamelCase.hpp` under `include/NeoFOAM/<module>/` (e.g., `meshAdapter.hpp`, `pressureVelocityCoupling.hpp`)
- Source files: `lowerCamelCase.cpp` under `src/<module>/` (e.g., `foamMesh.cpp`, `setup.cpp`)
- Test files: `test_<snake_case>.cpp` under `test/` (e.g., `test_unstructuredMesh.cpp`, `test_distributedPressureVelocityCoupling.cpp`)
- OpenFOAM case setup dirs: `setup_<camelCase>/` under `test/` (e.g., `setup_pressureVelocityCoupling/`, `setup_advection/`)

**Classes:**
- `CamelCase` for all class, struct, enum, and union names (enforced by `.clang-tidy`)
- Example: `MeshAdapter`, `EqualsRangeMatcher`, `ApproxScalar`, `ApproxVector`

**Functions and Methods:**
- `camelBack` (lower first letter) for all functions and methods (enforced by `.clang-tidy`)
- Example: `computeOffset()`, `createMesh()`, `readOpenFOAMMesh()`, `flatBCField()`
- Exception: OpenFOAM-conventional single-letter physics variables (`U`, `T`, `A`, `HbyA`) are exempt from the camelBack rule (listed in `.clang-tidy` `VariableIgnoredRegexp` and `ParameterIgnoredRegexp`)

**Variables and Parameters:**
- `camelBack` (lower first letter), enforced by `.clang-tidy`
- Example: `execName`, `nfMesh`, `runTime`, `fieldName`
- Physics exception: `U`, `T`, `A`, `HbyA` are allowed as-is

**Namespaces:**
- Project namespace: `NeoFOAM` (all NeoFOAM code lives here)
- NeoN library namespace aliases are declared at the top of each file or header:
  ```cpp
  namespace fvcc = NeoN::finiteVolume::cellCentred;
  namespace dsl  = NeoN::dsl;
  namespace la   = NeoN::la;
  namespace nf   = NeoFOAM;      // used in test files
  namespace fvc  = Foam::fvc;    // used in test/algorithm files
  namespace fvm  = Foam::fvm;    // used in test/algorithm files
  ```
- Alias declarations belong at file scope, before the `namespace NeoFOAM {` block in headers, or at the top of `.cpp` files

## Code Style

**Formatter:** clang-format, configured via `.clang-format` (based on LLVM / OpenFOAM style)

**Key settings:**
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

**Linter:** clang-tidy, configured via `.clang-tidy`
- Only `readability-identifier-naming` check is enabled
- Warnings are NOT treated as errors (`WarningsAsErrors: ''`)
- Filter applies only to `include/NeoFOAM/` headers (`HeaderFilterRegex`)

## License Headers

Every source and header file starts with a two-line SPDX block:
```cpp
// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
```
Some newer files use MIT license (primarily in `test/catch2/`). Match the license of the directory you are adding to.

## Include Organization

- OpenFOAM headers: NO angle-bracket prefix sorting — include order must remain exactly as written, because OpenFOAM uses macro-based includes (`createTime.H`, `setRootCase.H`) and ordering matters
- `SortIncludes: false` is explicitly set to prevent clang-format from reordering
- Typical order in implementation files: project header first, then NeoN headers, then OpenFOAM headers

## Key Patterns

**Executor threading:**
- `NeoN::Executor` is always passed in from the call site — never hard-coded
- The one acceptable use of `SerialExecutor` directly is in `meshAdapter.cpp` for the `fullMeshOnGPU=false` branch (mesh data that does not need to live on GPU)
- In tests, the `allAvailableExecutor()` generator (`test/catch2/executorGenerator.hpp`) returns all compiled-in executors: always includes `SerialExecutor`, adds `CPUExecutor` if OpenMP/threads enabled, adds `GPUExecutor` if CUDA/HIP/SYCL enabled
- MPI distributed tests currently hard-code `CPUExecutor` (pending full distributed GPU support):
  ```cpp
  auto [execName, exec] = GENERATE(
      std::pair<std::string, NeoN::Executor>{"CPUExecutor", NeoN::CPUExecutor{}}
  );
  ```
- Runtime executor selection from dictionaries: `NeoFOAM::createExecutor(dict)` in `src/auxiliary/setup.cpp`

**OF↔NeoN conversion:**
- All type conversions go through overloaded `NeoFOAM::convert()` functions in `src/auxiliary/convert.cpp`
- One overload per type pair — no ownership transfer
- Supported: `Foam::vector` ↔ `NeoN::Vec3`, `Foam::scalar` ↔ `NeoN::scalar`, `Foam::label` ↔ `NeoN::label`, `Foam::word` → `std::string`, `Foam::ITstream` → `NeoN::TokenList`, `Foam::dictionary` → `NeoN::Dictionary`
- When adding a new conversion, add both directions in `include/NeoFOAM/auxiliary/convert.hpp` (declaration) and `src/auxiliary/convert.cpp` (definition)

**Boundary patch ordering (non-processor first, processor last):**
- Both `meshAdapter.cpp` and `test/common.hpp` iterate boundary patches in two passes: first all non-processor patches, then all processor patches
- This matches the NeoN `BoundaryMesh` layout where processor patches occupy the trailing tail of the boundary data arrays
- New code iterating over OF boundary patches must follow this two-pass pattern

**RunTime aggregate:**
- `NeoFOAM::RunTime` (declared in `include/NeoFOAM/datastructures/runTime.hpp`) bundles `exec`, `mesh`, `nfMesh`, `db`, `fvSchemesDict`, `fvSolutionDict`, `mpiEnvironment`, and time-step values
- Created via `NeoFOAM::createAdapterRunTime(runTime, exec)` in `src/auxiliary/setup.cpp`
- All algorithmic functions receive `RunTime&` rather than individual components

**Comment style:**
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
