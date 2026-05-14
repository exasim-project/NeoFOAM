# AGENTS.md

This file provides guidance when working with code in this repository.

## What is NeoFOAM

NeoFOAM bridges **OpenFOAM** (the CFD framework) and **NeoN** (a GPU-portable computational backend built on Kokkos). It provides bidirectional data-structure converters, finite-volume CFD algorithms, and example solvers that run on CPUs and accelerator devices with the same source code.

## Prerequisites

OpenFOAM 2406+ must be sourced in the shell before any CMake commands:
```sh
source /path/to/OpenFOAM-2406/etc/bashrc
```

## Build Commands

```sh
cmake --list-presets                # show available presets
cmake --preset develop              # configure (debug + tests + examples)
cmake --build --preset develop      # build
cmake --preset production           # configure (release, no tests)
cmake --build --preset production   # build
```

Presets write build artifacts to `build/<presetName>/`. Binaries go to `build/<preset>/bin/`, libraries to `build/<preset>/lib/`.

NeoN is located automatically: if `src/NeoN/CMakeLists.txt` exists it is used directly; otherwise it is fetched via CPM. Override with `-DNEOFOAM_NEON_DIR=<path>`.

## Running Tests

Tests require the `develop` preset (sets `NEOFOAM_BUILD_TESTS=ON`).

```sh
# Run all tests
ctest --preset develop

# Run a single test binary directly (must be run from its setup directory)
cd test/setup_pressureVelocityCoupling
../../build/develop/bin/tests/neofoam_pressureVelocityCoupling

# Pass Catch2 options — separate them from OpenFOAM args with "---"
../../build/develop/bin/tests/neofoam_operators "[SomeTag]" --- -case .
```

Each test binary is built from `test/test_<name>.cpp` as `neofoam_<name>` and **must** run with its matching `test/setup_<name>/` directory as the working directory (CMakeLists sets this automatically for `ctest`).

## Architecture

### Namespace conventions used throughout the codebase

```cpp
namespace NeoFOAM         // this library
namespace dsl = NeoN::dsl // PDE expression DSL from NeoN
namespace fvcc = NeoN::finiteVolume::cellCentred  // NeoN cell-centred FV types
namespace nf = NeoFOAM    // short alias used in examples/tests
```

### Code conventions

- Use curly braces for for loops even for single line loop bodys

### `include/NeoFOAM/` + `src/` — the library

**`datastructures/`**
- `MeshAdapter` — extends `Foam::fvMesh` and owns a `NeoN::UnstructuredMesh`. The primary entry point for mesh access is `rt.mesh` / `rt.nfMesh`.
- `RunTime` — plain struct aggregating everything a solver needs: `NeoN::Database db`, `MeshAdapter& mesh`, `NeoN::UnstructuredMesh& nfMesh`, `NeoN::Executor exec`, time scalars `t`/`dt`, and pre-converted `controlDict`/`fvSolutionDict`/`fvSchemesDict`. Created via `NeoFOAM::createAdapterRunTime(runTime[, exec])`.
- `PDESolver<ValueType>` — wraps a `NeoN::dsl::Expression` with an owned `LinearSystem`. Provides `assemble()`, `solve()`, and `solve(rhs)`. Used directly in place of `fvMatrix` in solver loops.

**`auxiliary/`**
- `typeConversion.hpp` — `TypeMap<FoamType>` trait that maps every supported OpenFOAM field type to its NeoN `container_type` and `mapped_type`.
- `convert.hpp` — free functions converting scalars, vectors, words, dictionaries, and token lists in both directions (`Foam → NeoN` and `NeoN → Foam`).
- `readers.hpp` — `constructFrom(exec, nfMesh, foamField)` builds any NeoN field from an OpenFOAM field (dispatches on `TypeMap`). `constructAndRegister(collection, rt, foamField)` also registers it in the `VectorCollection`. `CreateFromFoamField` functor registers fields in a `NeoN::Database`.
- `writers.hpp` — writes `NeoN::Vector` or `fvcc::VolumeField` back to disk using OpenFOAM file format.
- `comparison.hpp` — `operator==` overloads for cross-framework field comparison (used in tests).
- `setup.hpp` — `createAdapterRunTime`, `syncRunTimes`, `createExecutor`.

**`compatibility/`**
- `fvSchemes.cpp` — `mapFvSchemes(dict)` translates an OpenFOAM `fvSchemes` dictionary to the NeoN scheme naming convention. Call this once after `createAdapterRunTime`.
- `fvSolution.cpp` — `mapFvSolution(solverDict)` translates per-field solver sub-dicts (solver type strings, preconditioner names) to NeoN format.

**`algorithms/`**
- `pressureVelocityCoupling.hpp/cpp` — PISO/SIMPLE helper functions: `computeRAU`, `computeRAUandHByA`, `updateFaceVelocity`, `updateVelocity`, `constrainHbyA`, `flux`.

### `examples/neoIcoFoam/`

A complete PISO incompressible solver that demonstrates the canonical usage pattern:
1. `createAdapterRunTime` → maps fvSchemes/fvSolution
2. `constructAndRegister` all fields into `VectorCollection`
3. Build `PDESolver<Vec3>` for momentum with `dsl::imp::ddt + div + laplacian`
4. PISO loop: `UEqn.solve(rhs)` → `computeRAUandHByA` → build pressure equation → `pEqn.solve()` → `updateFaceVelocity` → `updateVelocity`

### `test/`

Tests should generally validate behavior across all enabled executors (CPU/GPU) and avoid assumptions about deterministic floating-point ordering.

Each test file starts an OpenFOAM `Time`/mesh in `test_main.cpp`, then Catch2 test cases use `GENERATE(allAvailableExecutor())` to run the same case on every compiled executor (Serial, CPUExecutor for OpenMP/threads, GPUExecutor for CUDA/HIP/SYCL). Setup case files live in `test/setup_<name>/`.

## `NeoFOAM.hpp` is generated

`include/NeoFOAM/NeoFOAM.hpp` does not exist in the source tree — it is generated at configure time from `NeoFOAM.hpp.in` and `#include`s every header in `include/NeoFOAM/`. Always include it as `#include "NeoFOAM/NeoFOAM.hpp"`.

## Git workflow

-  Always create a backup branch before performing a rebase

## GPU considerations
- Device kernels must use NEON_LAMBDA
- Avoid host-only allocations/access inside kernels
- Prefer parallelFor abstractions over raw loops
- Explicit synchronization may be required before host reads ( NeoN::fence(exec) )
- Use .copyToHost() when validating vectors in tests
