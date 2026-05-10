<!-- refreshed: 2026-05-10 -->
# NeoN Directory Structure

**Analysis Date:** 2026-05-10

## Top-Level Layout

```
NeoN/
├── include/NeoN/        # All public API headers (mirrors src/ structure)
│   ├── core/            # Executors, Vector, primitives, mpi, logging, dictionary
│   ├── fields/          # Field<T>, BoundaryData<T> — generic boundary-aware container
│   ├── mesh/unstructured/  # UnstructuredMesh, BoundaryMesh
│   ├── distributed/     # CommunicationPattern
│   ├── dsl/             # Expression, SpatialOperator, TemporalOperator, solve()
│   ├── linearAlgebra/   # LinearSystem, SparsityPattern, FaceToMatrixAddress, Ginkgo, PETSc
│   ├── finiteVolume/    # FV cell-centred fields, BCs, operators, interpolation, stencil
│   ├── timeIntegration/ # TimeIntegration, Runge-Kutta, Sundials wrappers
│   └── helpers/         # Miscellaneous small utilities
├── src/                 # Implementation files (structure mirrors include/)
│   ├── core/            # Vector, Array, mpi, database
│   ├── executor/        # Executor concrete implementations
│   ├── dsl/             # Expression/operator source
│   ├── linearAlgebra/   # SparsityPattern, FaceToMatrixAddress, Ginkgo, PETSc, matrix
│   ├── finiteVolume/    # Operators, BCs, interpolation, fields
│   ├── mesh/unstructured/  # Mesh constructors, factory functions
│   ├── timeIntegration/ # Integrator implementations
│   └── neon/            # Top-level NeoN.hpp aggregation sources
├── test/                # Catch2 unit/integration tests
├── benchmarks/          # Performance benchmarks
├── scripts/             # Development/CI scripts
├── cmake/               # CMake helper modules, CPM.cmake
├── ci/                  # CI configuration
├── doc/                 # Doxygen config
├── assets/              # Logo/images
├── CMakeLists.txt       # Root build definition
└── CMakePresets.json    # develop / production / profiling presets
```

## Key Modules

### `include/NeoN/core/executor/`

**Purpose:** Hardware backend abstraction — the central portability mechanism.

| Header | Class | Role |
|--------|-------|------|
| `executor.hpp` | `Executor` (`std::variant`) | Type alias, `createDefaultExecutor()`, `fence()`, `executorName()` |
| `serialExecutor.hpp` | `SerialExecutor` | Kokkos::Serial + HostSpace |
| `CPUExecutor.hpp` | `CPUExecutor` | Kokkos::OpenMP or Kokkos::Threads |
| `GPUExecutor.hpp` | `GPUExecutor` | Kokkos::Cuda / HIP / SYCL |

All three concrete executors expose `alloc<T>`, `realloc<T>`, `free`, `createKokkosView`, `name()`, `memorySpace()`.

### `include/NeoN/core/vector/`

**Purpose:** Executor-aware owning data container.

| Header | Class | Role |
|--------|-------|------|
| `vector.hpp` | `Vector<T>` | Managed allocation, `view()`, `copyToExecutor()` |
| `vectorTypeDefs.hpp` | `scalarVector`, `vectorVector`, `labelVector` | Type aliases for common field types |
| `vectorFreeFunctions.hpp` | Free functions | `fill`, `map`, arithmetic ops |

### `include/NeoN/core/`

| Header | Purpose |
|--------|---------|
| `array.hpp` | `Array<T>` — raw allocation without math (used by sparsity structures) |
| `view.hpp` | `View<T>` — non-owning span (`Kokkos::View` wrapper) for kernel access |
| `parallelAlgorithms.hpp` | `parallelFor(exec, range, NEON_LAMBDA, name)`, `NEON_LAMBDA` macro |
| `runtimeSelectionFactory.hpp` | `RuntimeSelectionFactory<Base, Params>`, `Register<Derived>` CRTP mixin |
| `dictionary.hpp` | `Dictionary` — wraps nlohmann_json for runtime configuration |
| `input.hpp` | `Input = std::variant<Dictionary, TokenList>` |
| `error.hpp` | `NF_ASSERT`, `NF_ERROR_EXIT` macros |
| `logging.hpp` | `Logging::SupportsLoggingMixin`, `BaseLogger` |
| `primitives/` | `scalar`, `label`, `localIdx`, `Vec3`, `Tensor`, `SymmTensor` |
| `mpi/environment.hpp` | `mpi::Environment` (MPI_Comm wrapper), `mpi::Init` RAII |
| `mpi/halfDuplexCommBuffer.hpp` | Non-blocking point-to-point halo exchange buffer |
| `mpi/fullDuplexCommBuffer.hpp` | Full-duplex halo exchange buffer |
| `mpi/operators.hpp` | `mpi::getType<T>()`, MPI reduce/gather wrappers |
| `database/database.hpp` | `Database` — named `Collection` registry |
| `database/fieldCollection.hpp` | `FieldCollection` — stores VolumeField by name and time |
| `database/oldTimeCollection.hpp` | Old-time field store for time integration |

### `include/NeoN/fields/`

**Purpose:** Generic boundary-aware field container (below the FV layer).

| Header | Class | Role |
|--------|-------|------|
| `field.hpp` | `Field<T>` | `internalVector_` + `boundaryData_` |
| `boundaryData.hpp` | `BoundaryData<T>` | `value_`, `refValue_`, `valueFraction_`, `refGrad_`, patch offsets |

### `include/NeoN/mesh/unstructured/`

**Purpose:** Mesh geometry and boundary topology.

| Header | Class | Role |
|--------|-------|------|
| `unstructuredMesh.hpp` | `UnstructuredMesh` | Full mesh geometry, `boundaryMesh()`, `stencilDB()`, `globalOffset()` |
| `boundaryMesh.hpp` | `BoundaryMesh` | Boundary-face geometry, patch offsets, processor patch count, `neighbourRank_` |
| `communicator.hpp` | `Communicator` | Per-mesh MPI communicator wrapper |
| `uniformMeshDataGenerator.hpp` | Factory helpers | Generate 1D/2D/3D structured test meshes |

Factory functions (free functions in `unstructuredMesh.hpp`):
- `create1DUniformMesh(exec, nCells, Lx)`
- `create2DUniformMesh(exec, nx, ny, Lx, Ly)`
- `create3DUniformMesh(exec, nx, ny, nz, Lx, Ly, Lz)`
- `create1DUniformMeshPart(exec, nCells)` — for distributed tests
- `computeCommunicationPattern(mesh)` — builds `CommunicationPattern`

### `include/NeoN/distributed/`

| Header | Struct | Role |
|--------|--------|------|
| `communicationPattern.hpp` | `CommunicationPattern` | `sendCounts`, `recvIdx`, `boundaryMapVector`, `mpi::Environment` |
| `partitioning.hpp` | Free functions | Mesh partitioning utilities |

### `include/NeoN/dsl/`

**Purpose:** PDE expression building and top-level solve.

| Header | Class/Function | Role |
|--------|----------------|------|
| `expression.hpp` | `Expression<T>` | Aggregates operators, drives assembly |
| `spatialOperator.hpp` | `SpatialOperator<T>` | Type-erased spatial operator wrapper |
| `temporalOperator.hpp` | `TemporalOperator<T>` | Type-erased temporal operator wrapper |
| `operator.hpp` | `Operator`, `OperatorMixin<Out,In>` | Base types, `Operator::Type::Implicit/Explicit` |
| `coeff.hpp` | `Coeff` | Scalar or vector coefficient scaling |
| `solver.hpp` | `dsl::solve(exp, sol, t, dt, ...)` | Top-level solve entry point |
| `explicit.hpp` | Free functions | Explicit operator helpers |
| `implicit.hpp` | Free functions | Implicit operator helpers |

Operator composition pattern:
```cpp
// Build expression from concrete operator types
auto divOp = SpatialOperator<scalar>(DivOperator<scalar>(exec, mesh, input));
auto ddtOp = TemporalOperator<scalar>(DdtOperator<scalar>(exec, mesh, phi));
Expression<scalar> eqn = ddtOp + divOp;
dsl::solve(eqn, phi, t, dt, fvSchemes, fvSolution);
```

### `include/NeoN/linearAlgebra/`

**Purpose:** Sparse linear system assembly and solving.

| Header | Class | Role |
|--------|-------|------|
| `linearSystem.hpp` | `LinearSystem<V,M,B>` | CSR matrix + COO nonLocal + COO boundary + RHS + comm |
| `matrix.hpp` | `CSRMatrix<V,I>`, `COOMatrix<V,I>` | Sparse matrix storage types |
| `sparsityPattern.hpp` | `SparsityPattern<I>`, `CooSparsityPattern<I>` | CSR / COO index arrays |
| `faceToMatrixAddress.hpp` | `FaceToMatrixAddress<I>` | Face → matrix offset mapping; `diagIdx`, `upperIdx`, `lowerIdx` |
| `solver.hpp` | `Solver`, `SolverFactory` | Runtime-selectable solver wrapper |
| `ginkgo.hpp` | `GinkgoSolver` | Ginkgo backend (default) |
| `petsc.hpp` | `PetscSolver` | PETSc backend (optional) |
| `diagonalSolver.hpp` | `DiagonalSolver` | Test-only diagonal-only solver |
| `utilities.hpp` | Free functions | Matrix utility operations |

`createEmptyLinearSystem<V>(mesh)` is the standard entry point:
```cpp
auto ls = la::createEmptyLinearSystem<scalar>(mesh);
// then operator assembly fills ls.matrix(), ls.rhs(), ls.boundaryMatrix() etc.
```

### `include/NeoN/finiteVolume/cellCentred/`

**Purpose:** Finite-volume cell-centred field types, BCs, operators, interpolation.

#### `fields/`

| Header | Class | Role |
|--------|-------|------|
| `domain.hpp` | `DomainMixin<T>` | Base mixin: `exec_`, `mesh_`, `field_`, `name` |
| `volumeField.hpp` | `VolumeField<T>` | Cell-centred field + BC vector + optional Database |
| `surfaceField.hpp` | `SurfaceField<T>` | Face-centred field (fluxes) |

#### `boundary/`

| Header | Class | Role |
|--------|-------|------|
| `volumeBoundaryFactory.hpp` | `VolumeBoundaryFactory<T>`, `VolumeBoundary<T>` | BC factory base + type-erased BC handle |
| `surfaceBoundaryFactory.hpp` | `SurfaceBoundaryFactory<T>`, `SurfaceBoundary<T>` | Surface field BC factory |
| `boundaryPatchMixin.hpp` | `BoundaryPatchMixin` | Stores patch offset range `[start, end)` and `patchID` |
| `boundaryContext.hpp` | `BoundaryContext` | Optional context passed to BC correction |
| `volume/fixedValue.hpp` | `FixedValue<T>` | Fixed-value Dirichlet BC |
| `volume/fixedGradient.hpp` | `FixedGradient<T>` | Fixed-gradient Neumann BC |
| `volume/zeroGradient.hpp` → `volume/extrapolated.hpp` | `Extrapolated<T>` | Zero-gradient BC |
| `volume/calculated.hpp` | `Calculated<T>` | No-op placeholder BC |
| `volume/symmetry.hpp` | `Symmetry<T>` | Symmetry plane BC |
| `volume/empty.hpp` | `Empty<T>` | 2D empty-patch BC |
| `volume/processor.hpp` | `Processor<T>` | Processor-boundary BC (stages local values for MPI exchange) |

#### `operators/`

| Header | Factory/Concrete | Registered name |
|--------|-----------------|-----------------|
| `divOperator.hpp` | `DivOperatorFactory<T>` | — |
| `gaussGreenDiv.hpp` | `GaussGreenDiv<T>` | `"Gauss"` |
| `gradOperator.hpp` | `GradOperatorFactory<T>` | — |
| `gaussGreenGrad.hpp` | `GaussGreenGrad<T>` | `"Gauss"` |
| `laplacianOperator.hpp` | `LaplacianOperatorFactory<T>` | — |
| `gaussGreenLaplacian.hpp` | `GaussGreenLaplacian<T>` | `"Gauss"` |
| `ddtOperator.hpp` | `DdtOperator<T>` | Euler/backward time discretisation |
| `ddtFluxCorr.hpp` | `DdtFluxCorr<T>` | Flux correction for ddtSchemes |
| `sourceTerm.hpp` | `SourceTerm<T>` | Volume-source DSL operator |
| `surfaceIntegrate.hpp` | Free functions | Face-to-cell integration helpers |

Each concrete operator registers itself via `DivOperatorFactory<T>::Register<GaussGreenDiv<T>>` CRTP.

#### `interpolation/`

| Header | Factory/Concrete | Registered name |
|--------|-----------------|-----------------|
| `surfaceInterpolation.hpp` | `SurfaceInterpolationFactory<T>` | — |
| `linear.hpp` | `Linear<T>` | `"linear"` |
| `upwind.hpp` | `Upwind<T>` | `"upwind"` |
| `limitedLinear.hpp` | ... | various limiter names |

#### `stencil/`
Stencil data structures cached in `mesh.stencilDB()`. Used by higher-order interpolation schemes.

#### `auxiliary/`
Helper free functions for auxiliary field operations (e.g., flux corrections).

#### `faceNormalGradient/`
Face-normal gradient operators used by Laplacian and diffusion terms.

### `include/NeoN/timeIntegration/`

| Header | Class | Role |
|--------|-------|------|
| `timeIntegration.hpp` | `TimeIntegration<SolType>`, `TimeIntegratorBase<SolType>` | Factory + type-erased time integrator |
| `sundials.hpp` | Sundials wrappers | CVODE/ARKode integration (optional) |
| Runge-Kutta headers | `RungeKutta*` | Explicit RK schemes |

Time integrator is selected by `fvSchemes.subDict("timeIntegration").get<string>("type")`.

### `include/NeoN/helpers/`

Miscellaneous small utilities not belonging to a specific module.

## Naming Conventions

**Files:**
- Headers: `camelCase.hpp` (e.g., `volumeField.hpp`, `gaussGreenDiv.hpp`)
- Implementation: `camelCase.cpp` matching the header name
- Test files: `test_<featureName>.cpp` in `test/`

**Directories:**
- `camelCase` for subdirectories within modules (e.g., `cellCentred`, `unstructured`)
- Module names are lowercase plural nouns (e.g., `fields`, `operators`, `interpolation`)

**Classes:**
- `PascalCase` for classes (e.g., `VolumeField`, `GaussGreenDiv`, `LinearSystem`)
- `camelCase` for member functions and variables

**Namespace hierarchy:**
- `NeoN` — root namespace
- `NeoN::dsl` — DSL operators and expression
- `NeoN::la` — linear algebra
- `NeoN::mpi` — MPI utilities
- `NeoN::finiteVolume::cellCentred` — FV cell-centred module
- `NeoN::finiteVolume::cellCentred::volumeBoundary` — volume BC implementations
- `NeoN::timeIntegration` — time integration

## Entry Points for External Consumers

External code (e.g., NeoFOAM) consumes NeoN through:

1. **Top-level header:** `include/NeoN/NeoN.hpp` — includes all modules.

2. **Mesh construction:** Call a factory function from `include/NeoN/mesh/unstructured/unstructuredMesh.hpp`, or construct `UnstructuredMesh` directly with pre-built `Vector` geometry data and a `BoundaryMesh`.

3. **Field creation:**
   ```cpp
   // Create BCs from Dictionary
   std::vector<VolumeBoundary<scalar>> bcs = ...;
   VolumeField<scalar> phi(exec, "phi", mesh, bcs);
   ```

4. **DSL expression and solve:**
   ```cpp
   auto divOp = fcc::DivOperator<scalar>(exec, mesh, input);
   Expression<scalar> eqn = dsl::ddt(phi) + divOp;
   dsl::solve(eqn, phi, t, dt, fvSchemes, fvSolution);
   ```

5. **Direct linear system assembly** (bypassing DSL):
   ```cpp
   auto ls = la::createEmptyLinearSystem<scalar>(mesh);
   // fill ls.matrix(), ls.rhs() manually
   la::Solver solver(exec, fvSolution);
   solver.solve(ls, phi.internalVector());
   ```

## Where to Add New Code

**New spatial operator (e.g., a new divergence scheme):**
- Implementation header: `include/NeoN/finiteVolume/cellCentred/operators/<myScheme>.hpp`
- Implementation source: `src/finiteVolume/cellCentred/operators/<myScheme>.cpp`
- Inherit from `DivOperatorFactory<T>::Register<MyScheme<T>>`, define `static std::string name()` returning the scheme name string.
- Register explicit instantiations at bottom of header: `template class MyScheme<scalar>; template class MyScheme<Vec3>;`

**New boundary condition:**
- Header: `include/NeoN/finiteVolume/cellCentred/boundary/volume/<myBC>.hpp`
- Inherit from `VolumeBoundaryFactory<T>::Register<MyBC<T>>`.
- Implement `correctBoundaryCondition(Field<ValueType>&)`.

**New surface interpolation scheme:**
- Header: `include/NeoN/finiteVolume/cellCentred/interpolation/<myScheme>.hpp`
- Inherit from `SurfaceInterpolationFactory<T>::Register<MyScheme<T>>`.

**New linear solver backend:**
- Header: `include/NeoN/linearAlgebra/<myBackend>.hpp`
- Source: `src/linearAlgebra/<myBackend>.cpp`
- Inherit from `SolverFactory::Register<MyBackend>`.
- Implement `solve()` and `solveDist()` overloads for `scalar` and `Vec3`.

**New mesh factory:**
- Add free function to `include/NeoN/mesh/unstructured/unstructuredMesh.hpp`.
- Implement in `src/mesh/unstructured/`.

**New test:**
- File: `test/test_<featureName>.cpp`
- Use Catch2. MPI tests use `NeoN_MPI_SIZE` cmake variable to set process count.

## Special Directories

**`build/`:**
- Purpose: Out-of-tree CMake build output. One subdirectory per preset (`develop`, `production`, `profiling`).
- Generated: Yes
- Committed: No

**`cmake/`:**
- Purpose: CPM.cmake auto-fetch scripts, find-module helpers, CMake utilities.
- Generated: No
- Committed: Yes

**`src/NeoN/` (when this repo is used as NeoFOAM's submodule):**
- This entire NeoN repo appears at `src/NeoN/` inside NeoFOAM.
- NeoFOAM includes NeoN headers as `#include "NeoN/<module>/<header>.hpp"`.

---

*Structure analysis: 2026-05-10*
