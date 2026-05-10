<!-- refreshed: 2026-05-10 -->
# NeoN Architecture

**Analysis Date:** 2026-05-10

## Overview

NeoN is a C++20 library providing a portable compute substrate for CFD solvers. Its central design goal is hardware portability: the same PDE solver code runs on serial CPU, multi-core CPU (OpenMP/threads), and GPU (CUDA/HIP/SYCL) without modification. Portability is achieved through the Kokkos programming model and a `std::variant`-based Executor abstraction that all data containers carry. NeoN is not a standalone solver — it is consumed by adapter layers such as NeoFOAM, which bridge it to OpenFOAM data structures.

The library is structured as a stack of layers: a portable data container layer (`Vector`, `Field`), a mesh layer (`UnstructuredMesh`, `BoundaryMesh`), a finite-volume discretisation layer (`VolumeField`, `SurfaceField`, operators), a DSL layer (`Expression`, `SpatialOperator`, `TemporalOperator`) for composing PDE terms, and a linear algebra layer (`LinearSystem`, `SparsityPattern`, Ginkgo/PETSc solvers). Each layer depends only on layers below it. A runtime-selection factory (`RuntimeSelectionFactory`) enables plugin-style registration of schemes, boundary conditions, and solvers by name string.

Distributed MPI support is woven through the mesh and linear algebra layers. Processor-boundary patches are a special patch type in `BoundaryMesh`; the `LinearSystem` carries a separate `nonLocalMatrix_` (COO) for off-rank coupling coefficients, a `CommunicationPattern` describing the exchange topology, and a `communicate()` method that performs `MPI_Alltoallv` to fold received ghost-cell contributions into the local diagonal. The Ginkgo solver backend has a dedicated distributed path (`ginkgoDistributed.cpp`) selected automatically when `commPattern.sendCounts` is non-empty.

## Core Abstractions

### Executor
- **Type:** `std::variant<SerialExecutor, CPUExecutor, GPUExecutor>`
- **File:** `include/NeoN/core/executor/executor.hpp`
- All three concrete executors (`SerialExecutor`, `CPUExecutor`, `GPUExecutor`) expose identical allocation (`alloc<T>`, `free`) and Kokkos-view creation interfaces. Dispatch is via `std::visit`.
- `SerialExecutor` maps to `Kokkos::Serial` + `Kokkos::HostSpace`.
- `CPUExecutor` maps to `Kokkos::OpenMP` or `Kokkos::Threads`.
- `GPUExecutor` maps to `Kokkos::Cuda`, `Kokkos::HIP`, or `Kokkos::SYCL` depending on compile-time flags.
- `createDefaultExecutor()` selects the highest-capability backend detected at compile time.
- **Rule:** Executors must never be hard-coded. Always thread the executor from the call site.

### Vector (data container)
- **Template:** `Vector<ValueType>` — `include/NeoN/core/vector/vector.hpp`
- Executor-aware allocation: raw pointer `data_` managed via the executor's `alloc`/`free`.
- Direct `operator[]` is deleted to prevent inadvertent host access of device memory.
- Provides `.view()` returning `View<ValueType>` (a non-owning `Kokkos::View` span) for use inside `parallelFor` kernels.
- `copyToExecutor(dstExec)` performs cross-device copies.
- `Array<T>` (`include/NeoN/core/array.hpp`) is a lower-level raw allocation (no math), used by sparsity structures.

### Field (boundary-aware container)
- **Template:** `Field<ValueType>` — `include/NeoN/fields/field.hpp`
- Holds `Vector<ValueType> internalVector_` (cell/face data) and `BoundaryData<ValueType> boundaryData_` (computed values, reference values, value fractions, reference gradients, offsets).
- `BoundaryData` (`include/NeoN/fields/boundaryData.hpp`) organises boundary faces by patch via an offset vector. All patch sub-arrays are contiguous in a single `Vector` and indexed by `[offset[i], offset[i+1])`.

### UnstructuredMesh
- **Class:** `UnstructuredMesh` — `include/NeoN/mesh/unstructured/unstructuredMesh.hpp`
- Stores mesh geometry as `Vector` members: `points_`, `cellVolumes_`, `cellCentres_`, `faceAreas_`, `faceCentres_`, `magFaceAreas_`, `faceOwner_`, `faceNeighbour_`.
- Contains a `BoundaryMesh boundaryMesh_` which stores boundary-face-indexed geometry (`cf`, `cn`, `sf`, `magSf`, `nf`, `delta`, `weights`, `deltaCoeffs`) plus the patch offset vector.
- `BoundaryMesh` stores `procBoundaryPatches_` count and `neighbourRank_` vector — processor patches are always stored after physical patches: indices `[nBoundaryFaces(), nTotalFaces())` are processor faces.
- Carries a mutable `stencilDataBase_` (`Dictionary`) for caching computed stencils.
- `globalOffset_` provides the global cell index offset for distributed decomposition.
- Factory functions: `create1DUniformMesh`, `create2DUniformMesh`, `create3DUniformMesh`, `create1DUniformMeshPart`.

### VolumeField / SurfaceField
- **Files:** `include/NeoN/finiteVolume/cellCentred/fields/volumeField.hpp`, `surfaceField.hpp`
- Both inherit from `DomainMixin<ValueType>` (`include/NeoN/finiteVolume/cellCentred/fields/domain.hpp`) which holds: `name`, `exec_`, `mesh_` (const ref), `field_` (a `Field<ValueType>`).
- `VolumeField` additionally holds `std::vector<VolumeBoundary<ValueType>> boundaryConditions_` and an optional `Database*`.
- `correctBoundaryConditions()` iterates the BC vector and calls each BC's `correctBoundaryCondition(field_)`.

### DSL — Expression
- **Namespace:** `NeoN::dsl`
- **Files:** `include/NeoN/dsl/expression.hpp`, `spatialOperator.hpp`, `temporalOperator.hpp`, `operator.hpp`
- `Expression<ValueType>` aggregates `std::vector<TemporalOperator<ValueType>>` and `std::vector<SpatialOperator<ValueType>>`.
- `SpatialOperator<ValueType>` is a **type-erased wrapper** (Concept + Model pattern): it holds a `std::unique_ptr<OperatorConcept>` and dispatches `explicitOperation(Vector&)` and `implicitOperation(LinearSystem&)` polymorphically — without virtual base classes on the stored type.
- Operators tagged `Operator::Type::Explicit` contribute source terms; `Operator::Type::Implicit` contribute matrix coefficients.
- `Expression::assemble(mesh, t, dt)` calls `createEmptyLinearSystem(mesh)`, then dispatches to all implicit spatial and temporal operators.
- `dsl::solve(exp, solution, t, dt, fvSchemes, fvSolution)` is the top-level entry point: reads scheme config, selects a `TimeIntegration` strategy, either calls the explicit integrator path or assembles and passes to `la::Solver`.

### LinearSystem
- **Template:** `LinearSystem<ValueType, MatrixType, BoundaryMatrixType>` — `include/NeoN/linearAlgebra/linearSystem.hpp`
- Internal CSR matrix: `matrix_` (`CSRMatrix<ValueType, localIdx>`)
- Off-rank coupling: `nonLocalMatrix_` (`COOMatrix<ValueType, localIdx>`) — receives processor-face contributions before MPI exchange
- Boundary source/coefficients: `boundaryMatrix_` + `boundaryRhs_` — local Dirichlet-type contributions
- `CommunicationPattern commPattern_` — `sendCounts`, `recvIdx`, `boundaryMapVector`, `mpi::Environment`
- `communicate(commPattern)` runs `MPI_Alltoallv` and folds received values into the local diagonal
- `FaceToMatrixAddress<IndexType>` (`include/NeoN/linearAlgebra/faceToMatrixAddress.hpp`) maps each mesh face to its CSR offset: `ownerOffset_[f]` → `A[own, nei]`, `neighbourOffset_[f]` → `A[nei, own]`, `diagOffset_[celli]` → `A[celli, celli]`.

### Linear Solvers
- **File:** `include/NeoN/linearAlgebra/solver.hpp`
- `SolverFactory` is a `RuntimeSelectionFactory` keyed by `"solver"` string in the `fvSolution` dict.
- `Solver::solve()` automatically switches to `solveDist()` when `commPattern.sendCounts` is non-empty.
- **Ginkgo** (`include/NeoN/linearAlgebra/ginkgo.hpp`, `src/linearAlgebra/ginkgo/ginkgo.cpp`): default backend; wraps Ginkgo executor obtained from `getGkoExecutor(exec)`; supports JSON-configured solver factories.
- **Ginkgo distributed** (`src/linearAlgebra/ginkgo/ginkgoDistributed.cpp`): handles distributed MPI solves.
- **PETSc** (`include/NeoN/linearAlgebra/petsc.hpp`): optional alternative backend.
- **Diagonal solver** (`include/NeoN/linearAlgebra/diagonalSolver.hpp`): fallback for testing.

### RuntimeSelectionFactory
- **File:** `include/NeoN/core/runtimeSelectionFactory.hpp`
- CRTP + static-bool self-registration pattern.
- Derived classes inherit `Register<Derived>` and are added to the base class's static lookup table at program startup.
- Used by: `VolumeBoundaryFactory`, `SurfaceBoundaryFactory`, `DivOperatorFactory`, `GradOperatorFactory`, `LaplacianOperatorFactory`, `SurfaceInterpolationFactory`, `SolverFactory`, `TimeIntegratorBase`.

## Data Flow

### PDE Term → Matrix Contribution → Linear Solve

1. **Operator construction** — A concrete operator such as `GaussGreenDiv<scalar>` (`include/NeoN/finiteVolume/cellCentred/operators/gaussGreenDiv.hpp`) is instantiated with `exec`, `mesh`, and an `Input` (Dictionary or TokenList carrying scheme name). It wraps a `SurfaceInterpolation<ValueType>` obtained via `SurfaceInterpolationFactory::create`.

2. **Expression assembly** — The operator is wrapped in a `SpatialOperator<scalar>` (type erasure) and added to an `Expression<scalar>`. Multiple operators are composed with `operator+` / `operator-` on the expression.

3. **`dsl::solve` called** — `include/NeoN/dsl/solver.hpp`. Reads scheme config (`exp.read(fvSchemes)`). If temporal operators are present and an explicit integrator is selected, delegates to `TimeIntegration::solve`. Otherwise calls `detail::iterativeSolveImpl`.

4. **Linear system creation** — `createEmptyLinearSystem<ValueType>(mesh)` calls `createSparsityPatternFaceToMatrixAddress(mesh)` which builds `SparsityPattern` (CSR rowOffs + colIdxs from mesh topology) and `CommunicationPattern` (for distributed meshes). Returns a zero-initialised `LinearSystem`.

5. **Operator assembly into LinearSystem** — `exp.assemble(t, dt, ls)`:
   - Implicit spatial operators: `GaussGreenDiv::div(ls, faceFlux, phi, coeff)` → `computeDivProcBoundImpl` (processor patches → `nonLocalMatrix_`), `computeDivImp` (internal faces → `matrix_`), `computeDivBoundImpl` (boundary faces → `boundaryMatrix_` + `boundaryRhs_`).
   - Implicit temporal operators (e.g. `ddt`): contribute to diagonal and RHS.

6. **Explicit source subtracted** — `exp.explicitOperation(nCells)` accumulates explicit terms; result is subtracted from `ls.rhs()` in a `parallelFor` kernel.

7. **Solve** — `la::Solver::solve(ls, solution.internalVector())`. For distributed meshes, `ls.communicate(commPattern)` executes `MPI_Alltoallv` to exchange `nonLocalMatrix_` coefficients before the solve, folding them into the local diagonal. Ginkgo (or PETSc) then inverts the assembled system.

8. **Field update** — Solution is written back into `solution.internalVector()`. Boundary conditions are re-evaluated with `correctBoundaryConditions()`.

### Boundary Condition Evaluation

1. `VolumeField::correctBoundaryConditions()` iterates `boundaryConditions_`.
2. Each `VolumeBoundary<ValueType>` holds a `std::unique_ptr<VolumeBoundaryFactory<ValueType>>` created via `RuntimeSelectionFactory` by name (e.g. `"fixedValue"`, `"zeroGradient"`, `"processor"`).
3. The BC's `correctBoundaryCondition(Field<ValueType>&)` writes into `field_.boundaryData().value()` (and optionally `refGrad_`, `valueFraction_`).
4. Processor BC (`include/NeoN/finiteVolume/cellCentred/boundary/volume/processor.hpp`): copies own cell values into the boundary data. Actual ghost-cell exchange is handled at the linear-system level via `LinearSystem::communicate()`.

## Distributed Architecture

### Mesh Decomposition
- `UnstructuredMesh::globalOffset()` returns the global cell index of the first local cell.
- `BoundaryMesh` stores physical patches first, processor patches last. `nProcBoundaryPatches()` gives the count. `neighbourRank(patchID)` returns the MPI rank of the patch's remote partner (-1 for physical patches).
- `BoundaryMesh::isDistributed()` returns true when `nProcBoundaryFaces() > 0`.

### Communication Pattern
- `CommunicationPattern` (`include/NeoN/distributed/communicationPattern.hpp`): `sendCounts[rank]` = number of values to send to each rank; `recvIdx` = global indices of cells to receive; `boundaryMapVector` = local boundary face → matrix address mapping.
- Built by `computeCommunicationPattern(mesh)` (`include/NeoN/mesh/unstructured/unstructuredMesh.hpp`).
- Also constructed alongside `FaceToMatrixAddress` by `createSparsityPatternFaceToMatrixAddress(mesh)`.

### Field Halo Exchange
- `BoundaryData` contains `value()`, `refValue()`, `refGrad()` etc. for all boundary faces including processor faces.
- The processor BC (`volume/processor.hpp`) fills `value[i] = internalVector[faceCells[i]]` — it stages the local side's data.
- `HalfDuplexCommBuffer` (`include/NeoN/core/mpi/halfDuplexCommBuffer.hpp`) and `FullDuplexCommBuffer` provide non-blocking point-to-point MPI exchange primitives used for field halo exchange.
- `mpi::Environment` (`include/NeoN/core/mpi/environment.hpp`) wraps `MPI_Comm`, rank, and size; supports `MPI_THREAD_MULTIPLE` when `NF_REQUIRE_MPI_THREAD_SUPPORT` is defined.

### Matrix Exchange
- Processor-face operator contributions go to `nonLocalMatrix_` (COO), not the local CSR matrix.
- `LinearSystem::communicate()` sends these via `MPI_Alltoallv` and adds the received values to the corresponding diagonal entries of `matrix_` (ghost-cell coupling folded into Dirichlet-like diagonal fixup).
- `Solver::solve()` detects distributed mode by checking `commPattern.sendCounts.size() > 0` and routes to `solveDist()`.

### SurfaceField Dual-Storage Invariant
- `SurfaceField<T>` internal data covers faces `[0, nInternalFaces)`.
- Boundary face data starts at `internalVector()[nInternalFaces + boundaryOffset]` for physical patches.
- Processor-boundary face data starts at `internalVector()[nInternalFaces + nBoundaryFaces]`.
- `BoundaryMesh::cf()`, `sf()`, `faceCells()` etc. are **compressed** (boundary-face indexed, starting at 0). Kernels iterating over processor faces must use `bm.*` accessors, not the full `mesh.faceCentres()` which includes all internal faces.

## Design Patterns

### Type Erasure (DSL Operators)
`SpatialOperator<ValueType>` and `TemporalOperator<ValueType>` use the Concept + Model (non-virtual type erasure) pattern. Concrete operators implement either `explicitOperation(Vector<T>&)` or `implicitOperation(LinearSystem<T>&)` (enforced by C++20 concepts `HasExplicitOperator`, `HasImplicitOperator`). The `OperatorModel<T>` inner struct wraps the concrete type and erases it behind `OperatorConcept*`. This allows storing heterogeneous operators in `std::vector` and cloning via the Prototype pattern.

### CRTP Self-Registration
`RuntimeSelectionFactory<Base, Parameters<Args...>>` plus the `Register<Derived>` CRTP mixin. Derived classes declare `static std::string name()` and inherit `Register<Derived>` as a base. A `static bool REGISTERED` member is initialised via the class's entry in the factory table. The `static_assert` trick forces the linker to include the registration even in static library builds.

### Policy-Based Allocation
`Executor` variants carry an `AllocatorStrategy` (`include/NeoN/core/memory/allocator.hpp`) injected at construction. The `DefaultAllocator` uses Kokkos managed allocation. Umpire-based allocator is an optional alternative (`NeoN_WITH_UMPIRE`). This decouples memory policy from execution policy.

### Kokkos Kernel Dispatch
All parallel loops use `parallelFor(exec, {start, end}, NEON_LAMBDA(...){...}, "kernelName")` where `NEON_LAMBDA` expands to `KOKKOS_LAMBDA` (adds `__host__ __device__` qualifiers). `std::visit` on the `Executor` variant dispatches to the appropriate Kokkos execution space.

### View Pattern for Kernel Access
`Vector<T>::view()` returns `View<T>` (a non-owning span backed by a `Kokkos::View`). Kernels capture views, not `Vector` objects, preventing accidental host-memory access on GPU. `view()` on temporaries (rvalue) is deleted to prevent dangling pointers.

## Error Handling

**Strategy:** Assertion macros (`NF_ASSERT`, `NF_ERROR_EXIT`) defined in `include/NeoN/core/error.hpp`. They terminate with a message in debug builds. No exception-based error propagation in hot paths.

**Patterns:**
- Size/executor mismatches caught at construction (`LinearSystem::validate()`, `DomainMixin` constructor).
- Runtime-selection key misses call `keyExistsOrError(key)` which terminates with a descriptive message listing registered keys.
- `Kokkos::abort` inside GPU kernels for out-of-bounds sparsity access (`SparsityView::entry`).

## Cross-Cutting Concerns

**Logging:** `Logging::SupportsLoggingMixin` (`include/NeoN/core/logging.hpp`) mixed into executors. `getLogger(exec)` / `setLogger(exec, logger)` allow attaching a `BaseLogger`. `fenceIfLogger` forces GPU synchronisation before logging.

**Validation:** Constructor-level `NF_ASSERT` checks; `validate()` private methods on `LinearSystem`, `SparsityPattern`, `BoundaryMesh`.

**Configuration:** `Dictionary` wraps nlohmann_json for runtime configuration. `Input` is `std::variant<Dictionary, TokenList>`. Scheme names and solver configs are read from `Dictionary` at operator/solver construction time.

**Database:** `Database` (`include/NeoN/core/database/database.hpp`) is an optional field registry (name → `Collection`). `VolumeField` can register itself via `FieldDatabaseMixin`. Used for old-time field access in time integration (`OldTimeCollection`).

## Anti-Patterns

### Hard-coded SerialExecutor

**What happens:** Code calls `SerialExecutor{}` or `CPUExecutor{}` directly inside a class constructor or operator.
**Why it's wrong:** Breaks portability — the field or kernel runs on CPU even when the caller passed a `GPUExecutor`. Silent correctness errors if GPU data is accessed from a CPU kernel.
**Do this instead:** Accept `const Executor& exec` as a parameter and store it in `exec_`. Use `createDefaultExecutor()` only at the top-level application entry point.

### Direct `operator[]` on Vector

**What happens:** Code attempts `vec[i]` on a `Vector<T>`.
**Why it's wrong:** Deleted by design to prevent invalid host dereferencing of device memory.
**Do this instead:** Call `vec.view()` to obtain a `View<T>`, then index inside a `NEON_LAMBDA` passed to `parallelFor`.

### Stale Boundary Data After Matrix Communication

**What happens:** `SurfaceField` boundary data is updated for internal boundaries but processor-face `internalVector()[nIntF + nBndF + procFaceOffset]` is not re-synced after `LinearSystem::communicate()`.
**Why it's wrong:** Post-solve field reconstruction (`H/A` operations) reads stale ghost values, producing processor-boundary discontinuities.
**Do this instead:** After `solve`, re-run `correctBoundaryConditions()` and perform a halo exchange using `HalfDuplexCommBuffer` before any operator that reads boundary face values.

---

*Architecture analysis: 2026-05-10*
