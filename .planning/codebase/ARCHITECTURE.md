<!-- refreshed: 2026-05-10 -->
# Architecture

**Analysis Date:** 2026-05-10

## System Overview

```text
┌──────────────────────────────────────────────────────────────────┐
│                    OpenFOAM Layer (Foam::)                        │
│  fvMesh · volScalarField · volVectorField · surfaceScalarField   │
│  Time · IOdictionary · fvSchemes · fvSolution                    │
└────────────────────────┬─────────────────────────────────────────┘
                         │  Adapter Layer (NeoFOAM::)
          ┌──────────────▼──────────────────────────────────────┐
          │  datastructures/    auxiliary/      compatibility/   │
          │  MeshAdapter        readers.hpp     fvSchemes.cpp    │
          │  RunTime            convert.hpp     fvSolution.cpp   │
          │  PDESolver          typeConversion  writers.hpp      │
          └──────────────────────────┬──────────────────────────┘
                                     │
          ┌──────────────────────────▼──────────────────────────┐
          │           NeoN Layer (NeoN::)                        │
          │  UnstructuredMesh · VolumeField · SurfaceField       │
          │  dsl::Expression · la::LinearSystem · la::Solver     │
          │  Executor (Serial/CPU/GPU) · Database · MPI          │
          └──────────────────────────────────────────────────────┘
```

NeoFOAM is a one-way adapter layer that bridges OpenFOAM data structures and workflow conventions to the NeoN portable compute backend. The key design decision is that NeoFOAM owns no solver logic of its own — it converts OpenFOAM inputs into NeoN types at startup, delegates all computation to NeoN's DSL and linear algebra layer, and converts results back to OpenFOAM format for I/O. Users continue to use standard OpenFOAM case directories (`fvSchemes`, `fvSolution`, boundary condition files) without modification.

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

1. `Foam::Time` is constructed by the solver app (`examples/neoIcoFoam/neoIcoFoam.cpp`).
2. `createAdapterRunTime(runTime)` reads `executor` and `allocator` from `controlDict`, constructs `NeoN::Executor`, then calls `createMesh(exec, runTime)` which constructs `MeshAdapter`. `MeshAdapter`'s constructor calls `readOpenFOAMMesh()` — this copies face areas, cell volumes, face centres, owners/neighbours, and flattened boundary data from OF into NeoN device memory.
3. `RunTime` is populated with references to the mesh, NeoN database, executor, and pre-converted `fvSchemesDict`/`fvSolutionDict`.
4. OF fields (`ofP`, `ofU`, `ofPhi`) are read from disk via standard `Foam::IOobject`.
5. `constructAndRegister()` converts each OF field to a NeoN field and registers it in `fvcc::VectorCollection`. Internal data is copied with `fromFoamField(exec, field.primitiveField())` (reinterpret-cast, zero-copy where possible); boundary conditions are reconstructed via `readVolBoundaryConditions()` / `readSurfaceBoundaryConditions()`.
6. `mapFvSchemes()` and `mapFvSolution()` translate OF scheme/solver names to NeoN/Ginkgo equivalents (e.g. `Euler`→`BDF1`, `PCG`→`solver::Cg`, `DIC`→`preconditioner::Ic`).

### Time Loop (neoIcoFoam PISO)

1. `PDESolver<Vec3> UEqn(dsl::imp::ddt(U) + dsl::imp::div(phi, U) - dsl::imp::laplacian(nu, U), U, rt)` — builds the momentum expression using NeoN's DSL operators.
2. `UEqn.solve(-1.0 * dsl::exp::grad(p))` — assembles CSR linear system and invokes NeoN's Ginkgo-based iterative solver with preconditioner from `fvSolutionDict`.
3. `computeRAUandHByA(UEqn)` — reads the assembled matrix diagonal to compute `rAU = 1/aP` and `HbyA = H[U]/aP` via `NeoN::la::scaledInvDiagNegLUx`.
4. `flux(hByA)` — computes face-normal flux using linear surface interpolation; proc-boundary faces handled by a dedicated `parallelFor` kernel over `[nInt+nBnd, nTotal)`.
5. `PDESolver<scalar> pEqn(laplacian(rAU, p) - div(phiHbyA), p, rt)` — pressure Poisson equation.
6. `pEqn.solve()` — solves pressure; `PDESolver::SetReference` pins the reference cell on rank 0 only (distributed-safe).
7. `updateFaceVelocity(phiHbyA, pEqn, phi)` — corrects face velocity using assembled pressure matrix coefficients for both internal and proc-boundary faces.
8. `updateVelocity(hByA, rAU, p, U)` — cell velocity correction via Gauss-Green gradient of p.

### Write-back

1. `write(p, mesh)` / `write(U, mesh)` (in `writers.cpp`) — copies NeoN device field to host, reconstructs an `Foam::volScalarField`/`volVectorField` and calls `field.write()`, delegating file I/O entirely to OpenFOAM.

## Design Patterns

**Adapter Pattern** — `MeshAdapter : public Foam::fvMesh` extends fvMesh to carry `NeoN::UnstructuredMesh`; callers that only need OF mesh treat it as `fvMesh`, callers needing NeoN call `.nfMesh()`.

**Traits / Type Mapping** — `TypeMap<FoamType>` specializations provide compile-time mapping between OF and NeoN types, used by all `constructFrom` / `fromFoamField` templates.

**Factory Functions** — `createAdapterRunTime()`, `createMesh()`, `createExecutor()` centralize object construction and executor selection, keeping solver apps thin.

**Registry / Database** — NeoN `Database` (inside `RunTime::db`) is used as a runtime object registry. `readOrCreate<T>()` inserts or retrieves named objects (e.g. `LinearSystem`) to avoid redundant allocation across time steps.

**DSL Expression Builder** — NeoN's `dsl::imp::*` and `dsl::exp::*` free functions compose implicit/explicit operators into a `dsl::Expression` object before assembly, mirroring OpenFOAM's `fvMatrix` idiom.

**Post-Assembly Hook** — `PDESolver::SetReference` is a `PostAssemblyBase<T>` functor injected into the solve pipeline to pin a pressure reference cell; only executes on MPI rank 0.

## Module Boundaries

**`datastructures/`** is the foundational layer. All other modules depend on `MeshAdapter` and `RunTime`. `MeshAdapter` is the only class that owns both OF and NeoN mesh views simultaneously. Nothing outside `datastructures/` should construct a `NeoN::UnstructuredMesh` directly.

**`auxiliary/`** is the conversion utilities layer. It depends on `datastructures/` (for `RunTime`, `nfMesh`) and on NeoN field types. It has no knowledge of solver algorithms. `convert.hpp` handles scalar/vector/dictionary type conversion; `typeConversion.hpp` provides the `TypeMap` traits; `readers.hpp` builds NeoN fields from OF fields; `writers.hpp` serializes NeoN fields back via OF I/O; `comparison.hpp` provides test-only equality operators.

**`compatibility/`** is the configuration translation layer. It depends only on `NeoN::Dictionary`. `fvSchemes.cpp` renames time-integration schemes (e.g. `Euler`→`BDF1`). `fvSolution.cpp` maps OF solver/preconditioner names to Ginkgo names and wraps preconditioners in `preconditioner::Schwarz` for distributed runs (MPI rank count > 1). Neither file touches mesh or field data.

**`algorithms/`** implements PISO/SIMPLE helper operations (constrainHbyA, computeRAUandHByA, updateFaceVelocity, updateVelocity, flux) as free functions. It depends on `datastructures/PDESolver`, NeoN `la::*`, and NeoN `finiteVolume::cellCentred::*`. It does not depend on `auxiliary/`.

**`fvcc/surfaceInterpolation/`** registers OpenFOAM RunTimeSelectionTable entries (e.g. `linear`) that map to NeoN interpolation schemes. This is the sole OpenFOAM RTST integration point.

**`src/NeoN/`** (git submodule) provides `NeoN::Executor`, `NeoN::UnstructuredMesh`, `NeoN::finiteVolume::cellCentred::{VolumeField, SurfaceField}`, the DSL, `NeoN::la::LinearSystem`, and Ginkgo-backed solvers. NeoFOAM treats it as a black-box library — no NeoN sources are modified.

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

**What happens:** Code instantiates `NeoN::SerialExecutor{}` instead of propagating the executor from `RunTime`.
**Why it's wrong:** It silently disables GPU/multi-CPU execution even when the user configures a GPU executor; compute happens on the wrong device.
**Do this instead:** Always pass `RunTime::exec` or accept `const NeoN::Executor& exec` as a parameter, as done in `src/algorithms/pressureVelocityCoupling.cpp`.

### Using OF full-size face arrays for boundary kernels

**What happens:** A kernel indexes boundary faces using the full `mesh.faceCentres()` / `mesh.faceOwner()` arrays (size = nInternalFaces + nAllBoundaryFaces in OF ordering).
**Why it's wrong:** NeoN's `BoundaryMesh` (`bm.cf()`, `bm.sf()`, `bm.faceCells()`) and `SurfaceField` use the compressed boundary-only layout with proc patches at the tail; the index ranges do not match OF's full arrays.
**Do this instead:** Use `mesh.boundaryMesh().cf()` / `sf()` / `faceCells()` for boundary-only access; use `mesh.faceAreas()` / `mesh.faceCentres()` only for full-mesh (internal + boundary) kernels, as in `src/algorithms/pressureVelocityCoupling.cpp`.

## Error Handling

**Strategy:** Assertions via `NF_ASSERT` / `NF_ASSERT_EQUAL` macros for developer invariants; `Foam::FatalError` for OpenFOAM-facing fatal errors; `std::runtime_error` thrown for unsupported configuration (e.g. GAMG solver requested). Convergence feedback via `NeoN::Logging::info` after each linear solve.

## Cross-Cutting Concerns

**Logging:** `NeoN::Logging::info/warn` (fmt-style format strings) used throughout adapter and algorithm code; `Foam::Info` used in solver app and OpenFOAM-facing code.
**Validation:** `NF_ASSERT_EQUAL` in `readers.hpp` guards face count invariants during field construction. Catch2 `REQUIRE_THAT` / `RangeEquals` in tests validates OF ↔ NeoN numerical equivalence.
**MPI/Distribution:** `NeoN::mpi::Environment` provides rank/size queries. Distributed preconditioner wrapping in `compatibility/fvSolution.cpp` is conditional on `mpiEnv.isInitialized() && mpiEnv.sizeRank() > 1`.

---

*Architecture analysis: 2026-05-10*
