# NeoFOAM Distributed — Parallel neoIcoFoam

## What This Is

NeoFOAM is an adapter layer that connects OpenFOAM data structures to the NeoN GPU/CPU compute backend,
replacing standard OpenFOAM solvers with NeoN-accelerated variants. This milestone targets making
`neoIcoFoam` run correctly in parallel (MPI), validated on the cylinder3D case with 2–4 MPI ranks,
producing field results within L∞ tolerance of the single-process case — without changing the
OpenFOAM user workflow.

## Core Value

neoIcoFoam on cylinder3D with 4 MPI ranks produces field results within L∞ tolerance of the
serial run.

## Requirements

### Validated

- ✓ Serial neoIcoFoam runs on cylinder2D — confirmed via tutorials/cylinder2D
- ✓ NeoN executor-based field operations (VolumeField, SurfaceField) — established in NeoN core
- ✓ OpenFOAM mesh bridging (foamMesh / meshAdapter) — implemented in src/datastructures/
- ✓ Field conversion OF ↔ NeoN (volScalarField, volVectorField) — implemented in src/auxiliary/convert.cpp
- ✓ fvSchemes / fvSolution compatibility layer — implemented in src/compatibility/
- ✓ PISO algorithm structure in serial — implemented in src/algorithms/pressureVelocityCoupling.cpp
- ✓ Distributed UnstructuredMesh construction (4-rank hierarchical) — test_distributedUnstructuredMesh passes
- ✓ Distributed pressure-velocity coupling test skeleton — test_distributedPressureVelocityCoupling exists

### Active

**NeoFOAM correctness (critical path to running parallel neoIcoFoam):**
- [ ] `fixedValue` BC parsed correctly in parallel (readers.hpp token-count-1 bug)
- [ ] `procBoundary0to1` patch names handled in `readSurfaceBoundaryConditions`
- [ ] `PDESolver::needReference_` default-initialised to `false` (UB on every solve)
- [ ] `adjustPhi` and `constrainPressure` wired into the PISO loop in neoIcoFoam.cpp
- [ ] `continuityErrs` reporting added to neoIcoFoam PISO loop
- [ ] Reference cell convention correct for arbitrary rank decomposition (not rank-0 assumption)
- [ ] SurfaceField internalVector proc-face slots overwritten with received ghost value after MPI exchange
- [ ] `foamMesh.cpp` dead code path resolved (ODR risk with meshAdapter.cpp)

**NeoN distributed infrastructure correctness (must be stable for neoIcoFoam to converge):**
- [ ] `faceToMatrixAddress` diagIdx/upperIdx/lowerIdx correct in distributed mode
- [ ] `MPI_Alltoallv` uses separate sendCounts/recvCounts (symmetric assumption breaks non-uniform decomp)
- [ ] `SurfaceField::correctBoundaryConditions` overwrites internalVector proc-face entries with received values
- [ ] `std::abs` → `Kokkos::abs` in `updateWeights` proc-face kernel (GPU compilation failure)
- [ ] `DiagonalSolver::solveDist` not a silent no-op
- [ ] `removeBoundaryContributions` RHS non-local subtraction enabled (currently commented out)
- [ ] `deltaCoeffs` proc-face uses cell-to-cell distance (not just owner-to-face)
- [ ] Proc-face weight calculation in `basicGeometryScheme` validated on non-uniform mesh (graded)

**Validation:**
- [ ] cylinder3D runs to completion with 4 MPI ranks without crash or divergence
- [ ] L∞ error between parallel and serial neoIcoFoam fields (U, p) below 1e-6 at final timestep
- [ ] Test suite covers 4-rank decomposition (currently hard-coded to 3 ranks)
- [ ] At least one HPC-scale run confirms results (ranks TBD by available HPC system)

### Out of Scope

- GPU backends (CUDA, HIP) — CPU/OpenMP is the target for this milestone; GPU is next milestone
- Other solvers (simpleFoam, pisoFoam variants) — neoIcoFoam must work first
- Python bindings (nanobind) — NeoN feature, unrelated to distributed correctness
- PetscSolver distributed — blocked on external PETSc setup; Ginkgo is the target solver
- `processorCyclic` patches — cyclic+parallel cases deferred until simple proc patches work
- Performance optimisation (comm-pattern caching, Ginkgo solver caching) — correctness first

## Context

### Codebase state

The `stack/distributed` branch in NeoFOAM and the `fix/testsRebase` branch in the NeoN submodule
are both active WIP. All distributed work is in flight simultaneously across both repos.

Key known-broken areas (diagnosed from concerns audit 2026-05-10):

**NeoFOAM-side bugs:**
- `readers.hpp:67–109`: fixedValue BC silently dropped to `empty` when serialised token count is 1 in parallel
- `readers.hpp:174`: `processorCyclic` patch names (`procBoundary0to1`) not in patchInserter map → crash/UB
- `pdeSolver.hpp:284–286`: `needReference_`, `pRefCell_`, `pRefValue_` uninitialised → UB if setReference not called
- `neoIcoFoam.cpp:119–122`: `adjustPhi` and `constrainPressure` disabled → mass imbalance on mixed BCs
- `meshAdapter.cpp:194–197`: `computeNeighbRank` called twice, duplicate variable

**NeoN-side bugs (submodule at src/NeoN, branch fix/testsRebase):**
- `faceToMatrixAddress.hpp:127`: CSR matrix offsets incorrect for proc-adjacent cells in distributed mode
- `unstructuredMesh.cpp:491` / `boundaryData.hpp:270`: MPI_Alltoallv symmetric send/recv assumption
- `surfaceField.cpp:49`: `internalVector` proc-face slots not updated after MPI exchange
- `basicGeometryScheme.cpp:215`: `std::abs` in GPU lambda → build failure on CUDA/HIP
- `diagonalSolver.hpp:28–38`: `solveDist` is a no-op silent zero solution
- `linearSystem.hpp:376–377`: RHS non-local subtraction commented out (`// FIXME add`)
- `basicGeometryScheme.cpp:276–295`: deltaCoeffs uses only owner-to-face distance, not cell-to-cell
- `partitioning.hpp:68–112`: FIXME — only works for 3 ranks (hard-coded index arithmetic)

### Co-development workflow

NeoFOAM pins NeoN via a git submodule pointer. During active iteration, set `NEOFOAM_NEON_DIR=./src/NeoN`
in cmake configure to use the live submodule tree without bumping the pointer on every commit.
When pushing NeoFOAM changes, always ensure the submodule pointer is committed and pushed to NeoN's
remote (`github.com/exasim-project/NeoN`) first.

### Test infrastructure

- All distributed tests currently hard-code `REQUIRE(Foam::Pstream::nProcs() == 3)`
- The cylinder3D case uses a 4-rank hierarchical decomposition (`system/decomposeParDict`)
- The weight-symmetry test gap: all current tests use `simpleGrading (1 1 1)` → `w = 0.5`
  which hides weight-reversal bugs

## Constraints

- **Compatibility**: OpenFOAM ≥ 2406 required; `FOAM_SRC` must be set before build
- **Runtime**: NeoN executor must never be hard-coded; always thread through from call site
- **Submodule**: NeoN changes land on `fix/testsRebase` branch in `src/NeoN/`; coordinate commits
- **MPI**: Target 2–4 ranks for development; HPC validation on cluster (ranks TBD)
- **Build preset**: Use `develop` preset for all development (Debug + bounds checks)
- **Solver target**: Ginkgo distributed solver — PETSc is out of scope for this milestone
- **Mesh**: cylinder3D (4-rank hierarchical decomp) is the validation case; cylinder2D is regression

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Fix distributed stack broadly, not just icoFoam | Correctness fixes at the NeoN infrastructure level benefit all future solvers; icoFoam is the validation vehicle | — Pending |
| CPU/OpenMP target for this milestone | GPU correctness adds a separate dimension of complexity; validate parallel logic on CPU first | — Pending |
| Ginkgo distributed solver over PETSc | PETSc requires external setup; Ginkgo is already integrated in NeoN | — Pending |
| 4-rank hierarchical decomp as canonical test | cylinder3D already decomposed to 4 ranks; matches dev target; exercises non-trivial topology | — Pending |
| L∞ < 1e-6 as convergence criterion | Standard CFD tolerance; field-level check catches both solution divergence and boundary artefacts | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-05-10 after initialization*
