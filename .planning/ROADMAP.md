# Roadmap: NeoFOAM Distributed — Parallel neoIcoFoam

**Created:** 2026-05-10
**Milestone:** Correct parallel neoIcoFoam on cylinder3D (4 MPI ranks)

## Overview

This is a correctness fix campaign across two repositories (NeoFOAM adapter and NeoN submodule). The goal is neoIcoFoam running correctly on cylinder3D with 4 MPI ranks, producing L∞ field error < 1e-6 vs. the serial run. 21 confirmed defects are fixed in strict dependency order: MPI transport and BC parsing first (everything else reads from these), then linear system assembly, then geometry operators, then PISO loop wiring, then end-to-end validation.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: MPI & BC Infrastructure** - Fix BC parsing and MPI halo exchange so the solver can start without UB or data corruption (completed 2026-05-11)
- [ ] **Phase 2: Linear System Correctness** - Correct CSR sparsity pattern, RHS assembly, and N-rank generalisation so the distributed linear system is well-formed
- [x] **Phase 3: Geometry & Operator Correctness** - Fix deltaCoeffs and proc-face weights, add graded-mesh tests to expose geometry bugs (completed 2026-05-12)
- [ ] **Phase 4: PISO Loop Completeness** - Wire reference cell, continuity monitoring, and matrix guard into neoIcoFoam for a complete, diagnosable PISO loop
- [ ] **Phase 5: End-to-End Validation** - Run cylinder3D 4-rank vs. serial and confirm L∞ < 1e-6; extend test suite to 4 ranks; run at HPC scale

## Phase Details

### Phase 1: MPI & BC Infrastructure
**Goal**: The solver can be launched in parallel without undefined behaviour, crash, or data corruption at processor boundaries
**Depends on**: Nothing (first phase)
**Requirements**: BC-01, BC-02, BC-03, MPI-01, MPI-02, MPI-03, HLTH-01
**Success Criteria** (what must be TRUE):
  1. A `fixedValue` boundary condition with a single-token serialised value is correctly parsed on all ranks — no silent downgrade to `empty`
  2. Processor-cyclic patch names (`procBoundary0to1` etc.) are accepted by `readSurfaceBoundaryConditions` without crash or abort
  3. `PDESolver` can be constructed and `solve()` called without triggering undefined behaviour from uninitialised members
  4. After a halo exchange, proc-face slots in `SurfaceField::internalVector()` contain the received ghost values (not stale zeros)
  5. `MPI_Alltoallv` uses independent `sendCounts` and `recvCounts` arrays — a hierarchical 4-rank decomposition exchanges the correct number of values per neighbour
**Plans**: 3 plans in 2 waves

**Wave 1** *(parallel — no shared files)*
- [x] 01-01-PLAN.md — NeoFOAM-side fixes: BC-01 fixedValue size-1 fallback, BC-02 processorCyclic crash, BC-03 PDESolver uninitialised members, HLTH-01 foamMesh.cpp dead code
- [x] 01-02-PLAN.md — NeoN-side MPI fixes: MPI-01 symmetric Alltoallv in unstructuredMesh.cpp and boundaryData.hpp, MPI-02 stale internalVector proc-face slots, MPI-03 recvBuffer over-allocation

**Wave 2** *(blocked on Wave 1 completion)*
- [x] 01-03-PLAN.md — Build gate + test additions: single `cmake --build --preset develop -- -j4`, new TEST_CASEs in both distributed test files, full CTest suite green

**Cross-cutting constraints:** `cmake --build --preset develop -- -j4` (never exceed -j4); NeoN edits on `fix/testsRebase`; submodule pointer not bumped until phase end; two-pass boundary iteration invariant preserved; new tests use 3-rank assumption

### Phase 2: Linear System Correctness
**Goal**: The distributed linear system assembled by NeoN has the correct sparsity pattern and right-hand side for proc-adjacent cells, and can be built for any rank count
**Depends on**: Phase 1
**Requirements**: LSA-01, LSA-02, NRANK-01, NRANK-02
**Success Criteria** (what must be TRUE):
  1. `faceToMatrixAddress` returns correct `diagIdx`, `upperIdx`, and `lowerIdx` flat-index values for cells that own proc-boundary faces — a known 2-rank A*x==rhs check passes
  2. `removeBoundaryContributions` subtracts non-local RHS contributions at proc-adjacent rows — residual norm after one Ginkgo solve matches the serial baseline
  3. Distributed test binaries accept the actual MPI rank count passed at runtime — tests pass for 2, 3, 4, and 8 ranks without `REQUIRE(nProcs() == 3)` guards
  4. `partitioning.hpp` constructs a valid test mesh for 2, 3, 4, and 8 ranks without any hard-coded 3-rank index arithmetic
**Plans**: 4 plans in 2 waves

**Wave 1** *(parallel — no shared files)*
- [ ] 02-01-PLAN.md — LSA-01 spike + investigation: add Ginkgo Laplacian residual TEST_CASE tagged [LSA-01]; deep-read faceToMatrixAddress.cpp with file:line precision
- [ ] 02-02-PLAN.md — NRANK-01 partitioning generalisation: closed-form N-rank formula for partitionSurfaceField; preserve 3-rank oracle

**Wave 2** *(blocked on plan 02-01 for the test file and the investigation doc)*
- [ ] 02-03-PLAN.md — LSA-01 fix + LSA-02 fix: correct CSR offsets per investigation; implement non-local RHS subtraction in removeBoundaryContributions
- [ ] 02-04-PLAN.md — NRANK-02 guard removal + CTest multi-rank: remove all 7 nProcs==3 guards; register 2/4-rank CTest entries (3-rank kept; 8-rank deferred to Phase 5)

### Phase 3: Geometry & Operator Correctness
**Goal**: Proc-face deltaCoeffs and interpolation weights are geometrically correct, and the test suite can detect weight-reversal bugs on non-uniform meshes
**Depends on**: Phase 2
**Requirements**: GEO-01, GEO-02, GEO-03
**Success Criteria** (what must be TRUE):
  1. `deltaCoeffs` at processor faces equals `1 / (d_own + d_nei)` (cell-to-cell distance) — verified by comparing Laplacian operator result against analytical value on a 2-rank mesh
  2. A graded-mesh test case (`simpleGrading 2 1 1`) is added to the distributed test suite and passes with correct proc-face weights
  3. Proc-face interpolation weights on a non-uniform mesh are symmetric between owner and neighbour ranks — weight reversal (w > 0.5 on both sides) is detected and fails the test
**Plans**: 3 plans in 2 waves

**Wave 1** *(parallel — no shared files)*
- [x] 03-01-PLAN.md — GEO-03: create test/setup_pressureVelocityCoupling_graded/ with simpleGrading (2 1 1); regenerate constant/polyMesh, processor0/, processor1/ via blockMesh + decomposePar; commit all generated mesh files (test runner reads them directly)
- [x] 03-02-PLAN.md — GEO-01: fix updateDeltaCoeffs proc-face block in src/NeoN/.../basicGeometryScheme.cpp to use 1/(d_own + d_nei) by mirroring the existing updateNonOrthDeltaCoeffs pattern; remove [[maybe_unused]] on the function parameters; build green with cmake --build --preset develop -- -j4

**Wave 2** *(blocked on Plans 03-01 and 03-02)*
- [x] 03-03-PLAN.md — Add [GEO-01] and [GEO-02] TEST_CASEs to test/test_distributedUnstructuredMesh.cpp (MPI_Sendrecv tags 43 and 44; SECTION_IF nProcs==2 guards); register distributedUnstructuredMesh_graded CTest entry in test/CMakeLists.txt; full ctest --preset develop suite green

### Phase 4: PISO Loop Completeness
**Goal**: neoIcoFoam's PISO loop handles arbitrary rank decompositions, reports continuity errors for divergence monitoring, and guards against premature matrix use
**Depends on**: Phase 3
**Requirements**: PISO-02, PISO-03, HLTH-02
**Success Criteria** (what must be TRUE):
  1. The pressure reference cell is mapped to the correct MPI rank and local cell index for an arbitrary decomposition — the pressure system is non-singular when the reference cell is not on rank 0
  2. Continuity error is printed each PISO iteration — diverging runs report increasing continuity error before any NaN propagation
  3. `PDESolver::computeRAU` throws or auto-assembles when called before `assemble()` — the unassembled-matrix programming error is caught at runtime rather than producing a silent wrong answer
**Plans**: 3 plans in 2 waves

**Wave 1** *(parallel — no shared files)*
- [ ] 04-01-PLAN.md — HLTH-02 + PISO-02: add isAssembled_ guard to PDESolver; remove rank-0 guards from setReference()/SetReference::operator(); add runtime_error guards to computeRAU/computeRAUandHByA
- [ ] 04-02-PLAN.md — PISO-03: inline continuity error reporting in neoIcoFoam.cpp using NeoN-native div(phi) + MPI allReduce; matches icoFoam log format

**Wave 2** *(blocked on Wave 1 completion)*
- [ ] 04-03-PLAN.md — Build gate + [PISO-02] TEST_CASE in test_distributedPressureVelocityCoupling.cpp (rank-1 reference cell, pRefCell >= 0 gate); full ctest --preset develop suite green

### Phase 5: End-to-End Validation
**Goal**: neoIcoFoam on cylinder3D with 4 MPI ranks completes without crash, divergence, or field error exceeding L∞ < 1e-6 vs. the serial run, and results are confirmed at HPC scale
**Depends on**: Phase 4
**Requirements**: VAL-01, VAL-02, VAL-03, VAL-04
**Success Criteria** (what must be TRUE):
  1. cylinder3D runs to completion (all time steps) with 4 MPI ranks using the hierarchical decomposition — no process crash, no solver divergence, no NaN in U or p
  2. L∞ error between 4-rank parallel and serial neoIcoFoam fields U and p is below 1e-6 at the final timestep
  3. Distributed test suite covers 4-rank decomposition for mesh, momentum, and pressure-velocity coupling — all tests pass at 4 ranks
  4. At least one HPC-scale run (rank count determined by available hardware) completes without crash and produces results consistent with the 4-rank desktop run
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. MPI & BC Infrastructure | 3/3 | Complete   | 2026-05-11 |
| 2. Linear System Correctness | 0/TBD | Not started | - |
| 3. Geometry & Operator Correctness | 3/3 | Complete   | 2026-05-12 |
| 4. PISO Loop Completeness | 0/3 | Not started | - |
| 5. End-to-End Validation | 0/TBD | Not started | - |
