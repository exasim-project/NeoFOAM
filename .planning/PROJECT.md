# NeoFOAM Distributed — GPU Correctness

## What This Is

NeoFOAM is an adapter layer that connects OpenFOAM data structures to the NeoN GPU/CPU compute backend,
replacing standard OpenFOAM solvers with NeoN-accelerated variants.

**v1.0 (complete):** Made `neoIcoFoam` run correctly in parallel (MPI) on CPU — 4 MPI ranks,
cylinder3D, L∞ < 1e-6 vs serial. All 21 v1 requirements delivered across 5 phases.

**v2.0 (active):** Extend the parallel stack to `GPUExecutor` — correct GPU field exchange,
GPU-safe kernel math, test suite parameterized for GPU, and E2E validation on HPC.

## Core Value

neoIcoFoam on cylinder3D with 4 MPI ranks using `GPUExecutor` produces field results within
L∞ < 1e-6 of the CPU serial run.

## Requirements

### Validated (v1.0 complete)

- ✓ Serial neoIcoFoam runs on cylinder2D and cylinder3D
- ✓ NeoN CPU/OpenMP field operations (VolumeField, SurfaceField) — established
- ✓ OpenFOAM mesh bridging (foamMesh / meshAdapter)
- ✓ Field conversion OF ↔ NeoN (volScalarField, volVectorField)
- ✓ fvSchemes / fvSolution compatibility layer
- ✓ PISO algorithm with continuity monitoring and reference cell for arbitrary decomposition
- ✓ Distributed UnstructuredMesh (N-rank, hierarchical, graded mesh)
- ✓ MPI halo exchange with symmetric and asymmetric send/recv counts
- ✓ CSR sparsity pattern correct for proc-adjacent cells
- ✓ deltaCoeffs and interpolation weights correct on graded proc meshes
- ✓ Post-MPI fence for GPU-direct MPI stream sync (commit `83a83517`)
- ✓ 4-rank cylinder3D CPU run without divergence — L∞ < 1e-6 vs serial (HPC pending confirmation)

### Active (v2.0)

**GPU MPI transport (blocks all GPU testing):**
- [ ] `communicateBoundaryData` host-stages MPI buffers when CUDA-aware MPI not available
- [ ] `NeoN_CUDA_AWARE_MPI` cmake flag (default OFF) controls host-staged vs GPU-direct path

**GPU kernel math (blocks GPU build):**
- [ ] All `NEON_LAMBDA` device bodies use `Kokkos::abs/sqrt/max/min` (not `std::`)
- [ ] GPU build (CUDA backend) exits 0 with no device-compilation errors

**Test parameterization (validates GPU correctness locally):**
- [ ] Distributed tests accept `GPUExecutor` as parameter; run on GPU when CUDA compiled
- [ ] Local WSL2 multi-rank GPU test suite passes (`UCX_TLS=tcp`, host-staged MPI)

**Ginkgo GPU:**
- [ ] Ginkgo distributed solve verified correct with `GPUExecutor` (2-rank test)

**Validation:**
- [ ] cylinder3D 4-rank `GPUExecutor` run without crash or divergence (WSL2 host-staged + HPC GPU-direct)
- [ ] L∞ error GPU 4-rank vs CPU serial < 1e-6

### Out of Scope

- ROCm/HIP, SYCL/Intel GPU — AMD/Intel GPU correctness follows CUDA; deferred to v3
- Other solvers (simpleFoam, pisoFoam variants) — infrastructure correct; port deferred
- Python bindings (nanobind) — NeoN feature, unrelated to GPU correctness
- PetscSolver distributed — blocked on external PETSc setup; Ginkgo is the target solver
- `processorCyclic` patches — deferred; simple proc patches stable in v1
- Performance optimisation (comm-pattern caching, Ginkgo solver caching) — correctness first
- Non-orthogonal corrections at proc faces — orthogonal meshes first

## Context

### Codebase state (v2.0 start)

**NeoFOAM branch:** `feat/gpu-distributed` (from `feat/distributed-correctness`)
**NeoN branch:** `feat/gpu-distributed` (from `fix/testsRebase`)

v1.0 fixed all CPU-distributed bugs. Known remaining issues entering v2.0:

**GPU-specific (NeoN, `feat/gpu-distributed`):**
- `boundaryData.hpp`: `MPI_Alltoallv` passes raw device pointers → crash without CUDA-aware MPI
- `basicGeometryScheme.cpp:215`: `std::abs` in `NEON_LAMBDA` → CUDA build failure (deferred GPU-01)
- Other device lambdas may use `std::sqrt`/`std::max`/`std::min` — full audit needed
- Ginkgo GPU executor interaction not yet tested in distributed mode

**Local test environment (WSL2):**
- No CUDA-aware MPI (OpenMPI 4.1.6 built without CUDA support)
- No MPS (WSL2 WDDM `/dev/dxg` only — no `/dev/nvidiactl`)
- Multi-rank GPU testing: `mpirun -np N UCX_TLS=tcp` with host-staged MPI buffers
- Single physical GPU (RTX 3070, 8GB), CUDA 13.2, driver 596.21

### Co-development workflow

NeoFOAM pins NeoN via a git submodule pointer. During active iteration, set `NEOFOAM_NEON_DIR=./src/NeoN`
in cmake configure to use the live submodule tree without bumping the pointer on every commit.
When pushing NeoFOAM changes, always ensure the submodule pointer is committed and pushed to NeoN's
remote (`github.com/exasim-project/NeoN`) first.

### Test infrastructure

- Distributed tests use N-rank generalized partitioning (fixed in v1, Phase 2)
- Graded-mesh test case added in v1 Phase 3 (`test/setup_pressureVelocityCoupling_graded/`)
- cylinder3D 4-rank hierarchical decomp in `tutorials/cylinder3D/` (reference fields in `tutorials/cylinder3D_ref/`)
- Local GPU testing: `mpirun -np N UCX_TLS=tcp ./test` with `GPUExecutor`, host-staged MPI
- HPC GPU testing: ad-hoc run + report back workflow

## Constraints

- **Compatibility**: OpenFOAM ≥ 2406 required; `FOAM_SRC` must be set before build
- **Runtime**: NeoN executor must never be hard-coded; always thread through from call site
- **Submodule**: NeoN changes land on `feat/gpu-distributed` branch in `src/NeoN/`; NeoFOAM on `feat/gpu-distributed`
- **MPI local**: Host-staged MPI (`UCX_TLS=tcp`) for WSL2 GPU runs — no CUDA-aware MPI locally
- **MPI HPC**: CUDA-aware MPI available on HPC — `NeoN_CUDA_AWARE_MPI=ON` for HPC builds
- **Build preset**: Use `develop` preset for all development (Debug + bounds checks)
- **Build concurrency**: `cmake --build --preset develop -- -j4` (never exceed -j4)
- **Solver target**: Ginkgo distributed solver — PETSc is out of scope
- **Mesh**: cylinder3D (4-rank hierarchical decomp) is the validation case; cylinder2D is regression
- **GPU local**: WSL2 single RTX 3070, `/dev/dxg` WDDM mode, no MPS — multi-process CUDA via time-slicing

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Fix distributed stack broadly, not just icoFoam | Infrastructure fixes benefit all future solvers | ✓ Delivered in v1 |
| CPU/OpenMP first, GPU second | GPU adds separate failure dimension; validate CPU logic first | ✓ v1 CPU done; v2 GPU |
| Ginkgo distributed solver over PETSc | PETSc requires external setup; Ginkgo already integrated | ✓ Working in v1 |
| 4-rank hierarchical decomp as canonical test | cylinder3D already at 4 ranks; exercises real topology | ✓ Used throughout |
| L∞ < 1e-6 as convergence criterion | Field-level check catches both divergence and boundary artefacts | ✓ v1 gate; continued in v2 |
| Host-staged MPI for GPU (default OFF for GPU-direct) | WSL2 has no CUDA-aware MPI; host-staging enables local GPU testing without MPS | v2 — decided 2026-05-13 |
| No MPS on WSL2 | WSL2 WDDM (`/dev/dxg`) — `nvidia-cuda-mps-control` requires `/dev/nvidiactl` which is absent | v2 — confirmed 2026-05-13 |
| UCX_TLS=tcp for local GPU MPI runs | Avoids cuda_ipc IPC handle collisions; works with host-staged buffers | v2 — decided 2026-05-13 |

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
*v1.0 initialized: 2026-05-10 | v2.0 initialized: 2026-05-13*
