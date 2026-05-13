# Requirements: NeoFOAM Distributed

## v1.0 — Parallel neoIcoFoam (CPU) — Complete

All 21 v1 requirements delivered across Phases 1–5 (2026-05-10 through 2026-05-13).
See `.planning/phases/01-*/` through `.planning/phases/05-*/` for plan summaries.

**Closed:** BC-01, BC-02, BC-03, MPI-01, MPI-02, MPI-03, HLTH-01, LSA-01, LSA-02,
NRANK-01, NRANK-02, GEO-01, GEO-02, GEO-03, PISO-02, PISO-03, HLTH-02,
VAL-01, VAL-02, VAL-03, VAL-04

GPU-01 (std::abs in updateWeights kernel) deferred to v2. PERF-01/02/03 deferred to v3+.

---

## v2.0 — GPU Correctness — Active

**Core Value:** `neoIcoFoam` on cylinder3D with 4 MPI ranks using `GPUExecutor` completes
without divergence and produces L∞ < 1e-6 vs CPU serial reference.

**Branches:** `feat/gpu-distributed` (NeoFOAM), `feat/gpu-distributed` (NeoN from `fix/testsRebase`)

**Local testing:** WSL2 single-GPU, host-staged MPI (`UCX_TLS=tcp`), no MPS (not available on WSL2 WDDM)
**HPC testing:** CUDA-aware MPI, multiple physical GPUs, ad-hoc report-back workflow

### GPU MPI Transport (NeoN)

- [x] **GPU-MPI-01**: `communicateBoundaryData` host-stages MPI buffers when CUDA-aware MPI is
  not available — device→host copy before `MPI_Alltoallv`, host→device after. Eliminates crash
  on systems where MPI cannot dereference device pointers (WSL2, any non-CUDA-aware OpenMPI).

- [x] **GPU-MPI-02**: `NeoN_CUDA_AWARE_MPI` CMake option (default OFF). When ON, bypasses
  host-staging and passes device pointers directly to MPI (GPU-direct path for HPC).
  When OFF, always host-stages regardless of executor type.

- [x] **GPU-MPI-03**: The post-MPI `fence(exec)` added in v1 (commit `83a83517` in NeoFOAM)
  is verified correct on the host-staged path — fence is still called after MPI on GPU executor
  to synchronize any in-flight device work before the unpack kernel.

### GPU Kernel Correctness (NeoN + NeoFOAM)

- [ ] **GPU-KRN-01**: All `NEON_LAMBDA` / `NeoN_LAMBDA` device bodies use `Kokkos::abs`
  instead of `std::abs`. Includes the deferred v1 item in `basicGeometryScheme.cpp:215`.

- [ ] **GPU-KRN-02**: All device lambdas use `Kokkos::sqrt` instead of `std::sqrt`.

- [ ] **GPU-KRN-03**: All device lambdas use `Kokkos::max` / `Kokkos::min` instead of
  `std::max` / `std::min`.

- [ ] **GPU-KRN-04**: `cmake --build --preset develop` with CUDA backend enabled exits 0
  with no device-compilation errors from non-GPU-safe math calls.

### Executor Parameterization (NeoN + NeoFOAM)

- [ ] **GPU-EXE-01**: Distributed test suite accepts executor as a test parameter and runs
  on `GPUExecutor` when CUDA is compiled in, in addition to always running on `CPUExecutor`.

- [ ] **GPU-EXE-02**: CI configuration unchanged — CI uses `CPUExecutor` only (no GPU runners).
  GPU test sections are compiled but gated on `KOKKOS_ENABLE_CUDA` / `KOKKOS_ENABLE_HIP`.

### Ginkgo GPU Integration (NeoN)

- [ ] **GPU-GNK-01**: Ginkgo distributed solve runs correctly with `GPUExecutor` — verified on
  a 2-rank test case. `NeoN_GINKGO_HOST_STAGE` setting documented for GPU systems without
  unified addressing.

### Local Multi-Rank GPU Validation (WSL2)

- [ ] **GPU-LOC-01**: `mpirun -np N UCX_TLS=tcp` with `GPUExecutor` and host-staged MPI
  runs the distributed unit test suite on WSL2 single-GPU without CUDA-aware MPI or MPS.
  Tests pass with `N = 2` (minimum for distributed correctness).

- [ ] **GPU-LOC-02**: Local WSL2 `neoIcoFoam` cylinder3D 4-rank `GPUExecutor` run completes
  short case (endTime 5e-3) without crash or NaN in U or p.

### End-to-End HPC Validation

- [ ] **GPU-VAL-01**: `neoIcoFoam` cylinder3D 4-rank `GPUExecutor` run completes all
  timesteps on HPC without crash, NaN, or divergence (using CUDA-aware MPI, multiple GPUs).

- [ ] **GPU-VAL-02**: L∞ error between 4-rank GPU neoIcoFoam and CPU serial neoIcoFoam
  is below 1e-6 at the final timestep.

- [ ] **GPU-VAL-03**: At least one HPC run with CUDA-aware MPI and multiple physical GPUs
  confirms the GPU-direct path produces correct results consistent with the host-staged path.

## Out of Scope (v2.0)

| Feature | Reason |
|---------|--------|
| ROCm / HIP GPU backend | AMD GPU correctness follows same pattern; defer until CUDA path is validated |
| SYCL / Intel GPU | Same rationale |
| Performance optimisation (comm caching, Ginkgo caching) | Correctness first — v3 |
| `processorCyclic` full support | CPU correctness established in v1; BC complexity deferred |
| Other solvers (simpleFoam, pisoFoam) | Infrastructure already correct; port deferred |
| PETSc distributed solver | External setup required; Ginkgo is the target |
| Dynamic load balancing | Not relevant until GPU E2E is stable |
| Python bindings | Unrelated to GPU correctness |
| Non-orthogonal corrections at proc faces | Correctness on orthogonal meshes first |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| GPU-MPI-01 | Phase 1: GPU MPI Transport | Complete |
| GPU-MPI-02 | Phase 1: GPU MPI Transport | Complete |
| GPU-MPI-03 | Phase 1: GPU MPI Transport | Complete |
| GPU-KRN-01 | Phase 2: GPU Kernel Fixes | Pending |
| GPU-KRN-02 | Phase 2: GPU Kernel Fixes | Pending |
| GPU-KRN-03 | Phase 2: GPU Kernel Fixes | Pending |
| GPU-KRN-04 | Phase 2: GPU Kernel Fixes | Pending |
| GPU-EXE-01 | Phase 3: Test Suite Parameterization | Pending |
| GPU-EXE-02 | Phase 3: Test Suite Parameterization | Pending |
| GPU-GNK-01 | Phase 4: Ginkgo GPU Integration | Pending |
| GPU-LOC-01 | Phase 3: Test Suite Parameterization | Pending |
| GPU-LOC-02 | Phase 5: E2E Validation | Pending |
| GPU-VAL-01 | Phase 5: E2E Validation | Pending |
| GPU-VAL-02 | Phase 5: E2E Validation | Pending |
| GPU-VAL-03 | Phase 5: E2E Validation | Pending |

**Coverage:** 15 v2 requirements, all mapped to phases ✓

---
*v1 requirements defined: 2026-05-10*
*v2 requirements defined: 2026-05-13*
