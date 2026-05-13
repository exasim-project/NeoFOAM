# Roadmap: NeoFOAM Distributed — GPU Correctness

**Created:** 2026-05-13
**Milestone:** v2.0 — Correct GPU distributed neoIcoFoam on cylinder3D (4 MPI ranks, GPUExecutor)

## Overview

Build on the v1.0 CPU-correct distributed stack and extend it to work with `GPUExecutor`.
The core challenge: GPU memory is not accessible by non-CUDA-aware MPI, and device-kernel math
calls require Kokkos wrappers. Fix both in dependency order — transport first (makes multi-rank
GPU testable), then kernels (makes compute correct), then test parameterization (validates both),
then Ginkgo GPU (enables full solver), then E2E (confirms milestone goal).

Local development uses WSL2 single-GPU with host-staged MPI (`UCX_TLS=tcp`, no MPS).
HPC validation uses CUDA-aware MPI with multiple physical GPUs (ad-hoc run + report back).

**NeoFOAM branch:** `feat/gpu-distributed` (from `feat/distributed-correctness`)
**NeoN branch:** `feat/gpu-distributed` (from `fix/testsRebase`)

## Phases

- [ ] **Phase 1: GPU MPI Transport** — Host-stage MPI buffers so GPUExecutor field exchange works without CUDA-aware MPI
- [ ] **Phase 2: GPU Kernel Fixes** — Replace all `std::` math in device lambdas with `Kokkos::` equivalents; GPU build clean
- [ ] **Phase 3: Test Suite Parameterization** — Distributed tests accept GPUExecutor; local WSL2 multi-rank GPU passes
- [ ] **Phase 4: Ginkgo GPU Integration** — Ginkgo distributed solve verified correct with GPUExecutor
- [ ] **Phase 5: E2E Validation** — cylinder3D 4-rank GPUExecutor completes; L∞ < 1e-6 vs CPU serial on HPC

## Phase Details

### Phase 1: GPU MPI Transport
**Goal**: `communicateBoundaryData` works with `GPUExecutor` on any system — crash-free on non-CUDA-aware MPI (WSL2) and GPU-direct-capable on HPC with CUDA-aware MPI
**Depends on**: Nothing (first phase)
**Requirements**: GPU-MPI-01, GPU-MPI-02, GPU-MPI-03
**Success Criteria** (what must be TRUE):
  1. `communicateBoundaryData` with `GPUExecutor` and `UCX_TLS=tcp` (no CUDA-aware MPI) completes without crash or wrong answer on WSL2
  2. A 2-rank `GPUExecutor` smoke test (send/recv of proc-boundary scalar field) passes locally
  3. `NeoN_CUDA_AWARE_MPI=OFF` is the cmake default; `=ON` builds and restores the GPU-direct path (device pointers to MPI)
  4. `fence(exec)` is still called after MPI on the host-staged path (device sync before unpack kernel)
**Plans**: 2 plans in 2 waves

**Wave 1** *(no deps)*
- [x] 01-01-PLAN.md — Host-stage `communicateBoundaryData`: add `NeoN_CUDA_AWARE_MPI` cmake flag (default OFF) to `src/NeoN/CMakeLists.txt` + `src/NeoN/include/CMakeLists.txt`; wrap `MPI_Alltoallv` in both scalar and Vec3 overloads of `boundaryData.hpp` with `#if defined(NEON_CUDA_AWARE_MPI)` / `#else` host-staging / `#endif`; all six `fence(exec)` calls preserved; build green on both flag states

**Wave 2** *(blocked on 01-01 building clean)*
- [ ] 01-02-PLAN.md — WSL2 smoke test: run `mpirun -np 3 UCX_TLS=tcp ./build/develop/bin/tests/neon_test_partitioning` with GPUExecutor (via `allAvailableExecutor()` on CUDA build); confirm proc-boundary scalar and Vec3 values match expected ghost-cell values on GPU path; `ctest --preset develop` CPU suite green

**Cross-cutting constraints:** NeoN edits on `feat/gpu-distributed`; `cmake --build --preset develop -- -j4` (never exceed -j4); `UCX_TLS=tcp` for all local GPU MPI runs on WSL2

---

### Phase 2: GPU Kernel Fixes
**Goal**: Every `NEON_LAMBDA` / device lambda in NeoN and NeoFOAM uses `Kokkos::` math — build with CUDA backend exits clean
**Depends on**: Phase 1 (branches exist, build infrastructure confirmed)
**Requirements**: GPU-KRN-01, GPU-KRN-02, GPU-KRN-03, GPU-KRN-04
**Success Criteria** (what must be TRUE):
  1. `grep -rn "std::abs\|std::sqrt\|std::max\|std::min" src/NeoN/ include/NeoFOAM/` inside any `NEON_LAMBDA` / `NeoN_LAMBDA` body returns zero matches
  2. `cmake --build --preset develop` with CUDA backend exits 0 — no device-compilation warnings from non-GPU-safe math
  3. CTest suite remains green on CPU after the replacements
**Plans**: 2 plans in 1 wave

**Wave 1** *(parallel — different file sets)*
- [ ] 02-01-PLAN.md — Audit + fix NeoN device lambdas: `grep -rn "NEON_LAMBDA\|NeoN_LAMBDA" src/NeoN/` → collect all lambda bodies → replace `std::` math with `Kokkos::` equivalents; build green; ctest CPU green
- [ ] 02-02-PLAN.md — Audit + fix NeoFOAM device lambdas (if any in `src/` or `include/NeoFOAM/`); build green; full ctest green

**Cross-cutting constraints:** Replacements only inside `NEON_LAMBDA`/`NeoN_LAMBDA` bodies — host-side code may legitimately use `std::` math

---

### Phase 3: Test Suite Parameterization
**Goal**: Distributed tests run on `GPUExecutor` in addition to `CPUExecutor`; local WSL2 multi-rank GPU test suite passes
**Depends on**: Phase 1 (host-staged MPI working), Phase 2 (GPU build clean)
**Requirements**: GPU-EXE-01, GPU-EXE-02, GPU-LOC-01
**Success Criteria** (what must be TRUE):
  1. `test_distributedPressureVelocityCoupling`, `test_distributedMomentum`, `test_distributedUnstructuredMesh` all accept executor as a parameter and run their GPU sections when `KOKKOS_ENABLE_CUDA` is set
  2. `mpirun -np 4 UCX_TLS=tcp ./neofoam_test_distributedPressureVelocityCoupling` passes on WSL2 with `GPUExecutor`
  3. `ctest --preset develop` exits 0 — CPU sections still pass; GPU sections pass on GPU-enabled builds
**Plans**: 2 plans in 2 waves

**Wave 1** *(no deps within phase)*
- [ ] 03-01-PLAN.md — Parameterize executor in distributed tests: add `GPUExecutor` to the GENERATE list (guarded by `#ifdef KOKKOS_ENABLE_CUDA`); CTest entries for GPU variants added; CPU ctest still green

**Wave 2** *(blocked on 03-01)*
- [ ] 03-02-PLAN.md — Local GPU validation: run full distributed test suite with `GPUExecutor` on WSL2 (`mpirun -np 4 UCX_TLS=tcp`); diagnose any GPU-specific failures; all tests pass

**Cross-cutting constraints:** CI `ctest` uses CPU only — GPU tests are compiled but run only on GPU hardware; `UCX_TLS=tcp` required for all local GPU MPI runs

---

### Phase 4: Ginkgo GPU Integration
**Goal**: Ginkgo distributed linear solve runs correctly with `GPUExecutor` — the full solver pipeline (assemble → exchange → solve) works on GPU
**Depends on**: Phase 3 (test parameterization in place for verification)
**Requirements**: GPU-GNK-01
**Success Criteria** (what must be TRUE):
  1. A 2-rank Ginkgo distributed solve test (existing Laplacian or new minimal case) passes with `GPUExecutor`
  2. `NeoN_GINKGO_HOST_STAGE` setting is documented in a test comment — default value for GPU correctness vs performance trade-off explained
  3. CTest GPU suite remains green after Ginkgo GPU fix
**Plans**: 1 plan in 1 wave

**Wave 1** *(no deps within phase)*
- [ ] 04-01-PLAN.md — Test Ginkgo with `GPUExecutor` on 2-rank case; if it fails, diagnose (likely needs `NeoN_GINKGO_HOST_STAGE=ON` or explicit Ginkgo CUDA executor threading); apply fix; test passes; document the setting

---

### Phase 5: E2E Validation
**Goal**: `neoIcoFoam` cylinder3D 4-rank `GPUExecutor` completes without crash and produces L∞ < 1e-6 vs CPU serial; HPC confirms GPU-direct path
**Depends on**: Phases 1–4
**Requirements**: GPU-LOC-02, GPU-VAL-01, GPU-VAL-02, GPU-VAL-03
**Success Criteria** (what must be TRUE):
  1. Local WSL2 4-rank `GPUExecutor` cylinder3D run (endTime 5e-3, host-staged MPI) completes without crash or NaN — field values look physically reasonable
  2. HPC 4-rank `GPUExecutor` run with CUDA-aware MPI completes all timesteps without crash
  3. L∞ error between GPU 4-rank and CPU serial neoIcoFoam is below 1e-6 at the final timestep
  4. GPU-direct HPC run result is consistent with local host-staged result (same L∞ order)
**Plans**: 2 plans in 2 waves

**Wave 1** *(no deps within phase)*
- [ ] 05-01-PLAN.md — Local WSL2 E2E: run `mpirun -np 4 UCX_TLS=tcp ./neoIcoFoam -parallel` with `GPUExecutor` on cylinder3D (endTime 5e-3); compare vs CPU serial reference (from `tutorials/cylinder3D_ref/`); confirm no crash/NaN; record L∞ norms

**Wave 2** *(blocked on 05-01 passing)*
- [ ] 05-02-PLAN.md — HPC submission: 4-rank `GPUExecutor` run with CUDA-aware MPI; user reports output; record L∞; declare milestone pass/fail

**Cross-cutting constraints:** `compare_fields.py --threshold 1e-6` is the final gate; HPC reporting is ad-hoc (user pastes output); host-staged and GPU-direct paths both tested

---

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. GPU MPI Transport | 1/2 | In Progress|  |
| 2. GPU Kernel Fixes | 0/2 | Not started | — |
| 3. Test Suite Parameterization | 0/2 | Not started | — |
| 4. Ginkgo GPU Integration | 0/1 | Not started | — |
| 5. E2E Validation | 0/2 | Not started | — |

**Execution Order:** 1 → 2 → 3 → 4 → 5 (phases sequential; plans within a wave parallel)
