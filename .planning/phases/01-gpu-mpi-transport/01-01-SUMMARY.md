---
phase: 01-gpu-mpi-transport
plan: "01"
subsystem: NeoN/distributed/MPI
tags: [gpu, mpi, host-staging, boundaryData, cmake]
dependency_graph:
  requires: []
  provides: [NEON_CUDA_AWARE_MPI compile flag, host-staged MPI path in communicateBoundaryData]
  affects: [src/NeoN/include/NeoN/fields/boundaryData.hpp, src/NeoN/CMakeLists.txt, src/NeoN/include/CMakeLists.txt]
tech_stack:
  added: [NeoN_CUDA_AWARE_MPI cmake option]
  patterns: [#if defined(NEON_CUDA_AWARE_MPI) preprocessor gate, copyToHost() D→H staging, Vector(exec, hostVec) H→D copy]
key_files:
  created: []
  modified:
    - src/NeoN/CMakeLists.txt
    - src/NeoN/include/CMakeLists.txt
    - src/NeoN/include/NeoN/fields/boundaryData.hpp
decisions:
  - "NeoN_CUDA_AWARE_MPI=OFF by default — safe for WSL2/OpenMPI without CUDA-aware MPI"
  - "Vec3 overload stages sendBuffer (not boundaryData) — pack kernel runs before fence, sendBuffer is the already-packed scalar array"
  - "recvBuffer declared inside both #if and #else with same name — downstream inV = recvBuffer.view() compiles on both paths"
metrics:
  duration: "49 minutes"
  completed: "2026-05-13"
  tasks_completed: 3
  files_modified: 3
---

# Phase 01 Plan 01: GPU MPI Transport Host-Staging Summary

**One-liner:** Add NeoN_CUDA_AWARE_MPI cmake flag (default OFF) and host-staged MPI path in both communicateBoundaryData overloads using copyToHost()/Vector(exec, host) primitives.

## What Was Built

Without this fix, `communicateBoundaryData` in `boundaryData.hpp` passed raw device pointers to
`MPI_Alltoallv`. On WSL2 with standard OpenMPI (no CUDA-aware MPI), this causes a SIGSEGV or
silent wrong-answer because MPI cannot dereference device memory.

This plan adds a compile-time flag `NeoN_CUDA_AWARE_MPI` (default OFF) that gates two code
paths in both overloads of `communicateBoundaryData`:

- `#if defined(NEON_CUDA_AWARE_MPI)`: original code — device pointers passed directly to MPI (GPU-direct, for HPC clusters)
- `#else`: new host-staged path — `copyToHost()` D→H, MPI with host pointers, `Vector(exec, recvHost)` H→D before unpack kernel

## Tasks Completed

### Task 1: Add NeoN_CUDA_AWARE_MPI cmake flag

- `src/NeoN/CMakeLists.txt`: inserted `option(NeoN_CUDA_AWARE_MPI ... OFF)` after `NeoN_GINKGO_HOST_STAGE` option
- `src/NeoN/include/CMakeLists.txt`: added `if(NeoN_CUDA_AWARE_MPI) target_compile_definitions(... NEON_CUDA_AWARE_MPI=1) endif()` inside the `if(NeoN_WITH_MPI)` block
- Commit: `b58afc284` (on NeoN `feat/gpu-distributed`)

### Task 2: Implement host-staging in both overloads

- **Scalar overload**: wrapped `MPI_Alltoallv` block with `#if NEON_CUDA_AWARE_MPI / #else / #endif`; `#else` path uses `boundaryData.copyToHost()` → `SerialExecutor {}` recv buffer → `Vector(exec, recvHost)` H→D copy
- **Vec3 specialization**: same gate around `MPI_Alltoallv`; stages `sendBuffer` (already packed scalar array), NOT `boundaryData` directly
- All 6 fence() calls preserved (pre-MPI, post-MPI, post-unpack × 2 overloads)
- Commit: `2731cc8ed` (on NeoN `feat/gpu-distributed`)

### Task 3: Build verification

- `cmake --build --preset develop -- -j4` exits 0 with `NeoN_CUDA_AWARE_MPI=OFF` (default)
- `cmake --preset develop -DNeoN_CUDA_AWARE_MPI=ON && cmake --build -- -j4` exits 0 (GPU-direct path)
- Restored `NeoN_CUDA_AWARE_MPI=OFF` in CMakeCache
- Commit (NeoFOAM submodule pointer bump): `b19352f2` (on `feat/gpu-distributed`)

## Verification

### cmake flags

```
grep -c "option(NeoN_CUDA_AWARE_MPI" src/NeoN/CMakeLists.txt
→ 1 ✓

grep -c "NEON_CUDA_AWARE_MPI=1" src/NeoN/include/CMakeLists.txt
→ 1 ✓
```

### Host-staging structure in boundaryData.hpp

```
grep -c "#if defined(NEON_CUDA_AWARE_MPI)" → 2 ✓ (one per overload)
grep -c "copyToHost()" → 2 ✓ (one per overload)
grep -c "SerialExecutor {}" for MPI → 2 ✓ (recvHost construction in each overload)
grep -c "sendBuffer.data()" → 1 ✓ (inside #if GPU-direct path only)
grep -c "boundaryData.data()" → 1 ✓ (inside #if GPU-direct path only)
```

### Fence call preservation (6 total)

```
Line 280: fence(boundaryData.exec())   — scalar pre-MPI
Line 318: fence(exec)                  — scalar post-MPI
Line 353: fence(exec)                  — scalar post-unpack
Line 417: fence(exec)                  — Vec3 pre-MPI (fence(exec) — exec = boundaryData.exec() assigned at L398)
Line 455: fence(exec)                  — Vec3 post-MPI
Line 490: fence(exec)                  — Vec3 post-unpack
Total: 6 fence calls ✓
```

Note: `grep -c "fence("` returns 8 (includes 2 comment lines mentioning fence). Actual fence() call sites = 6.

### Build exit codes

- `NeoN_CUDA_AWARE_MPI=OFF` build: exit 0 ✓
- `NeoN_CUDA_AWARE_MPI=ON` build: exit 0 ✓

## Deviations from Plan

### Auto-fixed: clang-format reformatting

- **Found during:** Task 2 commit
- **Issue:** clang-format pre-commit hook reformatted `boundaryData.hpp` (spacing around `SerialExecutor {}` and comment alignment)
- **Fix:** Re-staged the reformatted file and committed successfully on second attempt
- **Impact:** No semantic changes; formatting compliant with project `.clang-format` config

### Plan acceptance criteria note: fence count discrepancy

The plan stated `grep -c "fence(boundaryData.exec())"` returns 2. The original Vec3 specialization uses `fence(exec)` (not `fence(boundaryData.exec())`) for its pre-MPI fence because `exec` is assigned from `boundaryData.exec()` at line 398. This is correct behavior — the acceptance criterion in the plan had an incorrect expected count. The actual 6-fence-call invariant is satisfied.

## Threat Mitigations Applied

| Threat | Mitigation Applied |
|--------|--------------------|
| T-01-01: recvBuffer executor mismatch | `recvBuffer = Vector<T>(boundaryData.exec(), recvHost)` — device executor; Kokkos bounds-check guards confirm correct executor |
| T-01-02: stale device data sent to MPI | `fence(boundaryData.exec())` preserved before both `#if` and `#else` paths |
| T-01-03: CUDA-aware path build failure | GPU-direct path is exact original code; verified builds with =ON exit 0 |
| T-01-04: Vec3 stages wrong buffer | `sendBuffer.copyToHost()` used (not `boundaryData.copyToHost()`) in Vec3 `#else` path |

## Git Commits

| Task | Repo | Commit | Description |
|------|------|--------|-------------|
| 1 | NeoN (src/NeoN) | `b58afc284` | feat(01-01): add NeoN_CUDA_AWARE_MPI cmake flag (default OFF) |
| 2 | NeoN (src/NeoN) | `2731cc8ed` | feat(01-01): host-stage MPI buffers in both communicateBoundaryData overloads |
| 3 | NeoFOAM | `b19352f2` | chore(01-01): bump NeoN submodule to feat/gpu-distributed (host-stage MPI) |

## Self-Check: PASSED

- [x] `src/NeoN/CMakeLists.txt` modified with `option(NeoN_CUDA_AWARE_MPI)` — verified
- [x] `src/NeoN/include/CMakeLists.txt` modified with `NEON_CUDA_AWARE_MPI=1` compile def — verified
- [x] `src/NeoN/include/NeoN/fields/boundaryData.hpp` modified with both host-staged overloads — verified
- [x] NeoN commit `b58afc284` exists on `feat/gpu-distributed`
- [x] NeoN commit `2731cc8ed` exists on `feat/gpu-distributed`
- [x] NeoFOAM commit `b19352f2` exists on `feat/gpu-distributed`
- [x] Both cmake builds (OFF and ON) exit 0
- [x] CMakeCache restored to `NeoN_CUDA_AWARE_MPI:BOOL=OFF`
