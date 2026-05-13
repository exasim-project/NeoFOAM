---
gsd_state_version: 1.0
milestone: v2.0
milestone_name: GPU Correctness — Distributed neoIcoFoam
status: executing
stopped_at: Phase 1 not yet planned — ready to plan
last_updated: "2026-05-13T10:34:38.476Z"
last_activity: 2026-05-13
progress:
  total_phases: 6
  completed_phases: 5
  total_plans: 19
  completed_plans: 18
  percent: 95
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-05-13)

**Core value:** neoIcoFoam on cylinder3D with 4 MPI ranks using GPUExecutor produces field results within L∞ < 1e-6 of the CPU serial run.
**Current focus:** Phase 01 — gpu-mpi-transport

## v1.0 Summary (Complete)

All 5 phases delivered across 2026-05-10 through 2026-05-13:

- Phase 1: MPI & BC Infrastructure ✓
- Phase 2: Linear System Correctness ✓
- Phase 3: Geometry & Operator Correctness ✓
- Phase 4: PISO Loop Completeness ✓
- Phase 5: End-to-End Validation ✓ (GPU fence fix committed; HPC CPU E2E pending final report)

Key v1 commits carried into v2 base:

- `83a83517` fix(gpu-mpi): post-MPI device fence (NeoFOAM `feat/distributed-correctness`)
- `693219896` Added fences() (NeoN `fix/testsRebase`)

## Current Position

Phase: 01 (gpu-mpi-transport) — EXECUTING
Plan: 2 of 2
Status: Ready to execute
Last activity: 2026-05-13

Progress: [██████████] 95%

## Performance Metrics

**Velocity:**

- Total plans completed: 0
- Average duration: -
- Total execution time: 0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01 | - | - | - |
| 02 | - | - | - |
| 03 | - | - | - |
| 04 | - | - | - |
| 05 | - | - | - |
| Phase 01 P01 | 49 | 3 tasks | 3 files |

## Accumulated Context

### Decisions

- GPU MPI transport: host-stage MPI buffers by default (NeoN_CUDA_AWARE_MPI=OFF); GPU-direct via =ON for HPC
- Local GPU testing: WSL2 WDDM mode, no MPS, no CUDA-aware MPI; `mpirun -np N UCX_TLS=tcp` with host-staged buffers
- HPC GPU testing: ad-hoc run + report back; CUDA-aware MPI available on cluster
- Branches: `feat/gpu-distributed` in both NeoFOAM (from `feat/distributed-correctness`) and NeoN (from `fix/testsRebase`)

### Pending Todos

- Create `feat/gpu-distributed` branches in NeoFOAM and NeoN
- Run `/gsd-plan-phase 1` to start Phase 1 planning

### Blockers/Concerns

None at milestone start.

## Deferred Items

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| ROCm/HIP | AMD GPU correctness | v3 | v2.0 init |
| SYCL | Intel GPU correctness | v3 | v2.0 init |
| Performance | CommunicationPattern caching | v3 | v1.0 init |
| Performance | Ginkgo matrix/solver caching | v3 | v1.0 init |
| BC | processorCyclic full support | v3 | v1.0 init |

## Session Continuity

Last session: 2026-05-13T10:34:38.464Z
Stopped at: Phase 1 not yet planned — ready to plan
Resume file: None
