# Performance Report — occDrivAerStaticMesh best-practice study

**Date:** 2026-06-28
**Run:** `./run-all-studies-profiled.sh` (`KOKKOS_TOOL=space-time-stack`)
**Case:** occDrivAerStaticMesh, kOmegaSST, steady SIMPLE, 30 outer iterations
**Execution:** neoSimpleFoam, **4 MPI ranks, host/OpenMP** (Kokkos CUDA space = 0 kB — this is a CPU profiling build, *not* the GPU production build)
**Profiler:** Kokkos-Tools space-time-stack (per-rank summary folded into each run log + `*.kokkos-profile.txt`)

> Scope note: this invocation only refreshed the **best-practice** study (`param-study-best.sh`). The
> `mg-tuning`, `mg-headline` and `mixed-precision` studies still carry their 2026-06-27 data — they need
> `NEON_BUILD=production` and were not re-run here. Mixed-precision remains blocked at solver setup
> (Schwarz float/bf16 — see "Known blockers").

---

## 1. Headline result — a 25% speedup landed

The two final runs of this invocation:

| Config | Run | exec_s | s/step | p-iters/solve | Kokkos total |
|---|---|---:|---:|---:|---:|
| **Best practice** (Ginkgo Cg + localized Multigrid precond, cached, rebuild=100) | `pMG-localized-best-…-105220` | **152.99 s** | **5.10** | 53 | 157.1 s |
| Baseline (Ginkgo Cg + diagonal/Jacobi precond, no cache) | `pPCG-diagonal-best-…-105501` | **239.52 s** | **7.98** | 548 | 244.0 s |

Compared with the immediately-prior comparable runs (same config, earlier today / yesterday):

| Config | Before | After (this run) | Δ |
|---|---:|---:|---:|
| Best-practice MG | ~204.6 s (`…-095721`), long-standing ~200 s plateau | **153.0 s** | **−25.2 %** |
| PCG baseline | ~288–291 s (`…-094948 / …-100049`) | **239.5 s** | **−16.7 %** |

The jump is sharp and reproducible: every MG run up to 09:57 sat at ~200–205 s; from the 10:08 run onward
(`…-100819` 152.1 s, `…-102149` 152.5 s, `…-105220` 153.0 s) it is ~152 s. Same config name → this is a
**code/build change, not a tuning change** — commit `6027b5ca "fix linearUpwindV overhead"`.

---

## 2. Where the 50 s went — `luw.gradOpCtor`

The entire improvement is in the **momentum predictor**, specifically the linearUpwindV gradient-operator
construction inside the deferred correction:

| Region (MG run) | Slow `…-095721` | Fast `…-105220` | Δ |
|---|---:|---:|---:|
| `neoSimpleFoam.momentumPredictor` | 69.1 s (33.8 %) | 20.7 s (13.2 %) | **−48.4 s** |
| → `divlap.deferredCorr` | 55.2 s (27.0 %) | 7.8 s (5.0 %) | −47.4 s |
| → → **`luw.gradOpCtor`** | **51.5 s (25.2 %)** | **1.85 s (1.2 %)** | **−49.6 s (28× faster)** |
| → → `luw.applyCorr` | 1.11 s | 4.68 s | +3.6 s |

`luw.gradOpCtor` was reconstructing the gradient operator (and its backing `Vector` allocations — still the
single largest Kokkos-host-memory owner at the high-water mark, ~90 % of top allocations) on **every**
momentum assemble. The fix made the operator construction cheap. The same fix accounts for the PCG run's
~46 s drop (its momentum predictor went 69 s → 20.8 s identically; PCG is just dominated by its pressure
solve so the relative gain is smaller).

---

## 3. Current bottleneck profile (fast MG run, 157 s Kokkos / 30 steps)

| Rank | Region | Time | % | Notes |
|---:|---|---:|---:|---|
| 1 | `pressureCorrector.pEqn` | 55.6 s | 35.4 % | of which **MG solve ≈ 44.5 s**, `solverSetup` 7.25 s, `createMtx` 3.24 s |
| 2 | `setup` (one-time) | 43.4 s | 27.7 % | mesh/decomposition/IO — amortizes over step count |
| 3 | `momentumPredictor` | 20.7 s | 13.2 % | `assemble` 9.4 s (deferredCorr 7.8 s: applyCorr 4.68 s, gradOpCtor 1.85 s), `createMtx` 3.7 s, `construct` 2.34 s |
| 4 | `turbulenceCorrect` (k+ω) | 20.0 s | 12.8 % | 2 solves/step; `createMtx` 6.92 s dominant |
| 5 | `write` | 2.0 s | 1.3 % | |

**Cross-cutting:** `ginkgo.createMtx` aggregates to **13.9 s (8.8 %) over 120 calls** (bottom-up) — the matrix
is re-assembled into Ginkgo CSR for p, U, k and ω on every solve.

Convergence context: first p-solve 77 iters / 2053 ms, settling to ~20 iters / 564 ms by step 30; mean 53
iters. First-step `ExecutionTime` is 53.1 s (≈ 43 s setup + first solve), reaching 153 s at step 30.

---

## 4. Why "best practice" = MG (the baseline comparison)

| | MG (localized, cached) | PCG (diagonal/Jacobi) |
|---|---:|---:|
| p-iters / solve | 53 | 548 (~10×) |
| `pEqn` cost | 44.5 s (28 %) | **138 s (57 %)** (~3.1×) |
| Total | 153 s | 239 s |

The localized-Multigrid preconditioner cuts pressure iterations ~10× and the pressure solve ~3×. The PCG run
exists purely as the reference that justifies the MG choice — it is doing the same momentum/turbulence work
but spends 57 % of wall-time grinding the under-preconditioned pressure system.

---

## 5. Recommended next steps (prioritized)

**1. Reuse the Ginkgo matrix sparsity across SIMPLE iterations — ~14 s / 9 % (highest ROI, host).**
`createMtx` rebuilds the full CSR (symbolic + numeric) 120× even though the **sparsity is invariant** across
outer iterations — only the values change. Extend the existing solver/preconditioner cache (which already
reuses the MG hierarchy via `update_matrix_value`, rebuild=100) to also retain the CSR structure and do a
values-only update for p, U, k, ω. This is the single biggest *host-side* lever left after the gradOpCtor fix.

**2. `luw.applyCorr` is now the top momentum kernel — 4.68 s (3 %).** After gradOpCtor was fixed, the explicit
deferred-correction apply for linearUpwindV is the new leader inside `deferredCorr`, and it is 99 %
host-"remainder" (host-bound, not Kokkos kernel time). Candidate for kernel fusion / temporary elimination,
in the same spirit as the linearUpwindV fix.

**3. Move the perf frontier to GPU (production build).** This profiling run is host/OpenMP (CUDA space = 0).
The 44.5 s MG solve + assembly are CPU work; real throughput lives in the CUDA production build. Before/while
doing so, mind the two recorded GPU hazards:
  - the `cudafe gko::LinOp` signature bug — keep the solver cache in host-only `solveDist` members;
  - the **assemble allocator churn** — `momentum.assemble` host "remainder" balloons unless
    `allocator=UmpirePool` (raw `cudaMalloc/cudaFree` per temporary otherwise; `memPoolSize` is ignored unless
    the pool allocator is selected). Verify the pool is actually active on GPU.

**4. Pressure-solver tuning — chip at the 44.5 s solve.** 53 iters/solve (77 on the first step) leaves room:
sweep MG `max_levels`/smoother sweeps and `preconditionerRebuildInterval`, and consider a tighter
relTol/coarse-grid strategy. This is the largest single region and currently untuned beyond "localized + cache".

**5. `setup` 43 s (28 %) — only matters for short runs.** It is one-time mesh/decomposition/IO and amortizes to
near-zero over a production-length run (1000s of steps). Do **not** optimize it for throughput; only revisit if
startup latency itself becomes a goal. (It dominates *this* 30-step study, which inflates the per-step number —
real per-step cost is ~(153−43)/30 ≈ **3.7 s/step**, not 5.1.)

**6. Refresh the other studies on the production build.** `mg-tuning`, `mg-headline` and `mixed-precision` were
not re-run today and carry 2026-06-27 data. Re-run with `NEON_BUILD=production` to get them onto the
post-gradOpCtor-fix baseline.

### Known blockers (carried over)
- **Mixed precision** (`innerPrecision` float/bfloat16, per-node value_type): aborts at distributed-Schwarz
  setup — `float` throws `gko::NotSupported` in `extract_local_matrix` (can't `gko::as` the fp64 distributed
  matrix); `bfloat16` isn't in Schwarz's `value_type_list_base`. fp64 remains the only working localized path.

---

## 6. Run inventory (this invocation)

| Run | steps | exec_s | Kokkos total | result |
|---|---:|---:|---:|---|
| `pMG-localized-best-cache-rebuild100-20260628-103324` | 0 | — | — | aborted (early failure) |
| `pPCG-diagonal-best-20260628-103329` | 0 | — | — | aborted (early failure) |
| `pMG-localized-best-cache-rebuild100-20260628-105220` | 30 | 152.99 | 157.1 s | **OK** |
| `pPCG-diagonal-best-20260628-105501` | 30 | 239.52 | 244.0 s | **OK** |

The 10:33 pair aborted at step 0; the 10:52/10:55 pair is the clean completed run analyzed above.
