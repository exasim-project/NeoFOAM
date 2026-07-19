# Performance Report — occDrivAerStaticMesh (full study sweep)

**Date:** 2026-06-28 (second invocation, ~15:05–17:15)
**Run:** `./run-all-studies-profiled.sh` (`KOKKOS_TOOL=space-time-stack`)
**Case:** occDrivAerStaticMesh, kOmegaSST, steady SIMPLE, 30 outer iterations
**Execution:** neoSimpleFoam, **4 MPI ranks, host/OpenMP** (Kokkos CUDA space = 0 — CPU profiling build)
**Supersedes:** `PerformanceReport-2026-06-28.md` (which only captured the best-practice pair). **This run
executed all five studies on the current post-`gradOpCtor`-fix build.**

---

## 1. Headline — the 25% best-practice win is stable, and the full sweep now validates the config

| Config | This run | Prior (10:52) | Pre-fix baseline | Status |
|---|---:|---:|---:|---|
| **Best practice** (localized MG, cached) | **154.5 s** | 153.0 s | ~205 s | **stable ✓** |
| PCG/diagonal baseline | 242.1 s | 239.5 s | ~285 s | stable ✓ |

The `luw.gradOpCtor` fix (commit `6027b5ca`) holds: best-practice sits at ~154 s across three independent
runs. The new value this time is the **full preconditioner sweep**, which lets us confirm *why* the
best-practice config is best — and surfaces one severe anti-pattern.

---

## 2. The solver cache is worth ~33 s / 18% — and profiling shows exactly where

Best-practice (cached) vs the otherwise-identical localized-L10 run without the cache:

| | exec | p-iters | **`ginkgo.solverSetup`** | `pEqn` total |
|---|---:|---:|---:|---:|
| `pMG-localized-best` (cache, rebuild=100) | **154.5 s** | 53 | **6.9 s** | 55.5 s |
| `pMG-localized-L10-ukoSmooth` (no cache) | 187.7 s | 53 | **40.2 s (20.8%)** | 88.1 s |

Same iteration count, same per-solve numerics — the entire 33 s gap is `solverSetup`: without the cache the
MG hierarchy is **rebuilt from scratch on every solve** (40.2 s), with it the hierarchy is built once and
refreshed in place (6.9 s). The cache is the single biggest *host-side* win already in the config; keep it on.

---

## 3. MG depth sweep — L10 is the sweet spot (localized)

| max_levels | exec | p-iters/solve | p ms/solve |
|---:|---:|---:|---:|
| L2  | 241.0 s | 94 | 3593 |
| L4  | 204.1 s | 73 | 2121 |
| L6  | 193.0 s | 61 | 1678 |
| **L10** | **187.5 s** | **53** | **1422** |
| L15 | 187.7 s | 51 | 1388 |
| L20 | 189.6 s | 51 | 1389 |

Convergence improves steeply to L10, then **plateaus** (L15/L20 give ≤1% and are capped by
`min_coarse_rows=64`). The best-practice `max_levels=10` is optimal — **no further win from depth tuning.**

---

## 4. Why "localized" — the iteration/cost trade-off (at L10)

| Variant | exec | p-iters | p ms/solve |
|---|---:|---:|---:|
| **localized** (Schwarz block-Jacobi) | **187.5 s** | 53 | **1422** |
| global / non-localized (`sc0`) | 244.7 s | **28** | 2765 |
| FCG + MG precond | 241.7 s | 28 | 2788 |
| CG-coarse solver | 257.5 s | 15 | 3326 |
| smooth2 | 253.9 s | 23 | 3194 |

The global MG needs ~half the iterations (28 vs 53) but each solve costs **~2× more** (2765 vs 1422 ms) — the
localized block-Jacobi V-cycle is far cheaper per iteration and wins by 24%. None of the alternative coarse
solvers / Krylov wrappers beat it.

---

## 5. ⚠️ Scale-correction-localized is catastrophic — and now confirmed on the current build

| Config | exec | p-iters/solve | p ms/solve | vs localized |
|---|---:|---:|---:|---:|
| localized (ukoSmooth) | 187.7 s | 53 | 1,428 | 1.0× |
| scalecorr **non**-localized | 270.7 s | **11** | 3,760 | 1.4× |
| **scalecorr + localized** | **1,708.9 s** | **446** | **52,255** | **9.1×** |

This is **worse than the archived old-build run** (1364 s / 50 iters) investigated earlier. On the current
build the **inner solve no longer converges**: 446 iterations/solve and 52 s per pressure solve. The profiler
is unambiguous about where it goes:

```
scalecorr-localized   pressureCorrector.pEqn = 1610 s = 93.9% of total runtime
                      (solverSetup only 32.9 s — it is the SOLVE, not setup)
best-practice         pressureCorrector.pEqn =   55 s = 34.9%
```

**Root cause (two compounding effects, from the source-level investigation):**
1. **Per-cycle blow-up.** With `scale_correction: true`, every V-cycle level runs an OpenFOAM-GAMG
   Rayleigh-scaled correction on *both* the down and up pass (`multigrid.cpp:664-694`, `:742-777`) — each
   adds an extra SpMV + two dots + a full extra smoother apply. Combined with the config's `pre_smoother
   max_iters: 2` and `post_uses_pre: true`, that is ~8 smoother sweeps + 3 SpMVs per level vs 2 + 1 for plain
   localized (~3–4× heavier V-cycle).
2. **Localization breaks the correction.** The outer `solver::Ir` (`scale_correction: backward`, no Krylov
   acceleration) computes its scaling from per-subdomain dot products inside the Schwarz block. Across the
   4-way decomposition these local scaling factors are **inconsistent**, so convergence collapses (iteration
   count explodes 53 → 446). The non-localized scalecorr keeps the iteration win (11 iters) precisely because
   its scaling is global.

**Recommendation: do not use scale-correction with localized Schwarz** — it is a numerical dead end here.

---

## 6. Mixed precision — still blocked

| Config | result |
|---|---|
| `mp-double` (fp64) | 156.3 s ✓ (matches best-practice) |
| `mp-float`, `mp-bfloat16`, `precfloat`, `precbf16` | **abort at step 1** |

Unchanged: the localized distributed-Schwarz construction rejects float/bfloat16 at solver setup. fp64 remains
the only working localized path. Unlocking this (potential ~halved solve cost) is gated on Ginkgo Schwarz
`value_type` support — a real upside if pursued, but a Ginkgo-side change.

---

## 7. Solver-cases — U/k/ω solver & matrix format (minor)

| Config | exec | note |
|---|---:|---|
| `pPCG-ukoPBiCGStab` | **211.9 s** | PBiCGStab for U/k/ω — marginally fastest |
| `pPCG-ukoSmooth` | 214.7 s | smoothSolver |
| `pCGsellp-ukoSmooth` | 216.5 s | sellp matrix format — no benefit over default CSR |
| `pMGbase-ukoSmooth` | 227.2 s | |

PBiCGStab edges out smoothSolver by ~1.3% for the U/k/ω blocks; the `sellp` storage format gives nothing on
this CPU run. Both effects are small relative to the pressure solve.

---

## 8. Current bottleneck profile (best-practice, 159 s Kokkos / 30 steps)

| Rank | Region | Time | % | Notes |
|---:|---|---:|---:|---|
| 1 | `pressureCorrector.pEqn` | 55.5 s | 34.9 % | MG solve ~44 s + `solverSetup` 6.9 s |
| 2 | `setup` (one-time) | 44.9 s | 28.2 % | mesh/decomp/IO — amortizes over step count |
| 3 | `momentumPredictor` | 20.9 s | 13.1 % | assemble + createMtx + construct |
| 4 | `turbulenceCorrect` | 20.0 s | 12.6 % | k + ω solves |
| — | `ginkgo.createMtx` (cross-cutting) | 7.5 s | 4.7 % | CSR re-assembled 120× (static sparsity) |

Unchanged in shape from the morning run — the `gradOpCtor` regression is gone and the pressure solve + one-time
setup now dominate.

---

## 9. Recommended next steps (prioritized, profiling-driven)

1. **Move the frontier to GPU (production build).** This is a CPU/OpenMP run (CUDA space = 0); the 44 s MG
   solve and assembly are host work. The CUDA build is where real throughput lives. Mind the recorded hazards:
   keep the solver cache in host-only `solveDist` members (cudafe `gko::LinOp` bug) and ensure
   `allocator=UmpirePool` is actually active (else per-temporary `cudaMalloc/cudaFree` churn in `assemble`).
2. **Reuse the Ginkgo CSR sparsity (`createMtx`, 7.5 s / 4.7%).** Structure is invariant across SIMPLE
   iterations; only values change. Extend the existing cache to retain the CSR symbolic structure and do a
   values-only update for p/U/k/ω. Smaller than it was (was 8.8%) but still free host time.
3. **Stop tuning the pressure preconditioner — it's converged.** The sweep shows localized-L10 + cache is the
   floor; depth, global MG, FCG, CG-coarse, smooth2 and `sellp` were all tried and none beat it. Do not spend
   more cycles here; redirect to GPU + `createMtx`.
4. **Pursue the Schwarz mixed-precision fix (Ginkgo-side, high upside).** fp32/bf16 inner solves could roughly
   halve the 44 s solve, but require float/bfloat16 support in the distributed-Schwarz `value_type` path. This
   is the largest *numerical* lever left, but it's a Ginkgo change, not a config change.
5. **Adopt `ukoPBiCGStab` for the U/k/ω blocks** (~1.3% on the PCG baseline) if not already in the production
   template — trivial, already in `fvSolution.case3`'s direction.
6. **Reduce the one-time `setup` share for short studies, or just run longer.** 28% (45 s) is fixed cost that
   amortizes to near-zero over a production-length run; the real per-step cost is ~(154−45)/30 ≈ **3.6 s/step**.

### Anti-patterns confirmed this run
- **scale-correction + localized Schwarz** → 9× slowdown, broken convergence (§5). Never combine.
- **float/bfloat16 localized** → aborts at setup (§6).

---

## 10. Run inventory (this invocation)

| Study | runs | notable |
|---|---|---|
| best-practice | MG 154.5 s, PCG 242.1 s | stable vs morning |
| mg-tuning | L2–L20 localized + non-localized + FCG/coarse/smooth variants | L10 optimal |
| mg-headline | plain/localized/scalecorr/scalecorr-localized/solver | **scalecorr-localized 1709 s** |
| mixed-precision | double 156 s; float/bf16 abort | blocked |
| solver-cases | uko Smooth/PBiCGStab, sellp, MGbase | PBiCGStab marginally best |

A few entries aborted early (`pMG-directcoarse`, `pMG-localized-solver`, `pMG-L15-sc0` at 3 steps, the four
mixed-precision float/bf16 cases) — failed runs, excluded from the comparisons above.
