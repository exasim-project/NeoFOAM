# ⚠️ SUSPECT RESULTS — GPU oversubscription (fixed 2026-07-17)

**All timing results in this tree produced before 2026-07-17 are SUSPECT.** They were measured with
`paper-study-common.sh` binding `CUDA_VISIBLE_DEVICES=1,2,3,4` on a node whose GPUs are indexed 0–3.
Device "4" does not exist, so **two MPI ranks stacked onto GPU1 while GPU0 sat idle** — 3-GPU
oversubscription instead of 4-GPU parallelism.

## Confirmed impact (clean 50-step A/B, iteration counts identical between arms)

| cell | bind 1,2,3,4 (stacked) | bind 0,1,2,3 (1 rank/GPU) | p-solve | s/step |
|---|---|---|---|---|
| global no-sc L2/merge3   | 761.5 ms · 3.185 s/step | 177.3 ms · 1.661 s/step | **4.30× slower** | 1.92× |
| localized no-sc L6/merge3 | 732.6 ms · 3.096 s/step | 202.2 ms · 1.628 s/step | **3.62× slower** | 1.90× |

The stacked arm reproduced the plotted sweep numbers almost exactly (761↔763, 733↔733), proving the
figures were taken oversubscribed. **Absolute timings are inflated ~1.9× on s/step, up to ~4.3× on
pressure-solve. Relative rankings across cells are probably still valid** (every cell shared the same
handicap), but no absolute ms / s/step number here is publishable as-is.

## Fix

`paper-study-common.sh:32` → `export CUDA_VISIBLE_DEVICES=0,1,2,3` (one rank per GPU). Landed
2026-07-17. Every study run after this date via the paper harness is on the corrected binding.

## Re-run under the corrected binding — ✅ COMPLETE 2026-07-17 (`rerun-corrected-binding.sh`, 318 min, 0 errors)

The **newest (post-2026-07-17) log per cell** in these dirs is the trustworthy corrected number; older
logs remain oversubscribed and are ignored by the plotters (they pick the newest per cell). Corrected
figures regenerated: `reltol-grids-{pms,time,iters}.png`, `merge-sweep-pms.png`.

**Corrected rel-tol optima (steady s/step · p-solve):** global no-sc L4/tol0.15 0.810 s/174 ms ·
global sc-post L6/tol0.1 0.790 s/151 ms · localized no-sc L8/tol0.25 0.776 s/210 ms · localized
sc-post L8/tol0.1 0.814 s/243 ms. **All four branches now cluster at 0.78–0.81 s/step** — the binding
bug (not the preconditioner choice) had been the dominant term. Pre-fix these looked like 0.81 vs
2.2–2.4 s/step, wrongly making localized/sc-post appear far worse.

- `cost-breakdown/`        (phase1)
- `cache-compare/`         (phase3a)
- `mgnosc-coarse-reltol/`, `mgscpost-coarse-reltol/`, `localized-coarse-reltol/`, `localizedsc-coarse-reltol/`  (rel-tol grids, merge2 anchor)
- `*-merge1/`, `*-merge3/` (merge-sweep coarsener panels)

### What "merge*N*" means in these cell names

`mergeLevels` (NeoN's `MergedPgm`, named after OpenFOAM GAMG's parameter) runs Pgm aggregation `N`
times but exposes the result as **one** multigrid level, so a merge-3 level coarsens ~8× in a single
V-cycle visit and the hierarchy gets ~`N`× shorter. It is selected per case not by an `fvSolution`
key but inside the Ginkgo `configFile` JSON that `solvers { p { configFile ...; } }` points at, as
`"mg_level": ["neon::pgmMerge<N>"]`. That string is a *registry key*, not a Ginkgo type: the
`GinkgoSolver` ctor pre-registers `neon::pgmMerge{1,2,3,4}` (`NeoN .../ginkgo.hpp`), so `N` is fixed at
registration and the sweep harnesses just rewrite the `mg_level` entry with `jq`. `pgmMerge1` is plain
Pgm through the same code path (the control arm). **Important when reading these tables:** `max_levels`
counts *merged* levels, so effective Pgm depth ≈ `N × max_levels` — the merge panels above are not
apples-to-apples at a fixed `max_levels`. Full mechanism and cost: §4.14 of
`PaperOptimizationStudy-2026-07-09.md`, §2.5 of `PaperOptimizationStudyManuscript.md`.

## Everything else here is SUSPECT and NOT yet re-run

All other subdirectories (e.g. `spinup`, `cost-breakdown-average-iteration`, `l5c4-costbreakdown`,
`mgsc-*`, `scalecorr*`, `chebyshev-foci`, `comm-attribution`, `reltol-grids-*`, `solver-memory`, the
`_archived-*` trees, and every `*.png` derived from them) carry oversubscribed absolute timings. Treat
their wall-time / s/step / ms numbers as upper bounds only until re-measured. Iteration counts,
continuity errors, and convergence behaviour are unaffected by the binding and remain valid.
