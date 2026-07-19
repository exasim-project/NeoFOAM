# Study plan — impact of `mergeLevels` (aggressive MG coarsening via SpGEMM)

**Date:** 2026-07-09
**Case:** occDrivAerStaticMesh (DrivAer static mesh, kOmegaSST, steady SIMPLE, 4× H200, NP=4)
**Fits into:** `OptimizationStudyPlan-2026-07-08.md` Phase 3 (a Multigrid-tuning sweep), run from
the iteration-1000 restart, writing to `paperParamStudyResults/mergelevels/`.

---

## 1. What `mergeLevels` is (OpenFOAM) and why we want it here

In OpenFOAM's GAMG, **`mergeLevels N`** (an `fvSolution` GAMG control, default 1) combines `N`
consecutive agglomeration steps into a **single** coarser level. Increasing it makes the coarsening
more aggressive — the hierarchy has **fewer, coarser levels** for the same finest and coarsest grid,
trading a slightly weaker grid transfer for far fewer levels to sweep.

NeoN/Ginkgo has **no equivalent**. The reference uses `multigrid::Pgm` with the default ~4×
coarsening per level and `max_levels = 10`, which on this case builds an **~8-level** hierarchy
(§4.7 reconstructed L0 ≈ 17.3 M rows down to L7 ≈ 35 k, ~4× per level).

**Why it matters — the §4.7 dispatch-overhead finding.** The Phase-1 nsys break-down showed the
solve is **synchronization-bound**, not compute-bound: 95 k mostly-tiny kernels, GPU idle 78 %,
27 s of per-kernel stream/event sync. Critically, the **deep coarse levels do microseconds of
arithmetic but each SpMV/smoother kernel still pays the full ~3.4 µs launch + a stream sync** — the
tail of the hierarchy is nearly pure dispatch overhead. **Halving the level count directly removes
that tail.** This is the single most-motivated lever from the cost break-down after caching
(§4.9), because it attacks the *dominant* cost (dispatch), not the *minor* one (coarse-grid flops).

## 2. The SpGEMM approach (the prior-session idea, made concrete)

Pgm produces, at each level ℓ, a prolongation **Pℓ** (a Csr operator; restriction Rℓ = Pℓᵀ), and
the Galerkin coarse operator Aℓ₊₁ = Rℓ Aℓ Pℓ. Ginkgo exposes these per level via
`gko::multigrid::MultigridLevel::get_prolong_op()` / `get_restrict_op()` / `get_coarse_op()`
(confirmed in `multigrid_level.hpp`), and Csr·Csr **SpGEMM** is available (`core/matrix/csr.cpp`).

**Merging `k` levels** = compose the `k` consecutive prolongations into one:
```
P_merged = P_ℓ · P_{ℓ+1} · … · P_{ℓ+k-1}          (k-1 SpGEMMs, all Csr·Csr)
A_merged = P_mergedᵀ · A_ℓ · P_merged              (Galerkin, the same triple product NeoN
                                                    already forms — see §3.1 update_matrix_value)
```
i.e. run Pgm to build the fine-grained aggregation as today, then **collapse every `mergeLevels`
consecutive Pgm steps into a single V-cycle level** by multiplying their prolongations. The coarsest
grid is unchanged; only the number of intermediate levels drops (≈ `depth / mergeLevels`). This is
exactly OpenFOAM's mergeLevels semantics, realized on the existing Pgm aggregation via SpGEMM —
no new coarsening algorithm, just operator composition.

Note the composed `P_merged` is **structural** (aggregation-based, value-independent), so it is built
once and the merged `A_merged` values are refreshed each solve by the *same* `update_matrix_value`
Galerkin path already used for caching (§3.1) — **mergeLevels composes with the Phase-3a cache win**,
it does not fight it.

## 3. Two-track plan (cheap go/no-go first, then the feature)

### Track A — emulate "coarser/fewer levels" with existing knobs (NO new code; run first)
Before investing in the SpGEMM feature, answer *does a shallower hierarchy even help on this case?*
using knobs we already have, from the restart, 50-step window, **cached**:

- **`max_levels` sweep** — 10 (ref) → 8, 6, 4, 3, 2. Fewer levels = shallower hierarchy (a blunt
  proxy for merging: it truncates the bottom instead of merging, so the coarsest grid gets *larger*,
  but it isolates "is the deep tail worth its dispatch cost?").
- **`min_coarse_rows` sweep** — 64 (ref) → 1 k, 10 k, 100 k — stop coarsening earlier, same effect.
- Optionally **Pgm `max_unassigned_ratio` / `max_iterations`** — looser matching → slightly more
  aggressive single-step coarsening (not true multi-level merge, but nudges the ratio > 4×).

**Read:** s/step (marginal), p-iters (mean/max), continuity, **and per-solve V-cycle kernel count**
(a 1-variant nsys re-trace on the best/worst to confirm the dispatch-overhead tail actually shrinks).
**Go/no-go:** if truncating to ~4 levels already recovers most of the dispatch tail *without*
wrecking convergence, the SpGEMM feature is worth building; if convergence collapses as soon as the
grid is coarsened, mergeLevels won't rescue it and we stop here.

### Track B — implement the SpGEMM level-merge (the actual `mergeLevels` feature)
If Track A is green, build it. **Implementation path (from reading the pinned Ginkgo + NeoN):**

- **Idiomatic form = a custom `mg_level` factory `MergedPgm`.** Ginkgo's `Multigrid` applies a
  coarsening factory (`mg_level`, a deferred factory vector) once per level. So the merge is best
  realized as a drop-in coarsener that jumps `k` levels: `MergedPgm` mirrors `Pgm`'s bases
  (`EnableLinOp` + `EnableMultigridLevel<ValueType>` + `UpdateMatrixValue`), runs `Pgm` `k` times
  internally, composes the prolongations, and publishes ONE merged level via
  `EnableMultigridLevel::set_multigrid_level(P_merged, A_merged, R_merged)`. No change to
  `Multigrid` itself; `max_levels` then counts merged levels.
- **Registration (named-registry pattern, same as the L1 criterion).** In `ginkgo.hpp` (ctor, where
  `gko::config::registry reg` is built and `l1CriterionKey` is `reg.emplace`'d), register
  `reg.emplace("neon::pgmMerge2", makeMergedPgmFactory(exec, 2))` (and 3, 4). The config JSON then
  references it by name: `"mg_level": ["neon::pgmMerge2"]`. (Ref: `neon-ginkgo-custom-criterion-registry`.)
- **The prolongation is NOT a plain Csr — key finding.** `Pgm::generate_local` returns the
  prolongation as a **`RowGatherer`** (`pgm.cpp:271`, "lightway prolongation" — aggregation just
  copies each coarse value to its fine members), and for a **distributed** matrix the prolong is
  wrapped into a `experimental::distributed::Matrix` (`pgm.cpp:297,372,492`). So composing two
  prolongations is:
  - *local path* (localized/Schwarz{Multigrid}): compose the RowGatherer index maps
    (`gather∘gather` = a single gather with composed row indices — cheaper than SpGEMM, no multiply),
    or convert to local Csr and SpGEMM.
  - *global distributed path* (the reference): the distributed prolong is **block-diagonal across
    ranks** — Pgm aggregates never cross the fixed partition — so composition is **rank-local**: each
    rank composes its local prolong block; no distributed SpGEMM is needed. Rebuild the merged
    `distributed::Matrix` prolong from the composed local blocks + the merged coarse partition.
    *(Assumption to verify first: that the distributed Pgm prolong has empty off-diagonal blocks.)*
- **Coarse operator:** after composing, `A_merged = R_merged · A · P_merged`. Convenient shortcut:
  iterating `Pgm` on each successive `get_coarse_op()` already yields, after `k` steps, exactly
  `A_merged = P_mergedᵀ A P_merged` — so the last inner-Pgm's coarse_op IS the merged coarse operator;
  only `P_merged`/`R_merged` need the explicit composition.
- **Cache/updatability:** `P_merged` is structural (compose once); implement
  `MergedPgm::update_matrix_value(A_new)` to recompute only `A_merged = R_merged A_new P_merged`,
  reusing the frozen `P_merged`. Verify `p-cache: reuse(update_matrix_value)` still engages.
- **Smoother placement:** the merged level keeps one pre/post smoother built against `A_merged`.

**Risk / de-risking:** the distributed rank-local composition is the crux and the main correctness
risk (partition/index-map bookkeeping across merged levels). Recommended de-risk: prototype
`MergedPgm` on the **localized** MG path first (`p-multigrid-localized-solver.json` — each rank owns a
purely LOCAL `Multigrid` on a local Csr, so prolong composition is a trivial local RowGatherer/Csr
merge with zero distributed complexity). Confirm the dispatch-overhead win and convergence there,
then extend the composition to the global distributed prolong. This keeps the first working result
cheap and isolates the hard distributed bookkeeping to a second step.

### Sweep (Track B)
From the restart, 50-step window, **cached**, Cg + global Multigrid (the §4.9 winner):
`mergeLevels ∈ {1 (=ref), 2, 3, 4}` → hierarchy depth ≈ {8, 4, 3, 2}.
Optionally cross with `max_levels` to control the resulting depth precisely.

## 4. Metrics & deliverable

Per variant (mirroring the cache/scalecorr studies, `paper-study-common.sh`, 50 steps):
- **marginal s/step** (the headline — does dispatch saving beat any convergence cost?),
- **# hierarchy levels** actually built (log it in setup),
- **p-iters** min/mean/max and **continuity** (convergence must not degrade),
- **per-solve V-cycle kernel count + GPU-idle %** from a single confirming nsys re-trace (Track A
  worst-vs-best and Track B ref-vs-best) — this is the direct evidence the dispatch tail shrank,
  tying the result back to §4.7.

**Expected outcome / hypothesis:** because the case is dispatch-bound (§4.7), merging 8→4 levels
should cut per-V-cycle kernels/syncs roughly in half and lower s/step, provided p-iters rise only
mildly (coarser grid transfer). The risk is the classic AMG trade — too-aggressive merging weakens
interpolation and inflates iteration count; the sweep finds the knee.

**Script:** `phase3c-paper-study-mergelevels.sh` (fork of the cache/scalecorr harness), variants
named `ml1 … ml4`, all cached, `STEPS=50`, results in `paperParamStudyResults/mergelevels/`.

## 5. Caveats to carry in
- **Krylov compatibility (from the scale-correction run):** scale correction made the MG
  preconditioner nonlinear and diverged under `Cg` (needs FCG/flexible or MG-as-solver). mergeLevels
  is a *linear* change to the preconditioner (fixed composed operators), so it stays Cg-compatible —
  but if Track B is ever combined with scale correction, use FCG.
- **Localized MG:** the merge composes per-rank in the localized/Schwarz path; global-MG first
  (matches the study's global reference). Do not combine with the known localized-scalecorr dead-end.
- **SpGEMM setup cost:** the `k-1` extra SpGEMMs per merged level are one-time (structural, cached);
  they must not be paid every solve — verify they sit on the build path, not the update path.
