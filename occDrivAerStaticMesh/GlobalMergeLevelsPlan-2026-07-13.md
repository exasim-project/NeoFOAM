# Implementation plan — **global (non-localized)** `mergeLevels` for Ginkgo MG

**Date:** 2026-07-13
**Case:** occDrivAerStaticMesh (kOmegaSST, steady SIMPLE, 4× H200, NP=4)
**Follows:** `MergeLevelsStudyPlan-2026-07-09.md` (Track B) and the completed **localized** MergedPgm
study (`mergelevels-no-speedup`: localized ml1…ml4, no speedup — dispatch saving canceled a +51 %
iter cost). This plan is **Track B step 2**: extend the level-merge to the **global distributed**
Multigrid path (the §4.9 reference), where merged levels *retain the cross-rank coarse coupling that
localization discards* — the reason the global trade-off may differ from the localized null result.

---

## 0. Why do the global version at all (given localized was a wash)

The localized study ran MergedPgm inside `preconditioner::Schwarz` on each rank's **local Csr**, so
every merged level threw away all cross-rank coupling. Two consequences that make its null result
**not** transferable to the global path:

- **Convergence:** the global coarse operator keeps inter-rank coupling (`communicate_non_local_agg`),
  so a merged global level is a *stronger* preconditioner per V-cycle than a merged localized level.
  The +51 % iter penalty measured under localization is an upper bound; global merging may cost less.
- **Cost mix:** but a merged global coarse op has denser non-local blocks **and** the same per-level
  halo exchange, so the dispatch/sync saving (the whole motivation, §4.7) is partly offset by
  retained MPI communication. The knee could land anywhere.

So the global result is genuinely unknown and worth one clean sweep. Hypothesis unchanged: 8→4 levels
halves per-V-cycle kernels; question is whether global convergence holds better than localized did.

---

## 1. Findings from reading the current tree (what's true *now*, not in the older plan)

Verified against `/storage/home/greole/code/NeoN` (branch `enh/turbulence`) and the pinned fork under
`NeoFOAM/build/profilingnvidia_h200/_deps/ginkgo-src`:

1. **Global vs localized config shape** (`occDrivaerRun.../system/gko/`):
   - *Global* (`p-multigrid.json`): top-level `solver::Cg` → `preconditioner: solver::Multigrid`
     applied **directly to the distributed matrix**; Pgm runs distributed; Schwarz sits *inside* each
     smoother's `solver`. **This is the target path.**
   - *Localized* (`p-multigrid-localized-solver.json`): top-level `preconditioner::Schwarz` →
     `local_solver: solver::Multigrid` on each rank's local Csr. (What the null-result study used.)

2. **Distributed Pgm prolong is block-diagonal — CONFIRMED** (`core/multigrid/pgm.cpp`,
   `distributed_setup`, lines ~448–467):
   - `coarse` (448–452): `distributed::Matrix` built **with** a non-local block
     (`result_non_local_csr`) + `coarse_imap` → carries cross-rank coupling.
   - `restrict_op` (453–459) and `prolong_op` (460–466): `distributed::Matrix` built from **only the
     local block** (`std::get<2>`/`std::get<0>` of `generate_local`), `dim<2>` given explicitly, **no
     non-local matrix, no explicit imap**. Ginkgo infers the block-diagonal partition from local
     sizes. ⇒ **prolong/restrict have empty off-diagonal blocks; composition is rank-local.**
   - The local prolong itself is a `matrix::RowGatherer<IndexType>` ("lightway prolongation",
     `generate_local` ~248) — an injection index map, not a value matrix.

3. **Merged coarse op comes for free.** Running Pgm `k` times, each on the previous distributed
   `get_coarse_op()`, yields after `k` steps exactly `A_merged = R_merged·A·P_merged` **as a proper
   distributed::Matrix with the correct non-local coupling and coarse partition/imap**. So only
   `P_merged`/`R_merged` need explicit composition; `A_merged` = the last inner Pgm's coarse op.

4. **Cache/update path exists in the pinned fork only.** `multigrid_level.hpp:22` defines the
   `UpdateMatrixValue` mixin; `pgm.cpp:518 Pgm::update_matrix_value` re-runs the distributed Galerkin
   refresh **reusing the frozen aggregation** `agg_`. MergedPgm must mirror this by chaining its
   retained inner Pgms. (Absent from stock `build/develop` Ginkgo — build against the pinned fork.)

5. **✅ Registry infra IS wired on `enh/kOmegaSST`** (re-checked 2026-07-13, was NOT on
   `enh/turbulence`). `include/NeoN/linearAlgebra/ginkgo.hpp:437-450` builds a populated
   `gko::config::registry reg`, `reg.emplace`s the L1 criterion **and already**
   `reg.emplace("neon::pgmMerge{2,3,4}", makeMergedPgmFactory<scalar>(gkoExec_, k))`. `factory_` is
   parsed against `reg` (line 494; a mixed-precision inner-factory path at 468 too). **Step 0 of the
   original plan is DONE — no registry work remains.**

6. **✅ `MergedPgm` already exists** — `include/NeoN/linearAlgebra/ginkgo/mergedPgm.hpp` (tracked on
   this branch) + `makeMergedPgmFactory`. It already does the §2 host index-composition
   (`mergedAgg[k]=aggThis[mergedAgg[k]]`, no SpGEMM), the retained-Pgm-chain `update_matrix_value`,
   and uses the "last coarse op == A_merged" shortcut. **But it is LOCALIZED-ONLY (finding 7).**

7. **⚠ The existing `MergedPgm` cannot take a distributed matrix — this is the actual remaining
   work.** `generate()` opens with `as_csr(system_matrix_)` (line 119): a `distributed::Matrix` is
   neither `Csr` nor `ConvertibleTo<Csr>`, so it throws `gko::NotSupported`. Same for `gather_agg`
   (casts the prolong straight to `RowGatherer`; the distributed Pgm prolong is a `distributed::Matrix`
   wrapping the RowGatherer) and `update_matrix_value` (`as_csr` on the distributed coarse op). So the
   global path is **not** a new class — it is a **distributed branch added to the existing
   `mergedPgm.hpp`** (§3).

---

## 2. The algorithm (global, per merged level, `k = mergeLevels`)

Inside `MergedPgm::generate()` given distributed fine op `A₀`:

```
levels = []
A = A₀
for i in 0..k-1:                          # run stock distributed Pgm k times
    lvl_i = Pgm(params).generate(A)       # lvl_i.prolong = block-diag dist::Matrix (local RowGatherer)
    levels.push(lvl_i)                    #  lvl_i.coarse  = dist::Matrix WITH non-local coupling
    A = lvl_i.get_coarse_op()             # feed coarse op to next Pgm
A_merged = A                              # == R_merged·A₀·P_merged, distributed, correct imap  (finding 3)

# compose prolongations — rank-local, block-diagonal (finding 2)
localProlong = identity index map on this rank's fine rows
for i in 0..k-1:
    g_i = row_idxs of lvl_i.prolong.get_local_matrix()          # RowGatherer local block
    localProlong = compose_index(localProlong, g_i)             # gather∘gather, host, O(nfine)
P_local = RowGatherer(localProlong)          # or its Csr injection, this rank only
R_local = transpose index map of P_local     # restrict = injection^T (as Pgm does, agg_to_restrict)

# re-wrap as block-diagonal distributed operators (mirror pgm.cpp:460-466 exactly)
P_merged = dist::Matrix::create(exec, comm, dim<2>(fineGlobal, coarseGlobal), P_local)
R_merged = dist::Matrix::create(exec, comm, dim<2>(coarseGlobal, fineGlobal), R_local)

set_multigrid_level(P_merged, A_merged, R_merged)
```

`compose_index(a, b)[f] = b[a[f]]` — a single fused gather (cheaper than SpGEMM, matches the localized
fix). Because prolong/restrict are block-diagonal, **no MPI in the composition**; the coarse
partition of `P_merged`/`R_merged` is inferred from the local coarse size, which equals
`A_merged`'s local row count by construction (same `k`-th Pgm). Keep `levels` alive as members for
the update path.

**Update (cache) path** — `MergedPgm::update_matrix_value(A_new)`:
```
A = A_new
for i in 0..k-1:
    levels[i].update_matrix_value(A)      # reuses frozen agg_, refreshes coarse values (pgm.cpp:518)
    A = levels[i].get_coarse_op()
# A_merged values now refreshed; P_merged/R_merged are structural — leave frozen
set_multigrid_level(P_merged, A /*refreshed A_merged*/, R_merged)
```
This keeps `[GinkgoSolver] p-cache: reuse(update_matrix_value)` engaged and pays the `k`-Pgm
generate cost only once.

---

## 3. Implementation steps

**Step 0 — DONE** (registry populated, factory registered; findings 5–6).

**Step 1 — DONE** (localized `MergedPgm` exists and is the null-result implementation).

**Step 2 — DONE** (`makeMergedPgmFactory` + `reg.emplace("neon::pgmMerge{2,3,4}")`).

**Step 3 (the only code work) — ✅ DONE 2026-07-13, builds clean.** Distributed branch added to
`mergedPgm.hpp` (`generateDistributed()` + `update_matrix_value` isDist branch + `dist_mtx` alias +
`agg_from_rowgatherer`/`composeAgg`/`make_injection` refactor; `prolong_`/`restrict_` widened to
`shared_ptr<const LinOp>` to hold either a Csr (local) or a block-diagonal `distributed::Matrix`
(global)). Verified via `scripts/coma/build-nvidia-h200-gcc.sh` (cuda/13.1.1, gcc/13.3.0,
openmpi-cuda/5.0.10): full NeoFOAM build 148/148, 0 errors; confirmed `NeoN_DEFINE_DP_LABEL=OFF`
(label=int32) so `dist_mtx = Matrix<double,int32,int64>` matches the registered `MergedPgm<scalar,
gko::int32>`. **Remaining: correctness gates §4 + sweep §5 (runtime, not yet done).** Design notes for
the distributed branch: guard with
`std::dynamic_pointer_cast<const gko::experimental::distributed::DistributedBase>(system_matrix_)`;
keep the existing Csr path verbatim for the localized/Schwarz case (A/B parity). In the distributed
branch:
- **Inner Pgm runs on the distributed op directly** — stock Pgm handles it (`pgm.cpp distributed_setup`),
  yielding `level->get_prolong_op()`/`get_coarse_op()` as `distributed::Matrix`. Feed each
  `get_coarse_op()` (distributed) to the next inner Pgm — do **not** `as_csr` it.
- **`gather_agg` sources from the local block:** `as<distributed::Matrix>(prolong)->get_local_matrix()`
  → cast to `RowGatherer` → pull `row_idxs` to host. Composition `mergedAgg[k]=aggThis[mergedAgg[k]]`
  is unchanged and **rank-local** — valid because the prolong is block-diagonal (finding 2).
- **`coarseRows` is the LOCAL coarse row count** on this rank (prolong local block col count).
- **Re-wrap merged prolong/restrict as block-diagonal `distributed::Matrix`** mirroring
  `pgm.cpp:460-466`: `dist::Matrix::create(exec, comm, dim<2>(fineGlobal, coarseGlobal), P_local)` with
  **no** non-local block, **no** explicit imap (Ginkgo infers the block-diagonal partition). Get `comm`
  from `as<DistributedBase>(system_matrix_)->get_communicator()`; global dims from
  `system_matrix_->get_size()` and the merged coarse op's size.
- **`A_merged` = the last inner Pgm's distributed `get_coarse_op()`** (already carries correct
  non-local coupling + coarse imap — finding 3); pass it straight to `set_multigrid_level`, no `as_csr`.
- **`update_matrix_value` distributed branch:** the existing retained-chain loop already works if the
  per-level `get_coarse_op()` is fed as-is (distributed) instead of `as_csr`'d; drop the final `as_csr`
  and set the distributed coarse op directly. `prolong_`/`restrict_` stay frozen.
- Note the registry emplaces `MergedPgm<scalar, gko::int32>`; distributed matrices here use index type
  `label`/`localIdx`. Confirm `int32 == localIdx` on the target build (it is on H200) or template the
  factory on the distributed index type.

**Step 4 — configs** (`system/gko/`): fork `p-multigrid.json` → `p-multigrid-merge{2,3,4}.json`,
changing only:
```json
"mg_level": [ { "type": "neon::pgmMerge2" } ]
```
Everything else (Cg wrapper, Schwarz-in-smoother, `max_levels`, coarsest solver, l1 criterion)
identical to the global reference so the sweep isolates merging. Keep `max_levels: 10` (counts merged
levels now → effective depth ≈ 10, but the hierarchy bottoms out earlier; optionally also emit a
`max_levels`-matched pair to control final depth precisely, per 07-09 §Sweep).

**Step 5 — build** against the pinned fork (`NEON_BUILD=profiling`, the checkout that has
`UpdateMatrixValue`). Rebuild NeoN + solver.

---

## 4. Correctness gates (before any perf sweep)

Run these on a **small** distributed case first (2–4 ranks, pitzDaily-scale) — cheap and they isolate
the distributed bookkeeping that is the whole risk:

1. **P_merged is a valid injection.** Each fine row maps to exactly one coarse col; coarse col count
   == `A_merged` local rows on every rank. Assert per rank.
2. **Block-diagonal invariant.** `P_merged.get_non_local_matrix()` is empty (num_stored == 0) on all
   ranks — the composition must not manufacture cross-rank prolong entries.
3. **Galerkin consistency.** Compare `A_merged` against an explicit `R_merged·A₀·P_merged` triple
   product (distributed apply on a random vector, check ‖·‖ agrees) — guards the "free coarse op"
   shortcut (finding 3).
4. **k=1 identity.** `neon::pgmMerge1` must be bit-for-bit the stock global `multigrid::Pgm` run
   (same iters, same residual trace). This is the single most valuable regression test.
5. **Cache reuse.** Second solve logs `p-cache: reuse(update_matrix_value)`; iters/residual match a
   fresh generate to tolerance.

Only after 1–5 pass on the small case, move to occDrivAer.

---

## 5. Sweep & metrics (occDrivAer, from the iter-1000 restart, 50-step window, **cached**)

Mirror `mergelevels-no-speedup` so results are directly comparable to the localized table:

| variant | mg_level | expected effective depth |
|:-:|:-:|:-:|
| gm1 (=global ref) | `multigrid::Pgm` | ~8 |
| gm2 | `neon::pgmMerge2` | ~4 |
| gm3 | `neon::pgmMerge3` | ~3 |
| gm4 | `neon::pgmMerge4` | ~2 |

Per variant record: **marginal s/step** (headline), **# levels actually built** (log in setup),
**p-iters min/mean/max**, **continuity**, and — on ref-vs-best only — a single **nsys re-trace** for
**per-solve V-cycle kernel count + GPU-idle %** (direct §4.7 evidence the dispatch tail shrank).
Add one extra column absent from the localized study: **per-solve halo-exchange count / MPI time**,
since that is exactly what the global path retains and the localized path lacked — it's the variable
that decides whether global merging beats the localized wash.

**Script:** `phase3d-paper-study-global-mergelevels.sh` (fork the localized harness), variants
`gm1…gm4`, all cached, `STEPS=50`, results in `paperParamStudyResults/mergelevels-global/`.

**Go/no-go:** global merging *wins* only if p-iters rise **less** than the +51 % seen localized
*and* the retained MPI cost doesn't eat the dispatch saving. If gm2 already shows the same flat/worse
wall time as ml2, stop — the coarsening penalty dominates on this case regardless of localization,
and the conclusion "plain Pgm (gm1) is best" stands globally too.

---

## 6. Risks / caveats to carry in

- **The crux risk is distributed partition/index-map bookkeeping**, not the composition math. Gates
  §4.1–4.3 exist to catch it. The block-diagonal finding (finding 2) is what makes it tractable —
  if a future Ginkgo bump changes `distributed_setup` to give the prolong a non-local block, this
  whole approach needs re-derivation.
- **Cg compatibility:** merging is a *linear* preconditioner change (fixed composed operators) → stays
  Cg-compatible. Do **not** combine with scale correction (needs FCG; see
  `occdrivaer-scalecorr-localized-deadend`).
- **Setup cost:** the `k` inner Pgm generates + host index compositions are **one-time/structural**;
  they must sit on the generate path, never the per-solve update path (verify via cache log, gate §4.5).
- **Prior expectation:** localized was a wash (`mergelevels-no-speedup`). Treat a null global result
  as the *likely* outcome and keep the sweep cheap (50 steps, cached, 4 variants) — this is a
  targeted confirm/deny of one hypothesis (does keeping cross-rank coarse coupling change the trade),
  not an open-ended tuning campaign.
