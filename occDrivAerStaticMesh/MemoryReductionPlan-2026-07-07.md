# occDrivAerStaticMesh — GPU peak-memory reduction plan

**Where we are.** Per rank (16.3 M cells / 49.3 M internal faces, 4 ranks): peak live **22.75 GB**,
high-water 25.3 GB, reserved pool 38.1 GB (fixed `memPoolSize`). The KOmegaSST scratch is now released
(coarse: end of `correct()`; fine-grained: at each field's last use, both gated on `devicePoolActive()`).
After the fine-grained frees the turbulence-assembly peak drops ~2 GB and the global peak reverts to the
**pressure solve (`pEqn`)**. Everything below targets that `pEqn` floor.

All figures are from the named allocation records (`*.allocRecords.txt` +
`summarize_alloc_records.py`); **measure every step the same way** — it is the source of truth.

## Peak composition (per rank, at `pEqn`)

| category | ~MB | note |
|---|---:|---|
| Mesh geometry (`GeometryScheme` + adapter: Vec3/scalar/int over faces & cells) | ~6,000 | persistent, read by every kernel |
| Persistent linear systems (p+k+ω scalar 3,003 + momentum 1,250) | ~4,250 | **all 4 co-resident all run**; each holds **CSR + COO** |
| KOmegaSST (persistent ~2,200; scratch ~3,400 during `correct()` only) | ~2,200 | scratch now released — off the `pEqn` peak |
| CFD fields (`VectorCollection` VolumeField/SurfaceField: U, p, phi, k, ω, νt, gradU, old-times) | ~2,000 | some trimmable |
| Sparsity bundles + stencils (int: CSR/COO patterns, cell↔face) | ~1,500 | shared already (`SharedSparsityBundle`) |
| Ginkgo p-solver workspace (cached MG hierarchy + Krylov vectors) | ~1,500? | not in allocRecords (Ginkgo-internal) — **verify** |
| unlisted tail | ~3,750 | ~30 smaller fields; raise the dump cap / read the full file |

## Levers, ordered by impact ÷ (effort × risk)

### 1. Phase-release the idle persistent linear systems  — **~2.5–3 GB — ATTEMPTED 2026-07-08, REVERTED (BLOCKED)**
> **Tried and it diverges.** Freeing the k/ω `LinearSystem` matrix values after their solve + regrowing at
> the next `assemble()` makes omega blow up on the FIRST post-release solve (step 2: Final residual 10.6,
> 1000 iters; step 1 with the fresh solver was fine). The `ginkgoDistributed.cpp:143` pointer-guard rebuilds
> the matrix *wrapper* on realloc, but the **persisted solver** (`readOrCreate<shared_ptr<Solver>>` for
> cacheSolver/`update_matrix_value` MG reuse) keeps preconditioner/reuse state tied to the old value buffer
> that the guard does not invalidate → stale solve. To do this safely, invalidate/rebuild the cached solver
> on the first solve after a release (partly defeats the MG cache) OR handle release at the NeoN
> LinearSystem/Solver layer so wrapper+preconditioner refresh atomically with the realloc. Needs NeoN-level
> work + a distributed convergence test. See memory `pde-linearsystem-release-breaks-ginkgo-cache`. Move
> lever 2/3/5 ahead of this.

*(original analysis, still valid as the target)*
`PDE::ls_` is `&readOrCreate<LinearSystem>("linearSystem"+psi.name)` — p, U, k, ω systems persist for the
whole run and are all resident at `pEqn`, though **only one is ever being solved** (SIMPLE is sequential:
momentum → pEqn → turbulence.correct()). `assemble()` already does `ls_->reset()` then re-fills, so the
storage is reuse-oriented; we can drop it between uses like the KOmegaSST scratch.
- **Mechanism:** after a field's `solve()`, resize its LinearSystem's CSR-values / COO-values / rhs vectors
  to 0; regrow on the next `assemble()`. Gate on the same pool check (`freeVecs` pattern) so it is a no-op
  without a pool. The KOmegaSST work is the template — consider lifting `freeVecs`/`devicePoolActive` into a
  shared `NeoFOAM::scratch` helper and reusing it in `PDE`.
- **Win:** at `pEqn`, the momentum (~1.25 GB) and k+ω (~2 GB) systems are idle → free ≈ **3 GB**.
- **Risk:** the Ginkgo solver may hold pointers into the system's matrix/rhs across the cached-solver
  lifetime (`cacheSolver`/`preconditionerRebuildInterval`). Freeing a matrix a cached preconditioner still
  references = crash. **Verify** the cache-reuse path re-points to the regrown matrix each assemble.
- **Verify:** allocRecords at `pEqn` should lose the momentum + k/ω `LinearSystem` blocks; continuity and
  p-iteration counts unchanged.

### 2. Eliminate the CSR+COO matrix duplication  — **~1–2 GB, medium-high effort, medium risk (NeoN)**
Every `LinearSystem<…, Matrix<Csr>, Matrix<Coo>>` materialises the matrix **values twice** — once COO
(assembly-friendly) and once CSR (solve-friendly) — sharing only the int sparsity (`SharedSparsityBundle`).
For the pressure system the value arrays are ~0.5–1 GB each.
- **Mechanism:** if COO is only an assembly staging buffer, free/resize-to-0 the COO values after the
  COO→CSR conversion (or assemble straight into CSR). NeoN-level change in `LinearSystem` / the assembler.
- **Win:** roughly halves matrix-value storage across all 4 systems (~1–2 GB, compounding with lever 1).
- **Verify:** confirm COO values are unused post-assembly (grep the solve path); check assembly correctness
  vs a reference run.

### 3. Ginkgo solver-workspace reuse / sharing  — **~1 GB?, medium effort, medium risk**
The cached p-solver keeps an MG hierarchy + Krylov vectors resident; U/k/ω smoothSolvers allocate their own.
Since solves are sequential, a shared scratch arena (or reusing the p-workspace bounds) avoids N copies.
See memory `neon-ginkgo-workspace-reuse-strategy3`.
- **Verify first:** it is Ginkgo-internal (not in allocRecords) — quantify with the replay reducer
  (`umpire_replay_peak.py`) which *does* see every allocation, to confirm the size before investing.

### 4. Lower `memPoolSize`  — **frees ~10 GB reserved, trivial, low risk**
High-water is now ~25.3 GB against a 38.1 GB reserved pool. Drop `memPoolSize` to ~28 GB (headroom for the
transient spikes). This does not cut *working set* but returns ~10 GB/GPU to the driver (bigger cases per
GPU, or co-tenancy). Do this **last**, after levers 1–3 lower the high-water further.

### 5. Trim persistent fields  — **~0.5–1 GB, low-medium effort, low risk**
- `gradU` (Tensor, 1.12 GB) is kept only for force objects / viscous stress. If forces are computed at write
  time, compute `gradU` on demand there instead of holding it every step.
- Old-time fields (`rotateOldTimes`): steady SIMPLE may not need a stored old-time for every field.
- **Verify** each is truly unused between the steps you free it across.

### 6. Mesh-geometry reduction  — **large pool (~6 GB) but hard, high effort/risk — defer**
Biggest single category, but every value (face centres/areas, cell centres, delta coeffs, weights) is read
by nearly every kernel; recomputing on the fly trades heavy compute for memory. Only pursue derived-quantity
elimination (e.g. `magSf` from `Sf`) after 1–5, and measure the kernel-time cost.

### 7. Mixed-precision matrices  — **~large, blocked**
float32 matrix values would ~halve the linear systems, but the float `value_type` path currently crashes in
distributed Schwarz setup (memory `neon-mixedprec-distributed-schwarz`). Track upstream; not actionable now.

## Recommended sequence

1. **Lever 1 (phase-release linear systems)** — reuses the mechanism we just built, biggest safe win (~3 GB).
   Do the Ginkgo-cache-pointer verification first.
2. **Lever 3 measurement** — run the replay reducer to size the Ginkgo workspace before deciding on it.
3. **Lever 2 (CSR/COO dedup)** — the other structural ~1–2 GB; NeoN change, so scope it after lever 1 proves
   the registry/solver interaction is safe to touch.
4. **Lever 5 (field trims)** — opportunistic ~0.5–1 GB.
5. **Lever 4 (lower `memPoolSize`)** — once high-water is down, reclaim the reserved headroom.
6. Levers 6–7 only if a hard per-GPU ceiling still forces it.

**Projected:** levers 1+2+5 ≈ **4–6 GB** off the working-set peak → ~17–19 GB/rank (from 22.75), at which
point the mesh geometry (~6 GB) and the CFD fields dominate and further gains need lever 6/7.

## Measurement protocol (unchanged, reuse each step)

```bash
./param-study-best.sh                 # MEM_ALLOC_RECORDS on (needs UMPIRE_ENABLE_BACKTRACE build)
python3 occDrivAerStaticMesh/summarize_alloc_records.py \
    paramStudyResults/best-practice/<run>.allocRecords.txt --top 30 -n 16333691
# high-water/live from the timeline; true per-rank peak + transients from the replay reducer.
```
Gate every release on `devicePoolActive()` (raw-allocator runs must stay churn-free), and confirm continuity
+ p-iteration counts are unchanged after each lever — this is validated numerics.
