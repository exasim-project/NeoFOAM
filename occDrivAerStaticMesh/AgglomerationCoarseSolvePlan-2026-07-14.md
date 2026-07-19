# Plan — rank-agglomeration (redundant) coarse solve for the global Multigrid

**Date:** 2026-07-14
**Case:** occDrivAerStaticMesh (kOmegaSST, steady SIMPLE, global Cg+Multigrid + MG-level sc + pgmMerge3)
**Motivation:** PaperOptimizationStudy §4.11h/i — the coarse-level halo exchanges are latency-bound
(sub-1KB data, ~1 ms/call, 0 % comm/compute overlap) because the coarse operator is **communication-
dense** (surface-to-volume ratio → ~40 % of coarse unknowns couple cross-rank vs ~1 % at the fine
level; §"why coarser levels are comm-dense"). The §4.11i **localized** coarse solve removes that comm
by solving each rank's block independently — zero comm, and *zero convergence penalty at 4 ranks
because the coarse coupling is weak there*. **Rank-agglomeration is the scale-robust alternative:** it
preserves the exact global coarse coupling (so it never loses convergence, unlike localized) while
replacing the many per-iteration coarse halos with **one small `allgather` per solve**.

---

## 0. When to build this (do NOT build for the 4-rank case)

**Decision from §4.11i data:** localized-coarse and global-coarse both give **9 pressure iters** at
4 ranks → the cross-rank coarse coupling is irrelevant to convergence here → agglomeration would also
give 9 iters, i.e. **parity with the localized-coarse solve but with an extra allgather it doesn't
need.** Agglomeration is *strictly dominated by localized at 4 ranks.*

**Build this only for a SCALE study.** It earns its keep where localized starts *losing* iterations:
many ranks (surface-to-volume keeps growing → localized ignores an increasingly significant coupling)
or a stronger/less-diagonal coarse operator. The deliverable of building it is the **crossover point**
— the rank count at which `redundant` beats `localized`. Target a 4 → 16 → 64 → 256-rank sweep.

---

## 1. What Ginkgo provides (and doesn't)

- **No built-in redundant/agglomerated coarse solver** (checked `core/`, `include/`): only low-level
  `mpi::gather` / `all_gather` (`include/ginkgo/core/base/mpi.hpp:930+`). So this is a **custom LinOp**,
  not a config switch.
- The distributed coarse operator is a `experimental::distributed::Matrix<value, local_idx, int64>`
  with `get_diag_matrix()` (local Csr block) + `get_off_diag_matrix()` (cross-rank coupling Csr) and an
  `index_map` (non-local col → global col). Row partition is block-wise by rank
  (`build_partition_from_local_size`; rank r owns `[offset_r, offset_r+n_r)`), same as everywhere else.
- The coarsest_solver is consumed by `Multigrid` as a `LinOpFactory` whose `generate()` is called on the
  distributed coarse op; its `apply(b, x)` runs each V-cycle bottom. So the custom solver plugs in there.
- Cache path: the coarse op **values** refresh each solve via the `update_matrix_value` chain (Pgm /
  MergedPgm). The agglomerated solver must implement `UpdateMatrixValue` too (structure fixed, values
  re-gathered) or caching (§4.9) breaks.

---

## 2. Design — `RedundantCoarseSolver` (a Ginkgo `LinOp`)

A drop-in coarsest_solver that gathers the tiny distributed coarse system to a **replicated** local Csr
on every rank and solves it locally (no per-iteration comm). Mirrors the `mergedPgm.hpp` idiom
(NeoN-side header, `EnableLinOp` + `UpdateMatrixValue`, registered by name).

**generate(coarse_dist_matrix)  — once per (re)build:**
1. From the distributed coarse op get: `diag = get_diag_matrix()` (local Csr, global rows
   `[offset_r, offset_r+n_r)`), `off_diag = get_off_diag_matrix()` (Csr, cols in non-local index space),
   the `index_map` (non-local col → **global** col), the communicator, and `n_r` per rank.
2. Re-express both blocks in **global (row, col)** COO on device→host: diag cols are already global
   (add `offset_r`); off-diag cols map through the index_map to global. Result: this rank's contribution
   to the global coarse matrix as `(gr, gc, val)` triples.
3. `allgatherv` the triple arrays across the communicator → every rank holds ALL triples → build one
   **replicated global Csr** `A_glob` (size `N × N`, `N = Σ n_r`, e.g. 64) identical on every rank.
4. Build a **local solver** on `A_glob` (see §2.1). Store `offset_r`, `n_r` for the scatter, and the
   partition sizes for the RHS allgatherv (`recvcounts = {n_0,…,n_{P-1}}`).

**apply(b, x)  — each V-cycle:**
1. `allgatherv` b's local values (`n_r` per rank) → replicated global RHS `b_glob` (size N) on every rank.
   *(One collective — the whole point.)*
2. `local_solver->apply(b_glob, x_glob)` — every rank solves the SAME global system **redundantly, with
   no communication.** Deterministic ⇒ bit-identical `x_glob` on all ranks.
3. Scatter: `x_local = x_glob[offset_r : offset_r + n_r]` (this rank's owned rows) → distributed x.

**update_matrix_value(new_coarse_dist_matrix):** structure is frozen from generate; only re-gather the
**values** (step 2–3 with the new diag/off-diag values into the same global-Csr sparsity) and refresh
the local solver. Keeps `p-cache: reuse(update_matrix_value)` engaged.

### 2.1 Local solver choice
- **Direct (recommended for a truly exact coarse solve):** dense LU / `gko::experimental::…LU` on the
  tiny `A_glob` — exact, so convergence matches the global coarse solve exactly. Refactor on update
  (cheap at N≈64; measure).
- **Iterative (simpler, matches existing config):** `Cg(k) + Jacobi` on `A_glob` — no factorization to
  refresh, just point at the new values. Use if LU refresh proves fiddly. k small (converges fast at N≈64).
Start with Cg+Jacobi (lower risk), switch to direct only if the coarse residual limits outer convergence.

---

## 3. Implementation steps

1. **Header** `NeoN/include/NeoN/linearAlgebra/ginkgo/redundantCoarseSolver.hpp` (new), modelled on
   `mergedPgm.hpp`: `EnableLinOp<RedundantCoarseSolver<Value,Index>>` + `UpdateMatrixValue`; factory
   param = the local-solver spec (or fix it internally for the prototype). Guard distributed code with
   `#ifdef NF_WITH_MPI_SUPPORT`; `dist_mtx = Matrix<Value,Index,gko::int64>` (as MergedPgm).
2. **Gather helpers**: `allgatherv` for (a) the global-index COO triples (generate), (b) the RHS/solution
   vectors (apply). Use `comm.all_gather` / manual `MPI_Allgatherv` via the Ginkgo `mpi::communicator`.
   Host-buffer path if not GPU-aware (mirror `RowGatherer`/`ginkgoDistributed.cpp`).
3. **Global Csr assembly** on host from the gathered triples (sort by row, dedup/accumulate duplicate
   `(gr,gc)` — the diag/off-diag split can produce both), clone to device. (Same host-assembly trick as
   the distributed MergedPgm injection, §MergedPgm impl note — avoids device SpGEMM.)
4. **Registration** (named-registry, same as MergedPgm / L1 criterion): in `ginkgo.hpp` GinkgoSolver
   ctor, `reg.emplace("neon::redundantCoarse", makeRedundantCoarseFactory<scalar>(exec))`.
5. **Config**: `p-multigrid-mgsc-m3-L5-c4-agg.json` = the L5c4 config with
   `"coarsest_solver": "neon::redundantCoarse"`.
6. **Build** against the pinned fork (`scripts/coma/build-nvidia-h200-gcc.sh`).

---

## 4. Correctness gates (before any perf/scale run)

The crux risk is the **distributed index bookkeeping** (mapping off-diag cols to global, the two
allgatherv patterns) — exactly the class of bug that sank the batched-sc-dots prototype (§4.11g). Gate
on a **small** distributed case (2–4 ranks, pitzDaily-scale) first:
1. **Replicated matrix is bit-identical across ranks** — hash `A_glob` on each rank, assert equal.
2. **Gathered A_glob == the true global coarse matrix** — compare `A_glob·v` (local) to the distributed
   coarse op `A_coarse·v_dist` (global) on a random v; norms must match.
3. **Redundant solve == distributed coarse solve** — `RedundantCoarseSolver.apply(b)` vs the baseline
   `Ir+Schwarz` (or a distributed Cg) coarse solve on a random b; match to tolerance.
4. **k=1 sanity**: with a diagonal coarse op, output == b/diag.
5. **Cache reuse**: 2nd solve logs `reuse(update_matrix_value)`; iters/residual match a fresh generate.

Only then run occDrivAer.

---

## 5. Scale sweep (the actual deliverable)

From the restart, 50 steps, cached, **compare three coarse solvers at each rank count**:
`{global (Ir+Schwarz), localized (Schwarz(Cg)), redundant (agglomeration)}` × NP ∈ {4, 16, 64, 256}.

Per cell record: **p̄-iters** (does localized start losing vs redundant/global as NP grows?), **s/step**,
**coarse-collective count** (nsys: sub-1KB `Alltoallv` + the single agglomeration `Allgatherv`), **cont**.
**The headline is the crossover** — the NP where `redundant` s/step drops below `localized` (localized's
rising iteration count finally outweighs redundant's one allgather). Expected shape: at NP=4 redundant ≈
localized (§4.11i, both 9 iters); as NP grows, localized's iters climb (coupling it ignores grows with
surface-to-volume) while redundant holds the global iteration count at the cost of one (growing)
allgather → redundant wins beyond some NP.

---

## 6. Caveats / risks to carry in

- **Parity at 4 ranks is expected, not failure** — the value is the *scale* crossover (§0). Do not judge
  the prototype by the 4-rank number.
- **The allgather grows with NP** (gathers N=Σn_r values to all P ranks → O(N·P) traffic + a P-way
  collective). At extreme scale the *redundant-to-all* pattern itself becomes a bottleneck; the
  production form is **gather-to-a-subset** (a coarse sub-communicator of √P ranks) — a later refinement.
  For the prototype, gather-to-all is fine up to O(100) ranks.
- **Index bookkeeping is the correctness risk** (§4 gates) — mirror the distributed-MergedPgm de-risking
  (prototype the gather on a tiny case, verify against the distributed apply) before trusting it.
- **Local-solver refresh cost** (LU refactor per update) must sit on the update path, not per-apply;
  verify via the cache log. Cg+Jacobi avoids this.
- **Do not combine with the localized-coarse config** — they are mutually exclusive coarsest_solvers.
- Effort: ~150–250 lines (custom LinOp + 2 allgatherv patterns + host Csr assembly + update path +
  registration), medium-high, real distributed-correctness risk. Build against the pinned fork.

**Related:** §4.11i (localized-coarse, the 4-rank incumbent), the surface-to-volume analysis,
`mergedPgm.hpp` (the distributed-LinOp + UpdateMatrixValue + named-registry idiom to copy),
`[[mglevel-scalecorrection-wins]]`, `[[l5c4-sync-attribution]]`.
