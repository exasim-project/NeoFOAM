# Phase-3c — Mixed-precision sweep for the multigrid preconditioner (occDrivAerStaticMesh)

**Date:** 2026-07-20
**Case:** occDrivAerStaticMesh, kOmegaSST, steady SIMPLE, 4× H200 (NP=4, one rank/GPU)
**Baseline:** the corrected champion (§4.15) — global Cg + Multigrid, MG-level scale-correction
**post**-pass (`NEON_MGSC_MODE=post`), `max_levels=6`, coarse rel-tol 0.1, `pgmMerge2`, **cached**
(`cacheSolver=true`, `preconditionerRebuildInterval=0`).
**Build:** `profilingnvidia_h200` (RelWithDebInfo). The production `-O3` build segfaults in
distributed-Schwarz setup, so all arms use the profiling build. Since every arm shares one build,
the **relative** comparison is valid; and the **pressure-solve** numbers are production-grade anyway
(Ginkgo is always compiled `-O3`; fp64 here = **152 ms**, matching §4.15's production **151 ms**).
**Driver:** `phase3-paper-study-mp.sh`, 50-step window from the iteration-1000 restart.

Every arm is the identical champion config, changed **only** by a Ginkgo per-node `value_type`
override, so any delta is precision, not structure.

---

## Result

| arm | value_type override | steps | p̄-iters | **p ms/solve** | continuity | s/step |
|---|---|---:|---:|---:|---:|---:|
| **fp64** (baseline) | — | 50 | 8.4 | **152** | 2.91e-6 | 3.512 |
| **mgFloat32** | `preconditioner.value_type=float32` | 50 | 8.4 | **324** | 2.91e-6 | 3.841 |
| **coarseFloat32** | `coarsest_solver.value_type=float32` | 50 | 8.4 | **150** | 2.91e-6 | 3.504 |
| **coarseBf16** | `coarsest_solver.value_type=bfloat16` | 1 | — | **abort** | — | — |

*(s/step is inflated ~4× vs §4.15's 0.79 s because this is the RelWithDebInfo build — the host-side
assembly/step work is slower; the GPU/Ginkgo pressure solve is not. Read `p ms/solve` for the real
signal.)*

---

## Findings

### 1. Whole-MG float32 is a 2.1× LOSS, not "free" — and convergence is untouched
`mgFloat32` runs the entire V-cycle (smoothers, Galerkin operators, coarse solve) in float32 with the
outer Cg in fp64. Convergence is **identical** to fp64 — same 8.4 mean pressure iterations, same
continuity (2.91e-6) to 6 significant figures — so the reduced precision costs nothing numerically.
But each pressure solve **doubles**, 152 → 324 ms, stable from the first solve (not a warmup/rebuild
artifact; caching is engaged in both). 

**Isolation (8-step controls, same restart) — the 2× is intrinsic to global float32, NOT a config artifact:**

| config | p̄-iters | p ms/solve |
|---|---:|---:|
| fp64 champion | 8.2 | 148 |
| float32 (sc + coarse Cg) | 8.2 | 320 |
| float32, **scale_correction OFF** | 16.4 | 358 |
| float32, **simple fixed coarse (Ir8)** | 9.4 | 372 |
| **fp64, simple fixed coarse (Ir8)** | 9.4 | **175** |

The decisive pair is the last two: **identical config, identical 9.4 iterations, only `value_type`
differs → 175 vs 372 ms (2.1×)**. So the slowdown is *not* scale-correction (turning it off *doubles*
iterations 8.2→16.4 — sc does real work even in float32) and *not* the iterative coarse Cg. It is
intrinsic to running the **global distributed** MG in float32.

**Mechanism (amortization fit — the penalty is the per-apply cost, NOT the refresh).** Holding the
preconditioner fixed and varying only the outer iteration count (via pressure relTol) decomposes the
p-solve into a fixed per-solve *refresh* and a per-iteration *V-cycle apply*, `p_ms = R + iters·V`:

| precision | R (refresh, fixed/solve) | V (per-iter apply) |
|---|---:|---:|
| fp64 | 45 ms | **12.6 ms/iter** |
| float32 | 56 ms | **32.0 ms/iter** |

(Fit from (3 it, 83/152 ms) and (17.7 it, 268/622 ms); predicts the champion's 8.2-it solves at 148 /
318 ms, matching measured 148 / 320.) **The refresh is essentially precision-insensitive (45→56 ms); the
2.5× blow-up is entirely in the per-V-cycle apply (12.6→32.0 ms/iter).** So it is *not* an operator
re-conversion at refresh — it is the **distributed V-cycle apply**. The global MG does distributed
operations at every level of every V-cycle (SpMV + halo exchange + sc Allreduce dots); running a
`Multigrid<float>` preconditioner inside the fp64 Cg on a *distributed* operator pays a per-apply,
per-level cost at the fp64↔float32 boundary (halo/vector precision handling + a less-efficient
mixed-precision distributed SpMV path), which the memory/sync-bound solve cannot hide.

The **localized** V-cycle, by contrast, is per-rank **local** — no distributed ops inside it (only the
outer Cg exchanges halos) — so its apply is precision-insensitive (**10 ms/iter in both fp64 and
float32**, matrix converted once by the Schwarz patch). That is exactly why localized float32 is neutral
while global float32 loses 2×: float32 inflates the *distributed* per-level apply, which the global MG
does everywhere and the localized MG does not do at all.

**Global-vs-localized 2×2 control (8-step, same restart, this build):**

| structure | fp64 | float32 | float32 effect |
|---|---:|---:|---|
| **Global** (champion MG) | 148 ms @ 8.2 it | 320 ms @ 8.2 it | **2.1× slower** |
| **Localized** (Schwarz{MG}, L10, no sc) | 216 ms @ 21.5 it | 228 ms @ 21.6 it | **neutral (~parity)** |

The localized structure runs a **per-rank local** V-cycle — no distributed operations inside it — so its
per-iteration apply is precision-insensitive (**~10 ms/iter in both fp64 and float32**) and localized
float32 lands at parity (228 vs 216 ms, same 21.6 iters). The global MG does **distributed** ops (SpMV +
halo) at every level, and float32 inflates that distributed per-level apply 2.5× (12.6→32 ms/iter, see
the amortization fit above) — hence the 2× loss. The difference is the *distributed per-level apply*, not
operator conversion at refresh (which is ~precision-flat, 45→56 ms).

**Correction to the prior "float faster" note (§3.3).** On this corrected build localized float32 is
**parity, not faster** (the old ~516 vs ~587 ms win was on the pre-oversubscription-fix build — a small,
non-reproducible effect). The robust finding: float32 **avoids the penalty on localized but does not win,
and is a clear 2× loss on global.** fp64 global champion (148 ms) stays the best cell regardless.

*(Aside: the localized config built with the champion's MG-level `scale_correction:true` **diverges in
float32** — 398 iters — because per-rank scale-correction is inconsistent across subdomains, §5. sc must
not be combined with the localized/Schwarz structure.)*

### 2. Coarse-only float32 is neutral — free, but pointless
`coarseFloat32` runs *only* the coarsest solve (the distributed `Schwarz{Cg+Jacobi}`) in float32.
It **works** (no setup crash — the distributed-Schwarz float32 path is valid on this build), converges
identically (8.4 iters, 2.91e-6), and the p-solve is within noise of fp64 (150 vs 152 ms). The coarse
solve is too small a fraction of the V-cycle for its precision to move the headline either way. Safe to
adopt but buys nothing.

### 3. Coarse-only bfloat16 is BLOCKED at setup
`coarseBf16` aborts at solver construction:
```
gko::InvalidStateError: dispatch.hpp:73: The provided runtime type >bfloat16<
doesn't match any of the allowed compile time types.
```
Even with `GINKGO_ENABLE_BFLOAT16=ON` in the build, the coarse solver's dispatch path was not
instantiated for bfloat16 — confirming the standing block in `[[neon-mixedprec-distributed-schwarz]]`.
Unblocking this is a Ginkgo-side template-instantiation change, not a config change.

---

## Recommendation

**Keep the champion in fp64.** None of the three mixed-precision levers helps the pressure MG
preconditioner at the current operating point: whole-MG float32 doubles the solve (conversion overhead
on a sync-bound V-cycle), coarse-only float32 is a numerical no-op, and coarse-only bf16 does not build.
The mixed-precision strategy (3.3) is closed as a **negative result** for this case. The remaining
pressure lever stays the one from §4.15.3: cut the per-solve coarse-operator refresh (`update_matrix_value`),
which is fp64-bound and independent of any precision choice.

*Data:* `paperParamStudyResults/mixed-precision/{fp64,mgFloat32,coarseFloat32,coarseBf16}-20260720-*.log`;
driver `phase3-paper-study-mp.sh`; configs `system/gko/mp-{fp64,mgFloat32,coarseFloat32,coarseBf16}.json`.

---

## Appendix — build/case fixes required to run the current `enh/kOmegaSST` HEAD on this case

Rebuilding `neoSimpleFoam` from the current WIP branch surfaced several incompatibilities with the
established paper-study harness; each was fixed to get the study running (all small, matching existing
idioms):

| # | Symptom | Fix | File |
|---|---|---|---|
| 1 | compile error `inlineWeightKernel()` too few args | pass `flux_` (dead-code operator) | `NeoN gaussGreenDdtDivLaplacian.cpp` |
| 2 | OpenFOAM parse error on `.mp-*.json` dotfile path | generate non-dot `mp-*.json` | `phase3-paper-study-mp.sh` |
| 3 | field read: `Unsupported BC type 'slip'` | stale binary predated `slip` support → rebuild | (rebuild) |
| 4 | fused div-lap operator rejects `bounded … linearUpwindV` | `optimize false` (use separate div/lap operators) | `fvSolution.case3` |
| 5 | `cacheSolver` read as bool → `bad_any_cast` on OpenFOAM switch | `readSwitch()` helper (bool/int/string) | `NeoN ginkgo.hpp` |
| 6 | `cacheSolver` leaks into Ginkgo config validator | strip it + `preconditionerRebuildInterval` in `parse()` | `NeoN ginkgo.cpp` |

Fixes #5 and #6 are genuine NeoN bugs (uncommitted). #4 is a real limitation of the fused
`GaussGreenDivLaplacian::read()` — it parses only `Gauss upwind`/`Gauss linearUpwind`, choking on the
`bounded` prefix and the `linearUpwindV` vector variant that the production case uses.

**Follow-up on #4 (parser fix + validation, uncommitted):** `GaussGreenDivLaplacian::read()` was patched
to strip a leading `bounded` token and accept `linearUpwindV` (SurfaceInterpolation already registers
that name for the cell-limited variant). This **removes the crash** — the fused path now parses the
production scheme and momentum solves (Ux, 2 iters). **But the fused path then DIVERGES the pressure
solve** (initial residual 0.046 vs 0.0013 non-fused, 1000 iters, residual growing), because the operator
does **not** implement the `bounded` boundedness source it now silently accepts (and its `linearUpwindV`
deferred correction differs). So the parser fix is correct but insufficient: the fused operator needs the
bounding term (and a validated linearUpwindV correction) before it is usable on this production case.
**`optimize false` (separate div/laplacian operators, which bound correctly — the non-fused pressure
converges at 8 iters) remains the required setting here.** The case is left at `optimize false`.
