# Optimization-paper parameter-study plan — occDrivAerStaticMesh

**Date:** 2026-07-08
**Case:** occDrivAerStaticMesh (driftAer static mesh, kOmegaSST, steady SIMPLE, 4× H200, NP=4)
**Goal:** a reproducible, publication-grade study that (1) establishes a semi-converged
restart baseline, (2) produces a computational-cost break-down and detailed profiling of an
unoptimized CG+Multigrid *reference* (instrumented), (3) then measures that same reference's
overhead-free time-per-timestep (all instrumentation off) as the headline number, and (4)
uses the break-down to decide, on evidence, which optimizations to sweep next.

This reuses the *design* of the existing `param-study-*.sh` harness but is kept **fully
isolated** from it so paper results never mix with the earlier studies:
- every new script is named `phaseN-paper-study-*.sh` (per-phase prefix), with the shared
  library `paper-study-common.sh`;
- all output goes to a dedicated results root **`paperParamStudyResults/`** (never the
  existing `paramStudyResults/`);
- the shared `param-study-common.sh` is **forked**, not modified, into
  `paper-study-common.sh` — so the old studies keep their exact current behavior.

The key methodological change vs. the current scripts is that every measured run starts from
a **semi-converged restart field at iteration 1000** instead of from the uniform `0/`
initial field.

---

## Why a restart baseline (motivation)

The current harness (`param-study-common.sh`) pins every run to the window `0 → STEPS=30`
from the uniform initial field and wipes all written times except `0/` (`reset_to_t0`).
The first ~30 SIMPLE iterations from a cold uniform field are an **atypical startup
transient**: the linear operator and RHS are far from what they look like in a production
run, iteration counts are inflated/erratic, and the relative cost of momentum vs. pressure
vs. turbulence is not representative. Solver-tuning conclusions drawn from that window do
not transfer to the converged regime the paper actually cares about.

**Fix:** spin the case up once to a semi-converged state (iteration 1000), freeze that
field as a restart, and run every sweep variant as a short `1000 → 1030` window from that
identical restart. Each variant then sees a **representative operator and RHS**, so
per-iteration solver cost and iteration counts are the ones that matter for the paper, and
runs remain strictly apples-to-apples (identical start field, identical window length).

---

## Phase 0 — Restart generation (spin-up)  → `phase0-paper-study-spinup.sh`

**Purpose:** produce the one semi-converged restart all later phases start from.
**Results dir:** `paperParamStudyResults/spinup/`.

- Config: the **reference** solver (fp64 PCG + global Multigrid, `p-multigrid.json`) — see
  Phase 1. Rationale: the restart must be a *clean, solver-agnostic* flow field. Spinning up
  with a heavily-optimized solver (mixed precision, aggressive cache intervals) would bake
  solver-specific error into the "ground-truth" restart and contaminate every downstream
  comparison. fp64 reference = neutral starting point.
- Run: `startTime=0`, `endTime=1000`, `writeInterval=1000` (write only the final restart).
  SIMPLE steady, `deltaT=1` (iterations). **The restart point is fixed at 1000** — not a
  tuned/swept parameter.
- Output: `processor*/1000/` across all 4 ranks — the frozen restart.
- Convergence check (for the paper's justification, not to re-choose the point): record and
  plot the residual history (U, p, k, omega) and the force coefficients (`neoForceCoeffs`,
  GPU-native — OF function objects can't run with neoSimpleFoam due to NO_REGISTER fields)
  over the 1000 iterations, confirming the field at 1000 is *semi*-converged (residuals
  plateauing, forces settling) — not fully converged, so the sweeps still exercise the
  solver on a non-trivial correction, and not so early that we are back in the transient.
- Guard: this run is **excluded** from `reset_to_restart`; the harness preserves
  `processor*/1000/` and only wipes intermediate/later times.

**Forked harness — `paper-study-common.sh` (copy of `param-study-common.sh` with):**
1. `RESULTS` root set to `paperParamStudyResults/<STUDY_TYPE>/` (never `paramStudyResults/`).
2. `RESTART=1000` (fixed) and `STEPS=${STEPS:-30}`.
3. Replace the fixed `startTime 0 / endTime STEPS` pinning with
   `startTime=$RESTART`, `endTime=$((RESTART+STEPS))`, `writeInterval=$((RESTART+STEPS))`
   for every measured run (Phases 1–3). Phase 0 is the exception (it writes `RESTART`).
4. Rename `reset_to_t0` → `reset_to_restart`: drop every written time dir **except `0/` and
   `$RESTART/`** (keep the restart; still keep `0/` for provenance), so each variant marches
   the identical `RESTART → RESTART+STEPS` window from the frozen field.
5. A `require_restart` preflight: abort with a clear message if `processor0/$RESTART/`
   is missing, telling the user to run `phase0-paper-study-spinup.sh` first.

The original `param-study-common.sh` is left untouched, so the earlier studies are unaffected.

---

## The reference config (used by Phases 1 and 2)

Both phases run the identical **reference** solver — only the instrumentation differs:

- Config: `system/gko/p-multigrid.json` — fp64 PCG (`Cg`) with a **global** (non-localized)
  Multigrid *preconditioner*, Pgm coarsening, `max_levels=10`, default smoother, **no**
  solver caching, **no** scale-correction, **no** mixed precision, **no** PMIS. This is the
  `base` variant of `param-study-mg.sh` and is deliberately the plainest correct CG+MG stack.
- Run from restart: window `1000 → 1030`, NP=4, `NEON_BUILD=profiling` (runs every variant
  reliably; the production `-O3` build still segfaults in distributed Schwarz setup).

**Order matters — profiling first, clean timing second.** All the instrumentation below
(`space-time-stack`, the memory tools, `NEOFOAM_MEM_TIMELINE`, `nsys`) adds real wall-clock
overhead and perturbs the per-timestep cost. So we **collect the cost break-down and detailed
profiling data first (Phase 1)**, then run the reference **again with every instrument off
(Phase 2)** to get the true, overhead-free time-per-timestep that becomes the paper's headline
number and the speedup denominator.

---

## Phase 1 — Cost break-down & detailed profiling of the reference  → `phase1-paper-study-costbreakdown.sh`

**Purpose:** decompose where the reference spends its time, so optimization targets are
chosen from evidence, not intuition. This phase is **allowed** to carry profiler overhead —
its outputs are *relative* attributions, not the headline wall time.
**Results dir:** `paperParamStudyResults/cost-breakdown/`.

Run the reference config under each instrument below, from the restart, window `1000 → 1030`.
Reuse the existing tooling — no new profilers needed:

1. **Region wall-time attribution (primary).**
   `KOKKOS_TOOL=space-time-stack` (already wired through the common harness /
   `kokkos_launch`, inherited by the fork). Report the per-region tree: `momentumPredictor`, pressure-equation
   **assemble** vs. pressure **solve**, turbulence (k/omega), field I/O, halo exchange.
   Cross-check against the `NF_MEM_SCOPE` / MemoryProbe regions already annotated in
   `neoSimpleFoam` + `pde` + `kOmegaSST`.
   *Prior evidence to confirm/refute on the restart field:* momentumPredictor ≈ 34%
   (uncached U-solver regen), `momentum.assemble` host "remainder" 88–93% =
   un-pooled Umpire raw `cudaMalloc/cudaFree` per temporary, pressure solve the remainder.
   The semi-converged restart may shift these ratios materially vs. the cold-start numbers —
   that is exactly why we re-measure here.

2. **GPU-vs-host split.** From the same space-time-stack report: fraction of wall time in
   Kokkos kernels vs. host/MPI/allocator. Prior cold-start figure was Kokkos ≈ 4.8%
   (host/overhead-bound). Confirm on the restart.

3. **Solver-cost isolation.** Per-solve pressure ms and iteration count (from the log) ×
   solves/iteration → pressure-solver share of an iteration. Same for U, k, omega.

4. **Memory footprint (secondary, for the memory story).**
   `KOKKOS_TOOL=memory-high-water-mark` and the `NEOFOAM_MEM_TIMELINE=1` CSV +
   `plot_memory_timeline.py` region ranking. Establishes the reference peak and the
   per-region allocation timeline the memory-reduction plan builds on.

5. *(Optional deep dive)* `LAUNCH_WRAPPER="nsys profile -o ..."` for a kernel-level trace of
   the single hottest region identified in (1).

**Deliverable:** a `PaperPerformanceReport-<date>-reference-breakdown.md` with (a) a
wall-time pie/table by region, (b) the GPU/host split, (c) the per-equation solver-cost
table, and (d) the memory HWM + timeline. This is the figure the "where does the time go"
section of the paper is built on.

---

## Phase 2 — Clean reference performance (overhead-free timing)  → `phase2-paper-study-reference.sh`

**The paper's headline number and the denominator every speedup is measured against.**
Run **after** Phase 1, because this run must carry **zero instrumentation overhead**.
**Results dir:** `paperParamStudyResults/reference/`.

- Same reference config and `1000 → 1030` restart window as Phase 1, but with **all profiling
  OFF**: `KOKKOS_TOOL` empty (no `space-time-stack`/memory connector), `NEOFOAM_MEM_TIMELINE`
  unset, no `LAUNCH_WRAPPER`/`nsys`, no `NF_MEM_SCOPE` memory-probe forwarding. The script must
  assert these are unset so a stray env var can't silently taint the headline timing.
- Primary metric: **time per timestep** = total ExecutionTime / number of SIMPLE iterations in
  the window (both already in the log). Also report the per-iteration pressure/U/k/omega iters
  and continuity error already parsed by `print_summary`.
- Repeat **N=3** and report mean ± spread — GPU/host timing on this case is host/overhead-bound,
  so single-shot wall times are noisy.
- This clean time-per-timestep, **not** any Phase-1 instrumented wall time, is the speedup
  denominator throughout the paper.

---

## Phase 3 — Optimization decision point (discussion, gated on Phases 1–2)

**Do not pre-commit the sweep list.** Rank the cost contributors from the Phase-1
break-down, then pick sweeps to attack the largest ones. Each candidate maps to an existing
study whose *logic* we reuse — but for the paper it is re-run as a `paper-`-prefixed wrapper
that sources `paper-study-common.sh` (restart window, `paperParamStudyResults/` root),
so the earlier `paramStudyResults/` runs are never touched or overwritten:

| If the break-down says the bottleneck is…            | Candidate optimization / existing tool                                   | Expected lever |
|------------------------------------------------------|--------------------------------------------------------------------------|----------------|
| Pressure **solve** dominates                         | Multigrid tuning: level sweep (`param-study-mg-level-sweep.sh`), coarse-solver (`param-study-mg-coarse-solve.sh`), PMIS vs Pgm coarsening (`param-study-mg-pmis.sh`) | fewer iters / cheaper V-cycle |
| Pressure **solver setup / regen** dominates          | Solver + preconditioner **cache** reuse (`param-study-mg.sh cache-sweep`, `preconditionerRebuildInterval`) | amortize MG hierarchy build |
| **momentumPredictor / U-solver regen** dominates     | U-solver persistence / `cacheSolver` (Layer A fix), already partly landed  | kill per-iteration regen |
| **assemble host "remainder"** (allocator churn) dominates | `allocator=UmpirePool` + `memPoolSize`, kill assemble temporaries        | remove raw cudaMalloc/Free sync |
| Overall arithmetic-bound in pressure                 | Mixed precision (`param-study-production-mp.sh`, float MG preconditioner) | fp32/bf16 throughput |
| Time-to-solution / accuracy trade                    | Tolerance sweep (`param-study-production-tol.sh`)                          | iters vs. accuracy |
| Peak memory                                          | Memory-reduction plan (`MemoryReductionPlan-*.md`)                         | HWM reduction |

**Known caveats to carry into the discussion (from prior runs):**
- Ginkgo MG `scale_correction` + localized Schwarz = 9× slower — never combine
  (`occdrivaer-scalecorr-localized-deadend`). `localized-L10 + cache` is currently best.
- Localized PMIS loses to Pgm (denser coarse grids); global PMIS untestable (no distributed
  path).
- Mixed-precision + distributed Schwarz crashes at setup (float `gko::as` NotSupported on
  fp64 distributed matrix; bf16 not in Schwarz `value_type_list`).
- Freeing a persistent `LinearSystem`'s matrix values between solves diverges omega with a
  cached solver (stale state) — needs NeoN-level cache invalidation, not a PDE-level release.

**Output of Phase 3:** a short ranked decision list ("optimize X next because it is Y% of
reference wall time, expected lever Z, tool W") that seeds the paper's production sweeps
(`phase3-paper-study-production*.sh`), all run from the iteration-1000 restart and writing to
`paperParamStudyResults/`.

---

## Execution order & deliverables

1. `phase0-paper-study-spinup.sh`  → `processor*/1000/` restart + convergence-justification plot.
2. `phase1-paper-study-costbreakdown.sh`  (instrumented) → `PaperPerformanceReport-<date>-reference-breakdown.md`.
3. `phase2-paper-study-reference.sh`  (clean, no instrumentation) → time-per-timestep (N=3), the speedup denominator.
4. **Decision doc** (Phase 3) ranking optimizations from the break-down.
5. Re-run the relevant sweeps as `paper-`-prefixed wrappers from the restart, feeding the
   paper's results tables.

Profiling (step 2) is deliberately collected **before** the clean timing (step 3) so the
headline time-per-timestep never carries instrumentation overhead.

All logs/sidecars land under `paperParamStudyResults/<phase>/`; all reports are
`Paper*-<date>.md`. Nothing writes to `paramStudyResults/` or the existing scripts.

## Harness work summary (the only new code — all `paper-`-prefixed, nothing existing modified)

- **New `paper-study-common.sh`:** fork of `param-study-common.sh` with the
  `paperParamStudyResults/` root, `RESTART=1000`, restart-window pinning, `reset_to_restart`
  (keep `0/` + `$RESTART/`), and the `require_restart` preflight.
- **New `phase0-paper-study-spinup.sh`:** Phase 0 (writes the restart; the one script that
  does *not* use restart-window pinning).
- **New `phase1-paper-study-costbreakdown.sh`** (run first): loops the reference config over
  `KOKKOS_TOOL ∈ {space-time-stack, memory-high-water-mark}` + `NEOFOAM_MEM_TIMELINE=1`,
  collecting sidecars via the inherited `collect_kokkos_output`.
- **New `phase2-paper-study-reference.sh`** (run after profiling): thin wrapper running the
  `base` (`p-multigrid.json`) config from the restart with **all instrumentation forced off**
  (asserts `KOKKOS_TOOL`/`NEOFOAM_MEM_TIMELINE`/`LAUNCH_WRAPPER` unset), N=3, for the
  overhead-free time-per-timestep.
- Phase-3 sweeps become `paper-`-prefixed wrappers that source `paper-study-common.sh`.
- The existing `param-study-*.sh` scripts and `paramStudyResults/` are **left untouched**.
