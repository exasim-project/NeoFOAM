<!--
SPDX-License-Identifier: GPL-3.0-or-later
SPDX-FileCopyrightText: 2026 NeoFOAM authors
-->

# One engine, many studies — a paramspace-native composable workflow

**Thesis (yours, sharpened):** the `neofoam.tooling.verification` *stack* is not
needed. A verification study is a **parameter sweep** (`paramspace`) plus a **DAG of
a few steps** — the same two things a mesh sweep is. If we name the right
abstraction, a new study is *a `params`/`discover` + a few step bodies*, and both
existing studies (mesh sweep, verification) fall out as two selections over one
engine.

This doc is the synthesis of the two siblings next to it:

- `simplification-plan.md` — the conservative on-disk-rules split. **Mostly done**
  (the study now carries `rules/*.smk` and `discover.py` enumerates only). It made
  verification *readable*; it did not *unify* it with `workflow`.
- `removing-verification-tooling.md` — "Proposal B": generalize `RuleKind` → a
  `Scope` enum + topological wiring. Right direction. This doc argues the `Scope`
  enum is **one notch too specific**, and that the notch it's missing is exactly the
  thing your "just paramspace" instinct points at.

---

## 1. What the two studies actually are (side by side)

Strip the domain code and both are the same skeleton — a **fan-out**, a **chain per
fanned-out row**, a **fan-in**:

```
mesh sweep        setup_mesh → blockMesh → snappyHexMesh → checkMesh   (once per MESH variant)
                                                              └─ setup → solve       (once per CASE)
                                                                            └─ all   (fan-in: all cases)

verification      build_case → swap_solver → run                       (once per (CASE, SOLVER))
                                                └─ compare              (fan-in: a case's solvers)
                                                        └─ report       (fan-in: all cases)
```

Look at the *keying* under each step — that is the whole story:

| step | runs once per… | in paramspace terms |
|---|---|---|
| `blockMesh`, `checkMesh` | mesh variant | keyed on `{mesh}` (shared across cases) |
| `setup`, `solve` | case | keyed on `{case}` |
| `build_case`, `swap`, `run` | (case, solver) | keyed on `{case, solver}` |
| `compare` | case, gathering its solvers | keyed on `{case}`, **consumes** `{case, solver}` |
| `report`, `all` | once | keyed on `{}` (the sink) |

Everything a step needs to be wired is in the middle column: **which subset of the
sweep's columns it is keyed on.** That is the abstraction.

---

## 2. The abstraction: a step is keyed on a *projection* of the sweep

A sweep is a table of rows (paramspace already: `sweep.csv` × `params.yaml`). Its
columns are the **dimensions** (`case`, `solver`, `mesh`, `cad`, …). One idea:

> **A step declares the tuple of dimensions it is keyed on (`over`). It runs once per
> distinct value of that projection. It is wired to another step by matching what it
> `produces` to what it `needs` — and the *relationship between the two projections*
> decides the wiring automatically.**

Three relationships, and they are *all three wirings the codegen already hardcodes
for meshes* — now derived instead of special-cased:

```
needs a step with the SAME projection        → chain     (build → swap → run)
needs a step with a COARSER projection        → 1:1 join, routed by variant_of
     on a different axis                        (setup on {case} needs the {mesh} it selects)
needs a step with a FINER projection          → fan-in    (compare on {case} gathers {case,solver};
                                                            report on {} gathers {case})
```

That is the complete wiring calculus. `expand()` for fan-in, a `variant_of` lambda
for the cross-axis join, a literal wildcard path for the chain — **the exact three
shapes `sweep_snakefile` emits today**, only chosen by comparing `over` tuples rather
than branching on `RuleKind`.

### Why this beats the `Scope` enum in the sibling doc

Proposal B offers `Scope ∈ {per_case, per_variant, sink}`. But verification's rows
are keyed on **(case, solver)** — a *cross-product*, not "per_case." Proposal B has
to thread `solver` in as an ad-hoc extra wildcard *outside* the scope enum (which is
precisely what today's study `Snakefile` does with its hand-rolled `SOLVER_LABELS`).
The enum can't name the row.

Under projections there is no special case: **`solver` is just another sweep
dimension**, and its "variant" is the app to swap in. `over=(case, solver)` names the
row directly. The model gets *smaller* by getting *more general* — and it lands
exactly where you pointed: it is **all paramspace**.

```python
@dataclass(frozen=True)
class Step:
    name: str                        # "build_case", "blockMesh", "compare"
    body: str                        # packaged .smk file (the domain-specific bit)
    over: tuple[str, ...] = ()        # projection: () = sink, ("case",), ("case","solver"), ("mesh",)
    needs: tuple[str, ...] = ()       # names of upstream steps
    produces: str = ".{name}.done"    # stamp, templated by `over`
    title: str = ""

# plan(steps) = topological sort by `needs`; for each edge compare over-tuples to
# pick chain | join | fan-in; assert one sink (over == ()). No step mentions "mesh".
```

`RuleKind` (7 mesh roles) and Proposal B's `Scope` (3 roles) both **collapse into
`over`** — a tuple the study author writes, open-ended, needing no new enum member
for the next axis someone invents (time step, Reynolds number, discretization
scheme…).

---

## 3. What a study becomes — "a few rules"

A workflow is then three things, and only the third is ever new:

1. **The fan-out** — a `params.yaml`/`sweep.csv`, or a `discover.py` that emits the
   rows. (Verification's `discover.py` stays: it enumerates tutorials → rows keyed on
   `case`, and `solver` variants come from `config.yaml apps:`.)
2. **The engine** — `paramspace` + `plan()` + the codegen + the `dag` renderer + one
   columns-driven `report`. **Shared, written once, never copied.**
3. **The steps** — a handful of `Step`s selected from the library, plus any the study
   invents. Each step is a `RuleSpec`-style declaration **+ a one-line `.smk` body**
   that shells to a worker verb. This is the "few rules a user adds."

```yaml
# verification study — the whole thing
title: "incompressibleFluid — tutorial drop-in"
discover: discover.py
dimensions:
  solver:                          # solver is a real sweep dimension now
    incompressiblefluid:     {app: "neofoam solver incompressiblefluid"}
    incompressiblefluidneon: {app: "neofoam solver incompressiblefluidneon"}
pipeline: [build_case, swap_solver, run, compare, report]   # ← the composition
report: {columns: [outcome, worst_abs, worst_rel, seconds]}
cases: [simpleFoam/pitzDaily, pimpleFoam/RAS/pitzDaily]      # discover selection
```

```yaml
# mesh sweep — a DIFFERENT selection over the SAME engine
pipeline: [setup_mesh, blockMesh, snappyHexMesh, checkMesh, setup, solve]
```

```yaml
# a study that does NOT exist yet, and needs no new engine code:
# mesh-convergence of the neofoam solver = reuse mesh steps + verification's diff
pipeline: [setup_mesh, blockMesh, setup, run, compare, report]
```

The step *bodies* that stay solver-specific are small and already written:
`stage`/`swap`/`compare` shell into `neofoam case {stage,swap,compare}` (today's
`verification/{stage,compare,execute,foamdict}.py`, ~500 LOC of real OpenFOAM logic —
**kept, unchanged**). `run` (`./Allrun`) and `report` (columns-driven) become fully
general.

---

## 4. Options (smallest → boldest)

**Option 0 — stop here.** Keep the two stacks; verification is at least readable now.
_Rejected: the duplication (2nd loader, 2nd codegen path, 3rd report) is the actual
cost, and it grows per new study._

**Option 1 — Proposal B verbatim** (`Scope` enum + topological wiring).
Unifies the stacks. 👍 real progress. 👎 the enum strains on the (case, solver)
cross-product and needs a `fan_in` bool bolted on; `solver` stays a second-class
wildcard rather than a paramspace dimension.

**Option 2 — projections ⭐ recommended.** Proposal B's wiring engine, but keyed on
an **`over` tuple** instead of a `Scope` enum, with `solver` promoted to a real sweep
dimension. Strictly more general, *fewer* concepts, and it is literally
"workflow + paramspace." The codegen change is the same size as Option 1 — it is the
*same three wirings*, chosen by tuple-comparison rather than by enum.

**Option 3 — steps from plugins (entry points).** External packages register `Step`s.
👍 ultimate openness. 👎 only meaningful once Option 2 exists and there are external
providers. **Defer** (same verdict the sibling doc reached for its Proposal C).

### Recommendation

**Option 2.** It is the honest form of your thesis. Do *not* build a speculative
fully-general projection engine, though — build exactly the projections the two real
studies need (`()`, `(case,)`, `(case, solver)`, `(mesh,)`, `(cad, mesh)`) and let
the tuple-typed design keep it open for free. `over` is data, not a framework.

---

## 5. What gets deleted vs kept

| Today | Fate |
|---|---|
| `verification/study.py` (2nd loader) | **deleted** → `workflow.load_cases` (POC proven in `further_simplification.py`) |
| `verification/report.py` (3rd report) | **deleted** → one columns-driven engine reporter |
| `verification/runner.py` (worker CLI) | **shrinks** → `neofoam case {stage,swap,compare,mark-failed}` verbs |
| study `rules/*.smk` (×5) | **generated** by the codegen from the `pipeline:` selection |
| `RuleKind` (7 mesh roles) + mesh-only `plan()` | **generalized** → `over` tuple + topological wiring |
| `sweep_snakefile` codegen | **generalized** → wires any `over`-tuple DAG (mesh = one case of it) |
| `verification/{stage,compare,execute,foamdict}.py` | **kept** — the step *bodies*, real OpenFOAM logic |
| per-study `discover.py` | **kept** — the one plug-in point per problem |

Net: `neofoam.tooling.verification` stops being a stack and becomes *a `discover.py`
+ a handful of step bodies + a `config.yaml` that composes*. The mesh sweep,
verification, and future studies run on one `plan()`, one codegen, one report.

---

## 6. Migration order (each step shippable & DAG-gated)

1. **`workflow.load_cases`** (+ `CaseManifest`); switch the study `Snakefile` import.
   *Already proven* — `further_simplification.py` builds the identical **452-job**
   DAG with no verification import. → deletes `study.py`.
2. **Promote `solver` to a sweep dimension.** `config.yaml dimensions.solver`;
   `discover` emits `(case, solver)` rows. The current hand-rolled `SOLVER_LABELS`
   wildcard block disappears into paramspace. → DAG unchanged (452).
3. **Register `build_case/swap/run/compare/report` as `Step`s** with `over`/`needs`.
   Mesh `plan()` still runs its old path (Proposal-A stepping stone). → DAG unchanged.
4. **Generalize `plan()` → topological wiring over `over`-tuples**; port the mesh
   sweep onto it. **Its DAG must not change — that is the regression test.**
5. **Generate the study `.smk` from the selection**; delete the on-disk `rules/*.smk`.
   Codegen emits chain/join/fan-in from `over` comparison. → DAG unchanged.
6. **Columns-driven `report`**; verification declares `report.columns`; delete
   `report.py`.

**Gate at every step (unchanged from the sibling doc):** the incompressibleFluid DAG
stays **452 jobs**, the mesh-sweep + VoF DAGs are byte-identical, and
`pytest test/tooling/{workflow,verification}` + one real `simpleFoam/pitzDaily` run
(MATCHED vs SOLVER_FAILED) stay green.

---

## 7. Honest risks & where this could be wrong

- **Fan-in on two levels is the load-bearing generalization.** The mesh codegen only
  fan-ins once (`all`). Verification needs `{case,solver}→{case}` *and* `{case}→{}`.
  If the `expand()`-from-`over` derivation gets fiddly (partial wildcards, the
  cross-axis `variant_of` join co-existing with a fan-in on the same step), that is
  where the real work and the real risk sit. Prototype step 4 on paper against the
  452-DAG *before* touching mesh.
- **`solver` as a paramspace dimension** is clean in theory; confirm the swap-target
  app survives the `params.yaml` → materialized-JSON round-trip the way mesh variants
  do. If `apps:` is awkward to express as a dimension, keep it a config list feeding
  `discover` (step 2 is then smaller) — the `over=(case, solver)` model is unaffected.
- **Is this over-engineered vs Option 1?** The test: Option 2 is *not more code* —
  it replaces an enum with a tuple in the same wiring engine. If, while building step
  4, the tuple version turns out to need *more* machinery than the enum, that is the
  signal to fall back to Proposal B's `Scope` + `fan_in` and treat `solver` as an
  extra wildcard. Decide it at step 4 on evidence, not now.
- **Don't generalize the report until one record schema is fixed.** Same blocker the
  POC flagged: both studies must emit one `results/<id>.json` shape, else the reporter
  learns study-specifics. Do steps 1–5 first; the schema falls out.
