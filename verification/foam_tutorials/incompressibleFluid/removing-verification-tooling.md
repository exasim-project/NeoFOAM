# General, composable rules — one rule library, many problems

**Goal:** stop shipping a *problem-specific* tooling stack per workflow. Build **one
library of general, composable rules** — small units that chain by a stamp contract
— so that verification, mesh sweeps, and problems we haven't written yet are each
just a **composition** of the same blocks. `neofoam.tooling.verification` then
disappears as a stack: it contributes a couple of blocks + a `discover.py`, and the
study is a config that *composes*.

## The one idea: a rule is a composable unit

A rule is general when it says nothing about *which problem* it serves — only:

```
inputs (stamp files)  ──▶  [ do one thing to a case/variant dir ]  ──▶  output stamp
```

Compose by wiring one rule's output stamp to the next rule's input. A *problem* is
then a **selection + wiring** of rules — a DAG — not a bespoke Snakefile. Add a
block once, reuse it in every composition that wants it.

```
blocks (the library)              compositions (the problems)
  stage   swap   run              verification = stage → swap → run → diff → report
  diff    report blockMesh        mesh sweep   = stageMesh → blockMesh → snappyHexMesh
  snappyHexMesh  setup  solve                    → checkMesh → setup → solve → post
  checkMesh  post  …              conv. study  = stageMesh → blockMesh → setup → run → diff
```

## We already built this — for meshes only

`neofoam.tooling.workflow.rules` is *exactly* a composable-rule library:

- **`RuleSpec`** — one block: `name`, its `.smk` file, `inputs`/`outputs` patterns,
  a stamp.
- **`RuleRegistry`** — the library of blocks in pipeline order.
- **`RuleRegistry.plan(enabled)`** — **the composer**: takes a *selection* of block
  names and resolves it into a validated `RulePlan` (wired by stamps), which the
  codegen turns into a Snakefile.

That is the machinery the user is describing. The problem: it is **hardwired to
mesh sweeps**, so verification couldn't use it and grew a parallel stack instead.
Two things pin it to meshes:

1. **`RuleKind`** enumerates *mesh* roles (`MESH_STAGE`, `GEOMETRY`, `MESH_CREATE`,
   `MESH_TOOL`, `CASE_SETUP`, `CASE`) and the stamp wiring is derived from them.
2. **`plan()`** hardcodes the mesh chain: it *requires* `setup`, `solve`,
   `setup_mesh`, and rejects any composition whose chain doesn't start with a
   mesh-creating tool.

So a verification composition (`stage → swap → run → diff → report`) is literally
unrepresentable in today's registry — hence the duplicate stack.

---

## Three proposals (smallest → boldest)

### Proposal A — Verification blocks as new `RuleSpec`s, mesh `plan()` unchanged

Register `stage`/`swap`/`run`/`diff`/`report` as `RuleSpec`s and copy them like the
mesh rules.

- 👍 Reuses `RuleSpec` + `rules_dir()` + the codegen's `include:` mechanism.
- 👎 **Doesn't actually compose.** `plan()` still demands `setup_mesh`/`solve` and a
  mesh chain, so verification can't call it — it would need its own composer. This
  is duplication with extra steps; it does **not** deliver "many problems, one
  library." Rejected as the target, useful only as a stepping stone.

### Proposal B — Generalize the composer: any wiring, one library  ⭐ recommended

Lift the two mesh-specific pins so `plan()` composes **any** selection:

- **A block declares its own dependencies.** Replace the mesh-role `RuleKind` with a
  generic contract on `RuleSpec`: `needs` (input stamps, possibly fan-in) and
  `produces` (output stamp), plus a `scope` (`per_case`, `per_variant`, or `sink`)
  that only says how often it runs. No block names "mesh."
- **`plan()` becomes a topological wiring** over the selected blocks by matching
  `produces → needs`, validating it's a DAG with a single sink. The mesh chain is
  then just *one* valid wiring, not a hardcoded requirement.
- **The registry holds every block** — mesh blocks *and* `stage`/`swap`/`run`/
  `diff`/`report`. Third-party/problem blocks register alongside.
- **A problem's `config.yaml` names its composition** (`pipeline: [stage, swap, run,
  diff, report]`); the study ships no `.smk` and no loader.

This is the literal "general composable rules → many problems." Verification, the
mesh sweep, and a new convergence study are three `pipeline:` selections over one
registry, one composer, one codegen, one report.

- 👍 One mechanism; a new problem is a config + (maybe) a new block, never a stack.
- 👍 The mesh sweep and verification stop diverging — same wiring engine.
- 👎 Real refactor of `RuleKind`/`plan()` (they're mesh-shaped today); needs the
  generic wiring covered by tests before either study switches to it.

### Proposal C — Blocks from plugins (entry points)

Let external packages contribute `RuleSpec`s via entry points, so the registry is
open and a problem is `neofoam workflow run <config>`.

- 👍 Ultimate generality — third parties add blocks without touching the core.
- 👎 Only worth it once B exists and there are several external block providers.
  Defer.

## Recommendation

**Do B, on top of the registry that already exists** — generalize `RuleKind`/`plan()`
into a problem-agnostic composer, then express both meshing and verification as
selections. A is only worth doing as the first commit of B (register the blocks)
before the composer is generalized. C waits.

---

## Sketch of the target (Proposal B)

### The generalized block model (`workflow.rules`)

```python
class Scope(Enum):          # replaces the mesh-shaped RuleKind — says only *cadence*
    PER_CASE = "per_case"       # runs once per (case, solver)
    PER_VARIANT = "per_variant" # runs once per mesh/cad variant (the keyed rows)
    SINK = "sink"               # a fan-in over all cases (report / all)

@dataclass(frozen=True)
class RuleSpec:
    name: str                       # "stage", "blockMesh", "diff"
    smk_file: str                   # packaged body
    scope: Scope = Scope.PER_CASE
    needs: tuple[str, ...] = ()      # input stamp basenames it chains off
    produces: str = ""               # output stamp basename it writes
    fan_in: bool = False             # sink over every case (diff/report)
    title: str = ""

# plan() = generic topological wiring: match produces→needs across the selection,
# assert one connected DAG with a single sink, return the ordered RulePlan.
# No rule mentions "mesh"; the mesh chain is just one valid wiring.
```

### The library gains verification blocks alongside the mesh ones

```python
def default_registry() -> RuleRegistry:
    return RuleRegistry([
        # ── general case blocks (new) ────────────────────────────────────────
        RuleSpec("stage",  "stage.smk",  produces=".built.json"),
        RuleSpec("swap",   "swap.smk",   needs=(".built.json",),  produces=".swapped.json"),
        RuleSpec("run",    "run.smk",    needs=(".swapped.json",), produces=".ran"),
        RuleSpec("diff",   "diff.smk",   fan_in=True, needs=(".ran",),
                 produces="results/{id}.json"),
        RuleSpec("report", "report.smk", scope=Scope.SINK,
                 needs=("results/{id}.json",), produces="report.html", title="report"),
        # ── mesh blocks (unchanged behaviour, now just more library entries) ──
        RuleSpec("setup_mesh",    "setup_mesh.smk",    scope=Scope.PER_VARIANT, …),
        RuleSpec("blockMesh",     "block_mesh.smk",    scope=Scope.PER_VARIANT, …),
        RuleSpec("snappyHexMesh", "snappy_hex_mesh.smk", scope=Scope.PER_VARIANT, …),
        RuleSpec("setup",         "setup.smk",         needs=(".mesh.done",), …),
        RuleSpec("solve",         "solve.smk",         needs=(".applied.json",), …),
    ])
```

`stage`/`swap`/`diff` shell into the small verification worker verbs (`neofoam case
stage|swap|compare`) — the only irreducibly solver-specific code left. `run` and
`report` are fully general (`./Allrun`; a columns-driven renderer).

### A problem is a composition in `config.yaml`

Same file for every problem; only the `pipeline:` selection and the problem's own
`discover.py` differ.

```yaml
# verification study
title: "incompressibleFluid — tutorial drop-in"
discover: discover.py
apps:
  - "neofoam solver incompressiblefluid"
  - "neofoam solver incompressiblefluidneon"
pipeline: [stage, swap, run, diff, report]     # ← the composition
report: {columns: [outcome, worst_abs, worst_rel, seconds]}
cases:
  - simpleFoam/pitzDaily
  - pimpleFoam/RAS/pitzDaily
```

```yaml
# mesh sweep — a DIFFERENT composition of the SAME library
pipeline: [setup_mesh, blockMesh, snappyHexMesh, checkMesh, setup, solve]
```

```yaml
# a new problem, no new code: mesh-convergence of the neofoam solver
pipeline: [setup_mesh, blockMesh, setup, run, diff, report]
```

### The Snakefile (general — copied verbatim into any study)

```python
import os, re
from neofoam.tooling.workflow import load_cases, default_registry, emit_snakefile

CONFIG = os.path.abspath(workflow.configfiles[-1])
STUDY  = load_cases(CONFIG)
PLAN   = default_registry().plan(STUDY.pipeline)   # ← the composer wires the selection

rule all:
    input: PLAN.sink_target(STUDY)

include: emit_snakefile(PLAN, STUDY)   # generic codegen writes the wired .smk, returns its path
```

The composer (`plan`) does the wiring; the codegen (`emit_snakefile`, sibling of the
existing `sweep_snakefile`) turns the `RulePlan` into a Snakefile whose `include`d
`.smk` bodies come straight from the packaged library. No study authors rules; the
generated file lands under `.snakemake/` for inspection.

## What gets deleted vs kept

| Today | Fate under B |
|---|---|
| `verification/study.py` (2nd loader) | **deleted** → `workflow.load_cases` |
| `verification/report.py` (3rd report) | **deleted** → general columns-driven renderer |
| study `rules/*.smk` (×5) + packaged VoF `rules/*.smk` | **deleted** → generated from the composition |
| `RuleKind` (mesh roles) + mesh-specific `plan()` | **generalized** → `Scope` + topological wiring |
| `verification/runner.py` | **shrinks** → `neofoam case {stage,swap,compare,mark-failed}` verbs |
| `verification/stage.py`/`compare.py`/`foamdict.py`/`execute.py` | **kept** — the block *bodies* (OpenFOAM domain logic) |
| per-study `discover.py` | **kept** — the one plug-in point per problem |

Net: `verification` stops being a parallel stack. It becomes **a handful of blocks +
a `discover.py` + a config that composes** — and the mesh sweep, verification, and
future problems all run on one library, one composer, one report.

## Migration order (each step independently shippable & green)

1. `workflow.load_cases` (+ `CaseManifest`); swap the study Snakefile import.
   *(Proven in `further_simplification.py` — identical 452-job DAG, no verification
   import.)*
2. Register `stage`/`swap`/`run`/`diff`/`report` as `RuleSpec`s in the library
   (Proposal A — blocks exist, mesh `plan()` untouched yet).
3. Generalize `RuleKind` → `Scope` and `plan()` → topological wiring; port the mesh
   sweep onto it (its DAG must not change — that's the regression test).
4. Point the verification study at `plan([stage, swap, run, diff, report])`; delete
   the study `.smk` files, the packaged VoF rules, `study.py`, and `report.py`.
5. `render_report` becomes columns-driven; verification declares `report.columns`.

**Gate at every step:** the incompressibleFluid DAG stays 452 jobs, the mesh sweep
and VoF DAGs are unchanged, and `pytest test/tooling/{workflow,verification}` + a
real `simpleFoam/pitzDaily` run (MATCHED vs SOLVER_FAILED) stay green.
