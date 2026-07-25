<!--
SPDX-License-Identifier: GPL-3.0-or-later
SPDX-FileCopyrightText: 2026 NeoFOAM authors
-->

# Verification suite — simplification plan

Plan for restructuring the `foam_tutorials` verification study so the pipeline is
readable and self-contained, `discover.py` only *selects* cases (no tiering), and
`config.yaml` is the source of truth for which tutorials are considered.

## Goals (from the discussion)

1. **The Snakemake is hard to read.** Today the study `Snakefile` holds only
   `rule all`; everything real is hidden in the *installed package* under
   `src/neofoam/tooling/verification/rules/` and reached via `rules_dir()`. Opening
   the study tells you nothing about what runs.
2. **Rules should be reusable and on disk.** The `.smk` rule files should live as
   real, editable files in a folder next to the study, `include:`d by relative path
   — not resolved out of `site-packages`.
3. **`discover.py` is too complex.** It both *enumerates* cases and *classifies*
   them into tiers A/B/C/D. The run path only needs the enumeration; classification
   is a separate concern that happens afterwards.
4. **`config.yaml` states which cases** are considered from the tutorial tree,
   instead of the code walking `$FOAM_TUTORIALS` and filtering by solver family.

## Locked decisions

| Question | Decision |
|---|---|
| Pipeline scope | Keep the **full** pipeline: run → compare-vs-native → HTML report. Simplify internals only. |
| Run length | Keep the **20-step truncation** (`STEP_BUDGET`, binary output) — the field diff still needs it. |
| Rule files location | **Per-study `rules/` folder**, copied into each study. Divergence between studies is accepted. |
| Rule granularity | **Three separate run-side rules**: `build_case` → `swap_solver` → `run`, each a DAG node with an on-disk handoff. Plus `compare` and `report`. |
| Case selection | `config.yaml` lists the cases; `discover.py` resolves + reads minimal run metadata, no tiers. |

## Target layout

```
verification/foam_tutorials/incompressibleFluid/
  Snakefile          # header (load study, case list, wildcards) + rule all + 5 include: lines
  config.yaml        # title, apps, and the case selection
  rules/             # <- copied, on-disk, editable
    build_case.smk
    swap_solver.smk
    run.smk
    compare.smk
    report.smk
  cases/             # runtime: cases/{id}/{solver}/ staged dirs + {solver}.status.json
  results/           # results/{id}.json   (unchanged)
  report/            # analysis notes       (unchanged)
  report.html        # sink                 (unchanged)
```

The per-run directory is named **`cases/`** (was `work/`), matching the neofoam
workflow's `cases/{case}/` convention. It's a runtime artifact (gitignored). The
`Snakefile` global for its path is `CASE_ROOT = "cases"` — distinct from the
`CASES` list of `Case` objects, which is a different thing.

The packaged `src/neofoam/tooling/verification/rules/` set stays as the **pristine
master** a study is scaffolded from (see *Open items* — an optional
`neofoam verify init` copies it in). `driver.smk` is retired: its header logic moves
*into* the study `Snakefile` so the whole entry point is readable in one file.

## The DAG (per case `{id}`, per solver `{solver}`)

```
build_case ─ swap_solver ─ run ─ {solver}.status.json ┐
  (per (id, solver): native reference + each candidate) │
                                                        ├─ compare ─ results/{id}.json ┐
                                                        │                              └─ report ─ report.html
```

`{solver}` is the native solver label (e.g. `simpleFoam`) or a candidate label
(`incompressiblefluid`, `incompressiblefluidneon`). The native side and every
candidate flow through the identical `build → swap → run` chain; `swap_solver` is a
**no-op for the native solver** (its `Allrun` is left pristine), which keeps the DAG
uniform on one `(id, solver)` wildcard pair.

### Why three rules costs a little

Today one `verify_run` rule does copy + truncate + `ensure_allrun` + `swap_solver` +
`./Allrun` in one in-process step, writing one `status.json`. Splitting into three
DAG nodes means the intermediate state (a staged case dir, a swapped `Allrun`) lives
on disk between nodes, and a terminal condition discovered early (staging crashed, or
no unique solver token to swap) must be **threaded forward** through the later nodes
rather than short-circuited in-process. That thread is a small stamp payload (below).
This is the deliberate price of the readability/reusability the split buys.

## The `.smk` rules

Each rule body stays a one-line `python -m neofoam.tooling.verification.runner`
call — same authoring convention as today, just three run-side rules instead of one.
Header globals (`CONFIG`, `CASE_ROOT`, `RESULTS`, `REPORT`, `IDS`, `THREADS`,
`CANDIDATE_LABELS`, `STUDY`) are defined by the `Snakefile`.

### `build_case.smk` — casebuild copy + prepare

```python
rule build_case:
    output:
        CASE_ROOT + "/{id}/{solver}/.built.json",
    threads: lambda wc: THREADS[wc.id]
    shell:
        "python -m neofoam.tooling.verification.runner --config '{CONFIG}'"
        " build --case {wildcards.id} --solver {wildcards.solver}"
        " --cases '{CASE_ROOT}' --stamp '{output}'"
        " || python -m neofoam.tooling.verification.runner --config '{CONFIG}'"
        " mark-failed --solver {wildcards.solver} --stamp '{output}'"
```

Stages the pristine tutorial into `cases/{id}/{solver}/` with
`from_template | clean() | truncate() | ensure_allrun()` — identical recipe for
every solver, so the case dir is byte-identical across sides before the swap. The
`|| mark-failed` backstop stays here because this is where casebuild can hard-exit
the interpreter on an uncatchable FOAM error; it writes a terminal `.built.json`.

### `swap_solver.smk` — make `Allrun` call our solver (candidate only)

```python
rule swap_solver:
    input:
        CASE_ROOT + "/{id}/{solver}/.built.json",
    output:
        CASE_ROOT + "/{id}/{solver}/.swapped.json",
    shell:
        "python -m neofoam.tooling.verification.runner --config '{CONFIG}'"
        " swap --case {wildcards.id} --solver {wildcards.solver}"
        " --cases '{CASE_ROOT}' --built '{input}' --stamp '{output}'"
```

For a **candidate** solver: apply the study's `neo_patch` deviation steps, then
`swap_solver(native, app)` to replace the solver token in `Allrun` (creating the
fallback `Allrun` already happened in `build_case`). For the **native** solver: a
no-op that just stamps. If the build stamp was terminal, or there is no unique token
to swap (`NoSwapPoint` → `UNSUPPORTED_CASE`), that outcome is written into
`.swapped.json` and carried forward — no exception escapes.

### `run.smk` — run `./Allrun`

```python
rule run:
    input:
        CASE_ROOT + "/{id}/{solver}/.swapped.json",
    output:
        CASE_ROOT + "/{id}/{solver}.status.json",
    threads: lambda wc: THREADS[wc.id]
    shell:
        "python -m neofoam.tooling.verification.runner --config '{CONFIG}'"
        " run --case {wildcards.id} --solver {wildcards.solver}"
        " --cases '{CASE_ROOT}' --swapped '{input}' --status '{output}'"
```

If `.swapped.json` is terminal (stage failed, or `no_swap`), `run` writes the
matching `status.json` **without** invoking `Allrun`. Otherwise it runs `run_allrun`
and merges the result. `threads` stays here (a parallel case shells out to `mpirun`
internally). The emitted `status.json` schema is **unchanged**, so `compare` and
`report` need no changes.

### `compare.smk` / `report.smk` — unchanged in substance

Same as today, just `include:`d from the local `rules/` folder. `compare` diffs each
candidate `status.json` against the native reference into `results/{id}.json`;
`report` folds those into `report.html`.

## Stamp contract (the thread through the three nodes)

```jsonc
// .built.json
{"solver": "...", "case_dir": "...", "ok": true}
{"ok": false, "stage_failed": true, "reason": "..."}      // build failed / mark-failed

// .swapped.json  (carries .built forward, adds swap outcome)
{"ok": true}                                              // swapped, or native no-op
{"ok": false, "no_swap": true, "reason": "..."}           // NoSwapPoint -> UNSUPPORTED_CASE
{"ok": false, "stage_failed": true, "reason": "..."}      // passthrough of a terminal build

// {solver}.status.json  (SAME schema as today)
{"solver", "case_dir", "finished", "timed_out", "no_swap", "reason", "seconds", ...}
```

Keeping `status.json` identical to today's is the key constraint that limits this
refactor to the run half — `compare.py._decide`, `report.py`, and the results format
are untouched.

## `Snakefile` (self-contained header, replaces `driver.smk`)

```python
import os, re
from pathlib import Path
from neofoam.tooling.verification.study import load_study

CONFIG = os.path.abspath(workflow.configfiles[-1])
STUDY  = load_study(Path(CONFIG))
CASES  = STUDY.cases                          # already the config's selection
IDS    = [c.id for c in CASES]
THREADS = {c.id: max(1, c.subdomains) for c in CASES}
CASE_ROOT, RESULTS, REPORT = "cases", "results", "report.html"

CANDIDATE_LABELS = STUDY.candidate_labels
SOLVER_LABELS = sorted({c.native_label for c in CASES} | set(CANDIDATE_LABELS))

if not IDS:
    raise WorkflowError("no cases selected — is OpenFOAM sourced and does config.yaml list real cases?")

wildcard_constraints:
    id="|".join(re.escape(i) for i in IDS),
    solver="|".join(re.escape(s) for s in SOLVER_LABELS),

rule all:
    input:
        REPORT,

include: "rules/build_case.smk"
include: "rules/swap_solver.smk"
include: "rules/run.smk"
include: "rules/compare.smk"
include: "rules/report.smk"
```

Open the study and you see the whole thing: what's loaded, the default target, and
the five rules — each a file you can open right there. The `only:` subset mechanism
is gone; the config's case list *is* the selection.

## `discover.py` — enumerate only

Drops: `NATIVE_SOLVERS` filter, `SUPPORTED_RAS`, `UNSUPPORTED_FEATURES`, the LES /
turbulence / AMI classification, and the whole tier A/B/C/D machinery. Keeps only:
resolve each configured case path under `$FOAM_TUTORIALS`, read `application` from
`controlDict` (the native solver to swap), read `numberOfSubdomains`, attach the
diffed `fields`. Roughly:

```python
def discover(selection: list[str]) -> list[Case]:
    root = tutorials_root("incompressible")
    if root is None:
        return []
    cases = []
    for name in selection:                       # names come from config.yaml
        case = root / name
        control = read(case / "system" / "controlDict")
        application = entry(control, "application")
        cases.append(Case(
            id=case_id(name), name=name, path=case,
            native_solver=application, app="",     # apps come from config, not here
            fields=CANDIDATE_FIELDS,
            subdomains=subdomains(case),
        ))
    return cases
```

`load_study` passes the config's selection in (see below). `TIER_TITLES` is removed.
`foamdict.turbulence` / `uses_ami` are no longer called from the run path — leave the
functions (they may serve the *afterwards* classification), but `discover.py` stops
importing them.

## `config.yaml` — the case selection

```yaml
title: "incompressibleFluid — OpenFOAM incompressible tutorial drop-in"

discover: discover.py

# Native reference vs. these neofoam backends (each diffed against native).
apps:
  - "neofoam solver incompressiblefluid"
  - "neofoam solver incompressiblefluidneon"

# The cases considered, path-relative to $FOAM_TUTORIALS/incompressible.
# This list IS the sweep — add/remove a line to change what runs.
cases:
  - simpleFoam/pitzDaily
  - simpleFoam/airFoil2D
  - pimpleFoam/laminar/cylinder2D
  # ...
```

`load_study` reads `cases:` and hands it to `discover(selection)`. (If the explicit
list proves unwieldy we can allow glob patterns expanded against the tree — noted as
an extension, not built now.)

## `Case` / `Study` / report deltas

- **`Case`**: drop `tier`, `reason`, `features`, `turbulence`, and the `supported`
  property from the run path. `native_label` / `neo_label` / `subdomains` stay.
  (If `report.py` still wants a turbulence column, compute it lazily in the
  *afterwards* classification, not during discovery.)
- **`Study.load_study`**: read `cases:` from config, call `module.discover(selection)`;
  drop `tier_titles`.
- **`report.py`**: today it groups by predicted tier. With tiers gone it groups by
  **observed outcome** (`MATCHED` / `FIELDS_DIFFER` / `SOLVER_FAILED` / …) — which
  `_OUTCOME_HELP` already documents. This is the one report-side change.

## Work order (each step independently verifiable)

1. **Snakefile absorbs `driver.smk`.** Inline the header into the study `Snakefile`,
   `include:` the *packaged* rules for now. → `snakemake -n` lists the same DAG.
2. **Copy `.smk` into `rules/`.** Add the per-study `rules/` folder, switch the
   `Snakefile` to relative `include:`s. → `snakemake -n` unchanged.
3. **Split `verify_run` → `build_case` + `swap_solver` + `run`.** Add `build` /
   `swap` runner subcommands + the stamp thread; keep `status.json` identical. →
   scoped test that a case produces the same `status.json` as before.
4. **Strip `discover.py` to enumeration; move selection to `config.yaml`.** →
   `load_study` returns exactly the configured cases.
5. **Report groups by outcome, not tier.** → report renders with tiers removed.

Verify per `.claude/CLAUDE.md`: `pytest test/tooling/verification -q` (mirror any new
runner subcommand with a test), then `pre-commit run --files <changed>`. A real
end-to-end check needs OpenFOAM sourced (`FOAM_TUTORIALS`); gate the pytest with
`importorskip` where it isn't.

## Tradeoffs & risks

- **More intermediate on-disk state.** Three nodes leave a staged dir + two stamps
  per `(id, solver)`. A rerun must start clean — `build_case` still `rmtree`s its
  target first.
- **Terminal-condition threading.** The single-rule design short-circuited a staging
  crash in-process; the split carries it through stamps. Tested by a case that fails
  staging and one with no unique solver token, asserting the same final outcomes.
- **Per-study rule copies drift.** Accepted. Mitigate with the scaffold command so
  new studies start from the pristine master.
- **Losing predicted tiers.** The report no longer says "this was *expected* to fail
  because LES". That signal moves to the afterwards classification; if we still want
  it inline, it can annotate `results/{id}.json` post-run without gating the sweep.

## Open items

- **Scaffold command?** Optional `neofoam verify init <dir>` that copies the pristine
  master `rules/` + a `config.yaml` template into a new study. Keeps per-study copies
  from starting divergent.
- **Case list ergonomics.** Explicit list now; add glob expansion if it grows large.
- **The "afterwards" classifier.** Out of scope here — decide separately whether it's
  a CLI over `results/`, or a notebook, and whether it reuses `foamdict.turbulence` /
  the old tier rules.
