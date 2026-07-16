---
name: understand-tests
description: Understand a Python test suite and answer "do the tests cover the most important features?" by combining two lenses into one analysis report. STATIC AST pass (call_graph.py): module-dependency + call graph, plus it STRATIFIES tests by altitude (e2e/integration → component → unit) so you can read top-down, and ranks public functions by centrality with the highest test level that reaches each — so an important feature with no integration test shows up as a GAP. DYNAMIC pass (coverage_contexts.py): per-test line coverage, which test ran which line, redundancy clusters. Produces a markdown "Test & Dependency Map" cross-referencing the two so feature gaps, dead code, and wrong-direction dependencies are visible. Use to onboard onto an unfamiliar test suite, check whether critical features/paths are tested, audit coverage, or verify dependency direction before a refactor. Read-only; does not edit code.
---

# Understand the test logic and how the files connect

To understand a test suite you need two questions answered, and neither alone is
enough:

- **Static** — *how do the files connect?* Which module imports which, which
  function calls which, and what can each test even reach. Answered by parsing
  the AST — no execution, safe on native-extension code.
- **Dynamic** — *what did the tests actually exercise?* Which test ran which
  line, which tests are coverage-redundant, which lines nothing touched. Answered
  by running the suite under coverage with per-test contexts.

This skill runs both, then **cross-references them** into a single report. That
cross-reference is where understanding lives: a function the call graph says a
test *reaches* but coverage says never *ran* is a dead branch or a mock boundary;
a module edge (`ui → mcp`) tells you a dependency the tests silently rely on.

Two scripts do the mechanical work; your job is to run them on the right target,
read the artifacts, and synthesize the report.

**Announce at start:** "I'm using the understand-tests skill — I'll run a static
call-graph pass and a dynamic coverage pass on `<target>`, then write a Test &
Dependency Map."

## Inputs you need

- **Target source** — the package/dir/file(s) to understand, e.g.
  `src/neofoam/ui` or `src/neofoam/framework/validation`.
- **Its tests** — the mirrored test dir/file(s), e.g. `test/ui`. (`test/` mirrors
  `src/neofoam/` 1:1 in this repo — see `.claude/TEST_STYLE.md`.)
- **The dotted module path(s)** of the target for the coverage pass, e.g.
  `neofoam.ui` (coverage scopes to this).

If the user names only a feature ("understand the sweep tests"), resolve these
yourself: find the source under `src/neofoam/…`, the tests under `test/…`, and
the dotted path.

## Workflow

### 1. Static pass — call graph + module dependencies (always runs)

Stdlib-only, no install needed, never imports the target:

```bash
python .claude/skills/understand-tests/call_graph.py \
    --source src/neofoam/<target> \
    --tests  test/<target> \
    --scope  neofoam \
    --out    <out>/callgraph
```

- `--scope neofoam` restricts the whole analysis to **neofoam functions** — every
  builtin / trame / pydantic / stdlib call is dropped (counted as
  `external_call_sites_dropped`, not graphed), so the graph, the unresolved list,
  and the confidence are all about *this project's* code, not framework noise.
  Module roots (`src/` vs `test/`) are auto-detected from `__init__.py` ancestry —
  no `--pkg-root` needed.
- Read `<out>/callgraph/callgraph.md` (Mermaid module graph + tables) and
  `callgraph.json` (machine-readable). `modules.dot` / `callgraph.dot` render with
  `dot -Tsvg` if you want images.

**How to read it:**
- **Module edges are authoritative** — they come from imports, so every one is
  real. This is the "how the files connect" answer and the one for
  dependency-direction questions (frontend → library, never back; see
  `refactor/README.md`). Flag any edge that points the wrong way.
- **Call edges are best-effort** — Python dynamic dispatch means not every call
  binds. The report prints a **resolution confidence %** = resolved / *neofoam*
  call sites (external calls are already excluded, so this is a real signal, not
  diluted by framework noise — expect ~70–90%). When it's lower, look at the "top
  unresolved names" table: those are neofoam calls the AST couldn't pin. The usual
  cause is a call into a neofoam module you didn't pass as `--source` (its defs
  aren't indexed) — **add that module to `--source` to resolve them**. The rest are
  genuine dynamic dispatch. Trust high-confidence edges; treat the call graph as a
  strong hint, not proof.
- **Test stratification (read top-down)** — the `test_strata` table labels each
  test by **altitude** = how many source *modules* it reaches: `integration` (≥4)
  · `component` (2–3) · `unit` (1) · `isolated` (0). Read the integration rows
  **first** — they tell you what the suite claims the system does end-to-end;
  then component, then unit, to see how the pieces are pinned down. Altitude is
  *touch surface*, not assertion focus (a test that boots the app reaches
  everything it fans out to) — a coarse but fast way to grasp an unfamiliar suite.
- **Feature coverage — the "are the important features proven?" answer.** The
  `feature_coverage` table lists every **public** source function ranked by
  in-degree (how many source calls depend on it — the centrality proxy for
  "important"), with `best_test_level` = the *highest* test altitude that reaches
  it. A high-in-degree function whose best level is **GAP** (or only `unit`) is an
  important feature with no integration test — the thing to fix first. The summary
  prints `public_api_static_gaps` (e.g. "62 of 99 public fns unreached").
- **Statically-unreached source functions** — nothing in the tests calls them.
  Candidates for missing coverage OR dead code — confirm against the dynamic pass
  (step 2) before concluding, since dynamic calls can reach what the AST can't.

### 2. Dynamic pass — per-test line coverage (runs if the package is installed)

Reuses the coverage helper that ships with the `spec-test-reviewer` agent. Needs
the package importable in the venv (`pip install .[all] -v`) and `coverage` +
`pytest`. Because this repo builds native pybind11 extensions, pass any that
abort under the tracer via `--preimport`:

```bash
python .claude/agents/coverage_contexts.py \
    --source neofoam.<target> \
    --tests  test/<target>/ \
    --out    <out>/coverage
    # add --preimport pybFoam   if importing the target aborts under coverage
```

Read `<out>/coverage/coverage.md` and `coverage.json`:
- **missing lines** — hard line-level gaps (a `raise`/branch nothing ran).
- **identical-coverage clusters** — tests running the exact same lines →
  parametrize candidates (only if their assertions differ merely in data).
- **sole-owner lines** — a test that solely owns a line is load-bearing.
- **`line_coverage_discriminates=false`** — module too small for coverage to tell
  tests apart; judge by assertions, not coverage redundancy.

If the package can't be imported / the coverage run fails, **say so and continue
with the static pass only** — the call graph alone still answers "how the files
connect" and gives static reachability.

### 3. Cross-reference and synthesize

This is the point of the skill. Put the two lenses side by side:

| Signal from static | Signal from dynamic | What it means |
|---|---|---|
| test reaches fn `X` | coverage: `X` ran | genuinely exercised ✓ |
| test reaches fn `X` | coverage: `X` never ran | dead branch / mocked-out boundary / path not taken |
| fn `X` unreached by any test | coverage: `X` never ran | untested (or dead code) — corroborated |
| fn `X` unreached statically | coverage: `X` ran | dynamic dispatch the AST missed — call graph blind spot |
| module edge `A → B` | tests import/exercise both | a real dependency the suite relies on; note if direction is wrong |

Read the *actual assertions* of a handful of the highest-reach tests to describe
what the suite proves in behavioral terms — the graphs tell you *what connects*,
the assertions tell you *what's guaranteed*.

**Render the feature-coverage chart** (optional, needs matplotlib) — one image
that answers "are the important features proven?" at a glance:

```bash
# static (bars colored by test altitude — "a test touches it"):
python .claude/skills/understand-tests/feature_chart.py \
    --json <out>/callgraph/callgraph.json --out <out>/feature_static.png
# dynamic (bars colored by measured line coverage — "lines actually ran"):
python .claude/skills/understand-tests/feature_chart.py \
    --json <out>/callgraph/callgraph.json \
    --coverage-json <out>/coverage/coverage.json --out <out>/feature_dynamic.png
```

x = centrality (in-degree). With `--coverage-json` a central function that is
**red/amber** = an important feature whose lines don't fully run — the priority.
The static→dynamic recolor is the honest correction: functions the static pass
called "GAP" often turn green once you measure, because the loop dispatches
through protocols the AST can't follow.

## Output — the Test & Dependency Map

Write the report to **`plans/<target>-test-map.md`** (create `plans/` if needed).
Structure:

1. `# Test & Dependency Map: <target>` + date/commit line, then a 3–5 sentence
   **Overview**: what the module does, the test-level histogram (e.g. "39
   integration / 24 component / 63 unit"), how many public API fns have no test,
   the measured line coverage %, the resolution confidence %, and the single most
   important thing you learned.
2. **Are the important features proven?** — lead with this. Embed the
   `feature_coverage` table (central public fns × best test level); for the top
   in-degree functions, say in prose whether each has a genuine integration test
   and name the **GAPs** (high-centrality fns with no/weak coverage) as the
   priority work. This is the answer to "does the suite cover what matters".
3. **Read top-down (e2e → unit)** — the `test_strata` table. Describe the shape:
   how integration-heavy vs unit-heavy the suite is, and what the top integration
   tests establish end-to-end. Flag if there are *no* true unit tests for a
   complex function (only reached via app-boot integration tests) — that's a
   diagnosability gap.
4. **How the files connect** — embed the Mermaid module graph. A short prose read:
   the layers, the key edges, and **any dependency pointing the wrong way** (call
   it out with carried symbols, e.g. ``ui.forms → mcp.tools`` carries `save_case`).
5. **Coverage & reachability gaps** — a table `Surface element (fn/branch) |
   Static reach | Dynamic cover | Verdict`. Verdict ∈ {covered, untested,
   dead-code?, dispatch-blindspot}. Cite line numbers for missing lines.
6. **Redundancy / simplification** — identical-coverage clusters from the dynamic
   pass as parametrize candidates (only where assertions differ merely in data);
   note sole-owner (load-bearing) tests that must NOT be merged. If coverage
   can't discriminate, say so.
7. **Caveats** — the resolution confidence %, whether the dynamic pass ran, and
   any file skipped (syntax error, unreadable).

Then return to the caller: the Overview, the biggest gap, any wrong-direction
dependency, and the path written.

## Scope / non-goals

- **Read-only.** This skill understands and reports; it does not edit tests,
  add coverage, or fix dependencies. Hand its report to `spec-loop` /
  `write-spec` or act on it in a separate step.
- The call graph is **static best-effort** — it will miss calls through dynamic
  dispatch, `functools.partial`, registries, and callbacks. Never assert "no test
  covers X" from the call graph alone; corroborate with the dynamic pass.
- The dynamic pass needs an **installed, importable** package. It is optional; the
  skill still produces value from the static pass when coverage can't run.
```
