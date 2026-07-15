# `neofoam.tooling` — model review (SOLID + simplification)

*Review date: 2026-07-15 — branch `polish/mcp`. Scope: `src/neofoam/tooling/`
(`workspace.py`, `casebuild/`, `workflow/`), ~3060 LOC. C++/NeoN and the `mcp`/`ui`
frontends are out of scope except where `tooling` depends on them.*

---

## 1. What the layer is

`tooling/` is the **frontend infrastructure above the library** — the code the wizard,
the MCP host, and the generated Snakemake sweeps sit on. Three sub-parts:

| Module | LOC | Responsibility |
|---|---|---|
| `workspace.py` | 93 | Path trust boundary — confine an untrusted `case_id` under a root. |
| `casebuild/` | 508 | Pipe-composed case construction: `Pipeline` value + `Step`s + a single in-process tool engine (`run_tool`). |
| `workflow/` | 2050 | Parameter sweeps: `paramspace` (csv+yaml), `rules` (Snakemake rule library), `sweep` (canvas ⇄ workflow dir), `sweep_runner` (per-case/per-mesh CLI workers), `snakemake_dag`, `mesh_inputs`, `patch_set`. |

The recent move folded the old `neofoam.workflow` under `tooling/` next to `casebuild/`
and re-routed case construction through the `casebuild.Pipeline`/`run_tool` engine. That
was the right direction — this review is about how far it got and what's left.

---

## 2. What's working well

These are load-bearing and should be preserved through any refactor:

- **`workspace.py` is exemplary SRP.** One class, one job (confine a path), a typed
  error that subclasses `ValueError` for non-breaking surfacing, and the whole thing is
  stdlib-only. Nothing to change.
- **`casebuild` value semantics.** `Pipeline` is an immutable `(start, steps)` value;
  nothing touches disk until `.build_at()`; `base | step` forks. `|` means compose,
  `.build_at()` means materialize — one meaning each. This is a genuinely clean core.
- **Tool registry OCP.** `neofoam.tools.registry` — a new preprocessing tool is a new
  self-registering module; no solver, CLI, or `casebuild` edit. `run_tool` is generic
  over `ToolSpec`, so the engine never grows a per-tool branch.
- **Dependency-light `__init__` discipline.** `import neofoam.tooling` stays stdlib-only
  (no trame/fastmcp/pybFoam), and there's a subprocess test pinning it. `paramspace` and
  `sweep` keep themselves UI-free and lazy-import PyYAML so a generated Snakefile can
  import them at parse time. This is real, tested layering discipline.
- **`KeyedDim` null-object.** The implicit single-variant axis (`space=None`) keeps the
  pipeline shape uniform whether or not a dimension is swept — no special-casing at the
  call sites.

---

## 3. SOLID assessment

### Single Responsibility — the main weakness is `sweep.py`

`sweep.py` (811 LOC) is **four modules wearing one hat**:

1. **Canvas node/edge model** — `SweepDimension`, `dim_node`, `rule_nodes`, `autowire`.
   This is VueFlow-shaped presentation (pixel positions, edge stroke colors,
   `FILE_EDGE_PROPS`) — lines 62–272.
2. **Dimension validation** — `validate_dimensions`, `validate_mesh_dimension`,
   `validate_cad_dimension`, `variant_errors`, `_variant_error`,
   `_short_validation_error` — lines 340–490.
3. **Snakefile code generation** — `sweep_snakefile` builds ~60 lines of
   Python/Snakemake source as an f-string, including a conditionally-swapped
   `mesh_stem_block` — lines 493–581.
4. **Export/load round-trip** — `export_sweep`, `load_sweep`, `_read_snakefile_header`
   (regex reverse-parse of #3's output) — lines 611–811.

These change for different reasons and by different people (a UI tweak vs a rule-graph
change vs a validation-message change). Splitting into `sweep_canvas.py`,
`sweep_validate.py`, `sweep_codegen.py`, `sweep_io.py` would cut the cognitive load
sharply and is a pure move (no behavior change).

Two smaller SRP smells:

- **`CaseDir.read_field`** (`pipeline.py:49`) bolts a *field-reading* concern (numpy +
  subprocess) onto the case-*construction* value type. It's the only reason `pipeline.py`
  reaches for `numpy`/`subprocess`/`tempfile`. It belongs on a small reader helper, not
  on the pipeline value. **✅ Addressed by S6** — moved to `casebuild/reader.py`
  (`read_field(case, name)`); `pipeline.py` is now numpy/subprocess-free.
- **`sweep_runner.main`** — ~110 lines of argparse wiring in the same module as the four
  worker functions. Fine for now, but if a fifth subcommand lands, split the parser.

### Open/Closed — the `RuleSpec` boolean-kind flags

`RuleSpec` (`rules/__init__.py:81`) carries **five** overlapping discriminator fields:
`keyed_by`, `consumes_case_dims`, `consumes_mesh_dim`, `creates_mesh`,
`produces_geometry`, plus a derived `is_mesh_tool`. Each new rule *category* has meant a
new boolean, and the same discrimination is decoded in **two** places:

- `RuleRegistry.plan()` (`rules/__init__.py:195`), and
- `sweep.rule_nodes.resolve()` (`sweep.py:171`) — a parallel `if spec.…: return (...)`.

That's a classic "add a variant ⇒ touch two switch statements" shape. A single `kind`
discriminator (e.g. `RuleKind.MESH_STAGE | MESH_TOOL | GEOMETRY | CASE_SETUP | SOLVE`)
would collapse the booleans into one field and let both `plan()` and `resolve()` branch
on one enum — closer to OCP and removes the risk of an inconsistent flag combination
(e.g. `creates_mesh=True` with `consumes_mesh_dim=True`) that nothing currently forbids.

> **✅ Addressed by S3.** `RuleSpec` now carries one `kind: RuleKind`; the five booleans
> survive only as *derived* read-only properties (a pure function of `kind`), so an illegal
> combination is unrepresentable and `plan()`/`resolve()` branch on the enum.

### Interface Segregation — mostly good

`_SupportsGet` (`paramspace.py:70`) is a textbook minimal Protocol (a `.get`, satisfied
by both plain dicts and Snakemake wildcards). The only ISP wrinkle is the `RuleSpec`
flag-set above: different consumers read different subsets, which is the flags-should-be-
a-kind story again.

### Dependency Inversion — one wrong-way edge

`sweep_runner.py` imports `resolve_solver` from **`neofoam.mcp.registry`**
(`sweep_runner.py:43`). A `tooling.workflow` worker reaching *sideways/up* into the MCP
frontend for solver resolution is an inverted layer edge — solver resolution is
domain-level, not MCP-level. If the MCP package is ever trimmed or reorganized, the sweep
runner breaks for an unrelated reason. Move `resolve_solver` (or a thin re-export) down
to `neofoam.framework.solver` and have both `mcp` and `tooling` depend on that.

> **✅ Addressed by S8.** The registry now lives in `neofoam.framework.solver.registry`;
> `neofoam.mcp.registry` re-exports it (back-compat), and `sweep_runner` imports from the
> framework — the `tooling → mcp` edge is gone.

Elsewhere DIP is fine: `casebuild.run_tool` is abstracted over `ToolSpec`; only the thin
`block_mesh`/`snappy_hex_mesh`/`box` wrappers name concrete tools, which is appropriate.

### Liskov — no issues

Little inheritance. `CaseAccessError(ValueError)` is a documented, intentional widening;
`KeyedDim`'s two modes are a null-object, not a subtype substitution.

---

## 4. Duplication & complexity hotspots

**H1 — Two tool-execution engines.** `casebuild.run_tool` (`meshing.py:39`) and
`framework.tools.graph.tool_graph_steps` (`graph.py:59`).

> **⚠ Corrected on closer inspection (S5).** These do *not* actually duplicate the
> ctx-seeding. `tool_graph_steps` is a DAG **chainer**: it receives `_foam_time` from the
> solver `Context` and threads `_prev_mesh` between `InitStep` outputs — it never
> constructs a `Time`/`fvMesh` itself. The `pyf.Time` + `pyf.fvMesh` seeding lives only in
> `casebuild.run_tool`, so there is no second copy to fold in. S5 therefore reduced to a
> *local* clarity extraction — `seed_tool_ctx(case, *, needs_prev_mesh)` in
> `casebuild/meshing.py` — that names the seeding contract in one place and documents why
> the graph engine deliberately does not share it.

**H2 — Encode-with-f-string / decode-with-regex round-trip.** `sweep_snakefile` emits a
Snakefile whose header carries `SOLVER`/`BASE_CASE`/`CAD_MODEL`/enabled-rules, and
`load_sweep` → `_read_snakefile_header` (`sweep.py:775`) *reverse-parses that same
generated source with regexes* (`SOLVER\s*=\s*("...")`, `rules_dir()\s*/\s*"..."`). The
generator and the parser are now coupled by string format: change one `json.dumps(...)`
line and the regex silently returns a default. This is the most brittle seam in the
layer.

**H3 — Validation done 3–4 times.** A variant payload is validated in
`sweep.validate_*` (export time), again in `YamlParamSpace._validate` (Snakefile parse
time), again in `sweep_runner.apply_configs` / `setup_mesh_case` (apply time). The code
even documents "the runner re-validates". Some of this is deliberate (defense at each
boundary), but the *shape* logic — "mesh variant is `{config: payload}`, cad variant is
`{alias: number}`, everything else is one config" — is written out in both the throwing
validators (`validate_mesh_dimension`/`validate_cad_dimension`) and the non-throwing
`_variant_error`. One source of truth, with a thin `raise`-wrapper over the
error-returning form, removes the divergence risk.

> **✅ Addressed by S4.** `_variant_error` is now the sole shape-checker (mesh/cad/regular);
> `variant_errors` collects its messages and the three `validate_*` functions are thin
> `_raise_variant_errors` wrappers over it. Validation at the paramspace/runner boundaries
> stays (deliberate defense-in-depth), but the *shape rule* exists in exactly one place.

**H4 — Name collision: two `run_tool`s.** `casebuild.run_tool` (the engine) and
`sweep_runner.run_tool` (the `tool` subcommand worker). `sweep_runner` already has to
`import run_tool as run_tool_on_case` to dodge the clash. Rename the worker to
`run_tool_command` / `run_preprocess_tool` so the engine keeps the clean name.

> **✅ Addressed by S7.** The worker is now `run_tool_command`; `sweep_runner` imports the
> engine plainly as `run_tool`, and the alias is gone.

---

## 5. Simplification options — ranked

Each is independent; ordered by payoff ÷ risk.

| # | Change | Effort | Risk | Payoff |
|---|---|---|---|---|
| **S1 ✅ done** | **Split `sweep.py`** into `sweep_canvas` / `sweep_validate` / `sweep_codegen` / `sweep_io`. Pure move + re-export from `sweep.py` for back-compat. | M | Low | High — turns the single hardest-to-read file into four single-purpose ones. |
| **S2 ✅ done** | **Replace the Snakefile-header regex round-trip (H2)** by writing a tiny `sweep.meta.json` sidecar (`{solver, base_case, enabled, cad_model}`) at export and reading *that* in `load_sweep`. Delete `_read_snakefile_header`. | S | Low | High — removes the most brittle coupling; `load_sweep` stops depending on generated source formatting. |
| **S3 ✅ done** | **Collapse `RuleSpec` booleans into one `kind` enum (OCP fix).** Update `plan()` and `sweep.resolve()` to branch on `kind`. | M | Med | Med-High — one place to add a rule category; kills illegal flag combos. |
| **S4 ✅ done** | **Single validation source (H3).** Make `_variant_error` the one shape-checker; have `validate_*` be `raise_if(_variant_error(...))`. | S | Low | Med — removes divergent copies of the same rule shape. |
| **S5 ✅ done** | **Extract shared ctx-seeding for the two tool engines (H1).** *Scope corrected:* the graph engine is a chainer that never seeds — so this became a local `seed_tool_ctx` extraction in `casebuild`, not a cross-engine merge. | S | Low | Low — names the `_foam_time`/`_prev_mesh` contract in one place. |
| **S6 ✅ done** | **Move `CaseDir.read_field` off the pipeline value** into `casebuild/reader.py`; keeps `pipeline.py` free of numpy/subprocess. | S | Low | Low-Med — restores SRP on the core value type. |
| **S7 ✅ done** | **Rename `sweep_runner.run_tool` → `run_tool_command` (H4)** and drop the import alias. | XS | Low | Low — removes a daily-friction naming clash. |
| **S8 ✅ done** | **Relocate `resolve_solver` to the framework layer (DIP fix).** | S | Med | Med — removes the `tooling → mcp` inverted edge. |

All eight options are now implemented (S1–S8). The original recommended order was: S2 + S7
+ S6 first (small, low-risk), then S1 (the big readability win), then S3/S4/S5, with S8
last (two packages, own test run).

---

## 6. Bottom line

The `tooling` layer was **structurally sound** to begin with — the `casebuild` core, the
workspace sandbox, the tool registry, and the dependency-light layering were always the
parts to keep. The debt was concentrated, not diffuse, and is now cleared:

- the **over-stuffed module** (`sweep.py`) is split four ways (S1);
- the **brittle Snakefile ⇄ regex round-trip** is replaced by a `sweep.meta.json` sidecar
  (S2);
- the **flag-based rule discriminator** is a single `RuleKind` enum (S3);
- the **validation shape rule** has one home (S4), the field reader is off the pipeline
  value (S6), the `run_tool` name clash is gone (S7); and
- the **inverted `tooling → mcp` layer edge** is removed by relocating the solver registry
  to the framework (S8).

S5's premise (two engines duplicating ctx-seeding) did not survive inspection — the graph
engine is a DAG chainer, not a second seeder — so it landed as a local `seed_tool_ctx`
extraction plus a documented note on why the two paths stay separate. Every change kept
the public import surface stable (facades / re-export shims) and left the test suite green.
