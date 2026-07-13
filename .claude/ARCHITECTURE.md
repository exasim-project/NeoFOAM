# NeoFOAM Python package — Architecture (where things live)

The detailed module map for the `neofoam` package (`src/neofoam/`). Linked from
[`CLAUDE.md`](../CLAUDE.md); keep the two in sync when the layout changes.

Everything is **config-driven**: a case is a set of validated pydantic configs that
serialize to OpenFOAM/JSON/YAML files.

- **`io/`** — `BaseConfig` + `@IOStrategy(OF("system/controlDict"))` bind a config
  class to a file. `write_configs(instances, case_dir)` groups co-owners of the
  same file, deep-merges, and writes each file once (FoamFile header injected for
  dict configs). `collect_config_classes`, `default_values`, strategies
  (openfoam/json/yaml). **`schema.py`** is the frontend-agnostic config-introspection
  surface (part of the ui/mcp→library refactor, see `refactor/README.md` Option C):
  `list_configs` · `config_schema` · `tool_catalog` · `model_catalog` + their result
  models (`ConfigInfo`/`ConfigSchema`/`ToolInfo`/`ModelSummary`). It's pure
  composition over `framework.solver.configurations` (`model_json_schema()` +
  `rjsf_uischema()` + `default_values()`); its framework/`tools` imports are **lazy
  inside the functions** so `import neofoam.io` pulls no `pybFoam` and no io↔framework
  cycle forms. `mcp.tools`/`mcp.dto` now re-export *from* here (frontend → library).
- **`fields/`** — boundary-condition types, `FieldValue[Scalar|Vector]`
  (`value_types.py`, uniform + non-uniform, per-writer serialization), `schema.py`
  (synthesised `0/<name>` field schemas), `synthesis.py`.
- **`foam/`** — `fvSchemes`/`fvSolution` per-spec config builders (`fv_configs.py`),
  FoamFile helpers.
- **`framework/model/`** — `ModelSpec`/`Model`: decorators `@spec.load` `@detect`
  `@resolve` `@build` `@operation`; `.config()` `.field()` `.register_with(family)`
  `.as_toggle(label)`. Optional **toggle** models (e.g. `boussinesq`,
  `adaptiveTimeStep`) are surfaced in UIs.
- **`framework/solver/`** — `SolverSpec`/`Solver`: `core_models()` (required, pick
  one) / `optional_models()` (zero+). `configurations(solver)` is the case-free
  config schema; `model_catalog()` / `toggle_models()` drive the wizard's
  required-vs-optional model selection (type-driven, no name-matching).
- **`framework/validation/`** — case-correctness checks (part of the ui/mcp→library
  refactor, `refactor/README.md` Option C). A `CheckRegistry` of **isolated** `Check`
  functions over a `CaseContext` (bc↔constraint-patch, GAMG, PIMPLE-final, Boussinesq,
  turbulence, …); `registry.run` enforces **honest `ok`** structurally — `ok = (no
  error findings) AND (every check ran)` — so a check that raises or hits an
  `Unreadable` leaf (from `io.dictread`) becomes an *error finding*, never a silent
  skip. `pimple_final` accepts grouped solver names (`"(U|k|epsilon)"`) so runnable
  cases don't false-fail. Owns the `Finding`/`ValidationReport` result types
  (`mcp.dto` re-exports them); `configurations`/`pybFoam`/`tools` imports are lazy so
  `import neofoam.framework.validation` is pybFoam-free. `mcp.validate_case` is a thin
  shim over it.
- **`framework/initialization/`** — staged init (load → resolve → build); `InitStep`
  via `lazy()`/`field()`/`model()`; `InitializerBuilder`. A model's `@build`
  `InitStep` can `depends_on=["models.solution_loop"]` and mutate the built engine.
- **`algorithms/solution_loop/`** — the pure-Python time loop. `SolutionLoop`
  engine holds an injectable `DeltaTConstraint` list and a **named measurement
  registry** (`publish(name,value)`/`measured(name)`). Time-step rules are
  **opt-in models** that install constraints via `install_constraints_step(...)`
  and register a `measurement_provider.<name>` — the core loop is never edited to
  add a new rule (see `solver/incompressibleFluid/models/adaptive_time_step.py`).
- **`solver/incompressibleFluid/`** — the reference solver: `incompressibleFluid.py`
  (the `SolverSpec` + bound families), `create_fields.py` (wires the init graph),
  `configs.py`, `models/` (pressure-velocity PIMPLE, viscosity, turbulence,
  boussinesq, adaptiveTimeStep).
- **`agent/`** — LLM case scaffolding. `case_fill.py` (pydantic-ai agent, output
  type = aggregate `CaseSpec`), `case_forms.py`, `pydantic_schema.py`, and
  `wizard_template.py` (the packaged marimo notebook the CLI scaffolds). The live
  wizard is `test/agent/hotRoom/case_wizard.py`; **keep `wizard_template.py` an
  exact copy** when the notebook changes (verified by
  `test/agent/test_wizard_template.py`).
- **`core/`** — `PluginSystem` (discriminated registries used by constraints,
  time-integration regimes, and model families).
- **`tooling/`** — the "above the library" layer being grown by the ui/mcp→library
  refactor (`refactor/README.md` Option C §5): trust-boundary, heavy-dep, and
  frontend-serving concerns. First module: **`workspace.py`** — a stdlib-only
  `Workspace` sandbox (owns `CaseAccessError(ValueError)`) that confines an untrusted
  `case_id`/relative path to a root dir and **rejects escapes** (`..`, absolute,
  interior-symlink traversal) via full `Path.resolve()` containment (`resolve` +
  `resolve_existing`). Fixes M3 (path confinement — mechanism wired through the six
  `mcp/tools.py` path verbs via a keyword-only `workspace=` seam; runtime activation
  on the server wire is owed to step 6) and M4 (`fill_case` validates its `source_dir`
  *before* the LLM agent runs, so a missing source errors instead of burning an API
  call). `import neofoam.tooling` is kept dependency-light (no trame/fastmcp/pybFoam).
