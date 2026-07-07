# CLAUDE.md — NeoFOAM Python package (`neofoam`)

Guidance for working on the **Python** side of this repo: the `neofoam` package
(`src/neofoam/`) and its tests (`test/`). The C++/NeoN core and the `pybFoam`
bindings are out of scope for these notes (mypy and these conventions skip them).

## Build & test

`uv` is used **only** to create the venv + bootstrap pip. After that use the plain
Python workflow — never `uv run` / `uv sync`.

```bash
uv venv                       # once
uv pip install pip            # bootstrap pip into the venv
pip install .[all] -v         # install (NOT `pip install -e .`)
pytest                        # testpaths=test (pytest.ini_options)
SKIP=reuse pre-commit run --all-files   # format + lint + mypy
```

- **Non-editable install.** The package builds C++ bindings via scikit-build, so
  edits to `src/neofoam/*.py` do **not** take effect until you re-run
  `pip install .[all] -v` (it copies sources into site-packages and recompiles).
  Plan for this when testing source changes.
- **Don't** `rm -rf _skbuild` between rebuilds — CMake reconfigures incrementally;
  wiping it forces a slow full recompile.
- Tooling: `ruff` (line-length 100, rules E/F/I incl. isort), `mypy` `strict=True`
  with `files=["src"]`, `mypy_path=["src"]`, excluding `src/NeoN`; `pybFoam`/`NeoN`
  imports are `ignore_missing_imports`. Poe tasks exist (`poe test|lint|format|
  type_check|build_docs`) but plain `pytest`/`ruff`/`mypy` are fine.
- CLI entry point: `neofoam` (Typer) → `neofoam solver …`, `neofoam agent wizard
  <dir>`, `neofoam agent fill …` (see `src/neofoam/cli/app.py`).

## Conventions

- **mypy is strict.** Fix type errors in code, not by relaxing `[tool.mypy]`.
  Known pre-existing whole-tree errors exist (`fields/schema.py`,
  `framework/solver/configurations.py`) — leave them; just keep *your* changed
  files clean.
- **No `getattr`/`setattr`.** Declare classes/fields explicitly (e.g. register a
  field as a Context field, not a dynamic runtime attribute).
- **pybFoam-gated tests** start with `pytest.importorskip("pybFoam")`; tests that
  need the native OpenFOAM binaries use the `@requires_openfoam` marker.
- **`Context.runtime`** is the attribute name (not `runTime`).
- **Pydantic v2** throughout (`model_validator`, `model_serializer(mode="wrap")`,
  discriminated unions, `Generic[T]`).
- **Commits:** do **not** add a `Co-Authored-By: Claude` trailer. Use
  `git commit --no-verify` — the pre-commit `mypy` (whole-tree, has pre-existing
  errors) and `reuse` (flags gitignored `.claude/`) fail regardless of your diff;
  still run `pre-commit run --files <changed>` and fix everything your change
  touches. Branch before committing on `main`.

### Tests
- `test/` mirrors `src/neofoam/` 1:1 — one `test_<name>.py` per source file; tests
  are free functions, not classes.
- Dict-reading tests load **real** OpenFOAM case files kept under `test/<area>/`,
  never dict content encoded as Python strings.
- `test/io/` has **no** `__init__.py` (would shadow the stdlib `io` module);
  `test/framework/__init__.py` **does** exist.

## Architecture (where things live)

Everything is **config-driven**: a case is a set of validated pydantic configs that
serialize to OpenFOAM/JSON/YAML files.

- **`io/`** — `BaseConfig` + `@IOStrategy(OF("system/controlDict"))` bind a config
  class to a file. `write_configs(instances, case_dir)` groups co-owners of the
  same file, deep-merges, and writes each file once (FoamFile header injected for
  dict configs). `collect_config_classes`, `default_values`, strategies
  (openfoam/json/yaml).
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
