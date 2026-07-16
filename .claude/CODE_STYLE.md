# NeoFOAM Python package — Code style

Conventions for `src/neofoam/`. Linked from [`CLAUDE.md`](../CLAUDE.md); the
architecture map is in [`ARCHITECTURE.md`](ARCHITECTURE.md), test conventions in
[`TEST_STYLE.md`](TEST_STYLE.md).

## Typing

- **mypy is strict** (`strict = true`, `files = ["src"]`, excludes `src/NeoN`).
  Fix type errors **in code**, never by relaxing `[tool.mypy]`.
- Known pre-existing whole-tree errors exist (`fields/schema.py`,
  `framework/solver/configurations.py`) — leave them; just keep *your* changed
  files clean.
- Annotate fully. When a suppression is genuinely unavoidable, use the **correct**
  code — a wrong `# type: ignore[...]` code is itself a lint error:
  - `# type: ignore[attr-defined]` — accessing dynamically-set / decorator-created
    attributes (e.g. `pimple`/`simple`/`piso` dynamic attrs in `base.py`).
  - `cast(Any, obj).attr` — dynamically-set attributes, especially in tests.
  - `cast(Any, config_class.load(...))` — when a `type[BaseConfig]` return hides
    concrete subclass attributes.
- Prefer `dict[str, Any]` (not `dict[str, object]`) for build/context closure dicts,
  and `Any` (not `object`) for protocol return types whose results need arithmetic
  or attribute access.

## Language / API rules

- **No `getattr`/`setattr`.** Declare classes/fields explicitly — e.g. register a
  field as a Context field, not a dynamic runtime attribute.
- **`Context.runtime`** is the attribute name (**not** `runTime`).
- **Pydantic v2** throughout: `model_validator`, `model_serializer(mode="wrap")`,
  discriminated unions, `Generic[T]`. Use `FieldInfo()` (not
  `FieldInfo(required=True)` / `FieldInfo(metadata=[])`).
- **Imports at module top — except the deliberate `pybFoam` lazy-import seam.**
  Import at module top by default. The one carved-out exception is `pybFoam` in
  **library layers that must stay import-free** — `io.schema`, `io.dictread`,
  `io.strategies.openfoam_strategy`, `framework.validation.checks` — where
  `import pybFoam` sits *inside* the reading function so `import neofoam.io` /
  `import neofoam.framework.validation` pulls no native backend and the frontends stay
  layerable. This is a load-bearing convention (a file-absent guard runs *before* the
  lazy import, so a missing file needs no backend), not a violation. Everywhere else
  (solver code, bindings, frontends) `pybFoam` is a hard dependency: import it at
  module top, and make file-absent paths explicit rather than a bare `except`.

## Formatting / lint

- `ruff` with line-length **100**, rules **E/F/I** (includes isort). `ruff-format`
  is the formatter.
- Run `pre-commit run --files <changed>` and fix everything your change touches.
