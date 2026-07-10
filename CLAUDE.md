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

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
