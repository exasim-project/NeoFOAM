# CLAUDE.md — NeoFOAM Python package (`neofoam`)

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


Guidance for the **Python** side of this repo: `src/neofoam/` and its tests
(`test/`). The C++/NeoN core and the `pybFoam` bindings are out of scope for these
notes (mypy and these conventions skip them).

## Build & test (the commands)

`uv` is used **only** to create the venv + bootstrap pip. After that use the plain
Python workflow — never `uv run` / `uv sync`.

```bash
uv venv                       # once
uv pip install pip            # bootstrap pip into the venv
pip install .[all] -v         # install (NOT `pip install -e .`)
pytest test/<area> -q         # scoped tests while working
pytest                        # whole suite (testpaths=test)
pre-commit run --files <changed>        # format + lint + mypy on your diff
```

### Gotchas that cost real time — read before editing

- **Non-editable install.** scikit-build copies sources into site-packages, so edits
  to `src/neofoam/*.py` do **not** take effect until `pip install .[all] -v` re-runs.
  Plan for this before testing source changes.
- **Never `rm -rf _skbuild`** — CMake reconfigures incrementally; wiping forces a
  slow full recompile.
- **Whole-repo checks have pre-existing failures.** Pre-commit `mypy` (whole-tree)
  and `reuse` fail regardless of your diff. Judge your work by `pre-commit run
  --files <changed>` and scoped pytest — fail only on NEW errors in files you touched.
- Tooling: `ruff` (line-length 100, E/F/I), `mypy` `strict=True` on `files=["src"]`
  (excludes `src/NeoN`; `pybFoam`/`NeoN` are `ignore_missing_imports`).
- CLI entry point: `neofoam` (Typer) — see `src/neofoam/cli/app.py`.

## Verification norms (definition of done)

A change is done when you have **run** its proof, not when it looks right:

1. the **scoped tests** for the touched area pass (`pytest test/<area> -q`) — after a
   fresh `pip install .[all] -v` if source changed;
2. `pre-commit run --files <changed>` is clean (or only pre-existing failures);
3. you report the actual commands + output, not an assertion that it works.

Never delete or weaken an existing test to get green — surface the conflict instead.
Optional/native deps gate with `pytest.importorskip(...)`, they don't fail.

## Conventions

Detailed guides — read the relevant one before writing code, update it when a
convention changes:

- **[`.claude/CODE_STYLE.md`](.claude/CODE_STYLE.md)** — typing/mypy, no
  `getattr`/`setattr`, `Context.runtime`, Pydantic v2, imports-at-top, ruff.
- **[`.claude/TEST_STYLE.md`](.claude/TEST_STYLE.md)** — `test/` mirrors
  `src/neofoam/` 1:1, free functions not classes, real OpenFOAM case files (never
  dicts-as-strings), pybFoam/OpenFOAM gating, `__init__.py` rules (`test/io/` has
  none — it would shadow stdlib `io`).

The essentials: **mypy is strict** — fix type errors in code, never by relaxing
`[tool.mypy]`; keep *your* changed files clean.

### Commits

- Do **not** add a `Co-Authored-By: Claude` trailer.
- Commit with `git commit --no-verify` (the whole-tree hooks fail pre-existing — see
  gotchas) — but only after `pre-commit run --files <changed>` is clean.
- Branch before committing on `main`. Commit only when asked.

## Architecture (where things live)

The module map lives in **[`.claude/ARCHITECTURE.md`](.claude/ARCHITECTURE.md)** —
read it to locate a config, model, solver, or init step; update it when the layout
changes.

In short: everything is **config-driven** — a case is a set of validated pydantic
configs serialized to OpenFOAM/JSON/YAML. `io/` binds configs to files,
`fields/`/`foam/` build field + scheme configs, `framework/` holds
`ModelSpec`/`SolverSpec` + staged init, `algorithms/solution_loop/` is the
pure-Python time loop, `solver/incompressibleFluid/` is the reference solver,
`agent/` is the LLM case-scaffolding layer.

## Spec workflow (larger features)

For features too large for one session, this repo carries a spec pipeline — use it
in this order:

1. **`/grill-spec`** — interrogate requirements out of the user → `plans/<name>-requirements.md`
2. **`/write-spec`** — turn the brief into a testable spec → `plans/<name>-spec.md`
3. **`/spec-loop plans/<name>-spec.md`** — implement it slice-by-slice with
   evidence-gated iterations under `loop/<feature>/` (never commits; you commit).

`/understand-tests` maps an unfamiliar test suite (coverage + call-graph) before
refactors.
