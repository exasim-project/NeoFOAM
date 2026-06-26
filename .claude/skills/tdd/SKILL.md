---
name: tdd
description: Test-driven development workflow for implementing a plan
---

# Test-Driven Development

Implement the given plan using strict Red-Green-Refactor TDD.

## Input

$ARGUMENTS

## Workflow

For each piece of behavior in the plan:

### 1. RED — Write a failing test first
- Write a minimal test that captures the next desired behavior.
- Build and run it. **Confirm it fails** (compile error or assertion failure).
- If it already passes, skip to the next behavior.

### 2. GREEN — Minimum implementation to pass
- Write only enough code to make the failing test pass.
- No extras, no abstractions, no "nice to haves".
- Build and run. **Confirm it passes.**
- run quality checks (pre-commit). **Confirm they pass.**

### 3. REFACTOR — Clean up while green
- Improve naming, remove duplication, improve structure.
- Run tests after each change. Nothing should break.
- run quality checks (pre-commit). **Confirm they pass.**

### 4. REPEAT
- Go back to step 1 for the next behavior.
- Continue until the plan is fully implemented.

## Rules

- **Never write implementation before a failing test.**
- **Small steps** — each cycle should be a focused increment.
- **Build frequently** — catch errors early.
- **Run tests frequently** — validate after every change.
- **If stuck**, write a simpler test for a smaller piece of behavior.

## Build & Test

Python is a **non-editable** install: edits to `src/neofoam/*.py` only take effect
after a reinstall, so reinstall **only when you changed `src/neofoam`** (it
recompiles the C++ bindings). **Never** `rm -rf _skbuild` — CMake reconfigures
incrementally. Never use `uv run` / `uv sync`.

```bash
# C++ side (when touching the native library / bindings)
cmake --build --preset develop -j   # do not use $(nproc)
ctest --preset develop -R <testName>

# Python side: reinstall only after a src/neofoam change, then run targeted tests
pip install .[all] -v               # NOT `pip install -e .`; skip if only test/ changed
pytest -k <testName>                # or: pytest test/path/to/test_x.py

# format, lint, type check (on the files you changed)
SKIP=reuse pre-commit run --files <changed files>
```

Notes:
- pybFoam-gated tests self-skip (`pytest.importorskip("pybFoam")`); a skip is not
  a failure.
- mypy runs whole-tree and has 3 known pre-existing errors (`fields/schema.py`,
  `framework/solver/configurations.py`) — tolerated; only NEW errors in your
  changed files count.
- Commits use `git commit --no-verify` (the whole-tree mypy/reuse hooks fail on
  pre-existing issues) and **no** `Co-Authored-By: Claude` trailer.

> To run the whole loop with subagents instead of by hand, use the
> **implement-plan** skill (it dispatches neofoam-implementer + neofoam-verifier).
