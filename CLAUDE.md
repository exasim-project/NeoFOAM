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

## Build & test

See **[Building & testing NeoFOAM](doc/reference/build-and-test.rst)** — environment
setup (pip workflow), scoped vs. full `pytest`, `pre-commit`, and the build gotchas
(non-editable install, never `rm -rf _skbuild`, pre-existing whole-repo failures,
tooling, CLI entry point).

Never delete or weaken an existing test to get green — surface the conflict instead.

## CLI

The `neofoam` command (`solver`, `preprocess`, `agent`, `mcp`, `telemetry`):
see **[doc/reference/cli.rst](doc/reference/cli.rst)**.

## Conventions

Detailed guides — read the relevant one before writing code or test

- **[`.claude/CODE_STYLE.md`](.claude/CODE_STYLE.md)** 
- **[`.claude/TEST_STYLE.md`](.claude/TEST_STYLE.md)** 

## Project layout (where things live)

The module map lives in the docs reference:
**[`doc/reference/project-layout.rst`](doc/reference/project-layout.rst)**

