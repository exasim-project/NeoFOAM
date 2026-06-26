---
name: implement-plan
description: Execute an approved implementation plan as a plan→implement→verify loop using subagents, tuned for the NeoFOAM Python package (non-editable rebuild, pytest, pre-commit). Use after writing-plans when you have an approved plan to implement. Drives each task with the neofoam-implementer + neofoam-verifier (+ neofoam-reviewer) agents; verifies but does NOT commit.
---

# Implement Plan (the loop)

Drive an approved plan to completion, one task at a time, using fresh subagents so
build logs and implementation churn stay out of this conversation. The loop is
**implement → verify → (review) → next**, with a final full verification. It
**verifies but never commits** — you hand the green, reviewed diff back to the user.

**Announce at start:** "I'm using the implement-plan skill to execute the plan."

## Input

`$ARGUMENTS` — path to the approved plan. If empty, use the most recent
`~/.claude/plans/*.md` (or ask which plan if ambiguous).

## Setup

1. Read the plan in full. Build a **task list** (use the Task tools) from its
   steps — `- [ ]` checkboxes if the plan has them, otherwise one task per
   "Files to change" row / phase. Keep tasks small and ordered.
2. Ensure you are on a feature branch. If on `main`, create one before any edits.
3. Note the project rules (also enforced by the agents):
   - Non-editable install: `src/neofoam/*.py` edits need `pip install .[all] -v`
     to take effect. **Never** wipe `_skbuild`. Never `uv run`/`uv sync`.
   - pybFoam-gated tests skip cleanly without pybFoam.
   - mypy has 3 known pre-existing whole-tree errors — tolerated; fail only on
     NEW errors in changed files.

## Per-task loop

For each task, mark it in-progress, then:

1. **Implement** — dispatch the `neofoam-implementer` agent (fresh) with: the task
   (files + behavior + any plan code snippets), and a pointer to `CLAUDE.md`. It
   returns CHANGED files + a TEST SELECTOR.
2. **Verify** — dispatch the `neofoam-verifier` agent in **smart** mode with that
   test selector. It rebuilds only if `src/neofoam` changed, runs the targeted
   tests, and returns `RESULT: PASS|FAIL`.
3. **If FAIL** — re-dispatch `neofoam-implementer` with the verifier's FAILURES
   block; re-verify. **Cap at 5 attempts** per task; if still failing, stop and
   surface the failure to the user — do not move on.
4. **If PASS** — (optional but recommended) dispatch `neofoam-reviewer` on the
   changed files; fix any **CRITICAL** items (back through implement→verify) and
   note Important/Suggestions for the user. Mark the task done.

Keep the running summary terse — one line per task (PASS, what ran). Don't echo
agent transcripts.

## Finalize

1. Run `neofoam-verifier` in **full** mode: whole `pytest` + `SKIP=reuse
   pre-commit run --files <all changed files>`.
2. If green, **stop without committing**. Print:
   - the task list (all done),
   - the changed-files summary (`git status --short`),
   - any Important/Suggestion review notes,
   - and: *"Loop complete and green. Review the diff and commit when ready —
     use `git commit --no-verify` and no `Co-Authored-By: Claude` trailer (the
     whole-tree mypy/reuse hooks fail on pre-existing issues)."*
3. If the final full run surfaces a failure the per-task loop missed, fix it via
   one more implement→verify cycle before declaring done.

## Rules

- **Never commit** from this skill — verify only; the user commits.
- One task per implementer dispatch; fresh agent each time (no shared transcript).
- Respect the per-task attempt cap (5) — escalate to the user instead of looping
  forever.
- If the plan itself is wrong or underspecified, stop and ask rather than guessing.
