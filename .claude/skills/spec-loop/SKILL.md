---
name: spec-loop
description: Loop-based engineering from a spec — drives planner → implementer(self-verifying) → reviewer as fresh subagents that hand off versioned markdown artifacts under loop/<feature>/iter-N/, re-planning each iteration from the spec plus the previous review. Verifies but does NOT commit. Use to implement a numbered spec (e.g. plans/<name>-spec.md) incrementally with on-disk, inspectable progress.
---

# Spec Loop (artifact-handoff loop)

Drive a spec to completion in **iterations**, where every stage is a fresh subagent
that **reads the previous stage's markdown and writes its own**. All artifacts for
iteration `N` live together in `loop/<feature>/iter-N/`, so progress is inspectable
on disk and nothing is hidden in this conversation's context.

**Announce at start:** "I'm using the spec-loop skill to implement the spec."

## The loop (one iteration)

```
                                                                ┌─▶ design reviewer ──3-review.md──┐
planner ─1-plan.md─▶ implementer (builds,tests,lints) ─2-implement.md─┤                            │
   ▲                                                            └─▶ test reviewer ──4-test-review.md┤
   └────────────── next iteration re-plans from 3-review.md + 4-test-review.md ───────────────────┘
```

Each iteration produces a new `loop/<feature>/iter-N/` folder with up to four files:
`1-plan.md`, `2-implement.md` (which carries a `VERIFY` block — the implementer
builds, runs the targeted pytest, and lints its own work), `3-review.md` (design /
SOLID), and `4-test-review.md` (test coverage / edge cases). The **two reviewers run
in parallel** and are both read-only. The **planner's input is the spec(s) + the
previous iteration's `3-review.md` AND `4-test-review.md`** — so the loop adapts to
both design and coverage findings.

## Input

`$ARGUMENTS` — path to the spec (e.g. `plans/interface-spec.md`). Derive the
`<feature>` slug from the spec filename (kebab-case, drop `-spec`). If companion
specs are referenced (e.g. an approach doc, a consumer spec), pass them to the
planner/reviewer as context too.

## Setup (once)

1. Ensure you are on a feature branch (you usually already are in a worktree). If
   on `main`, branch first.
2. Discover the project's rules so the agents enforce *them* — read its conventions
   file (`CLAUDE.md` / `AGENTS.md` / `CONTRIBUTING.md`) and config (`Makefile`,
   `pyproject.toml`, `package.json`, `Cargo.toml`) for:
   - the **build/reinstall** step source changes need, if any (many projects need
     none);
   - the **test** command and how optional/native-dependency tests self-skip;
   - the **lint/format/type-check** commands, and which whole-repo errors are
     pre-existing (fail only on NEW errors in changed files).
   The agents read this themselves, but knowing it lets you scope each iteration.

## Per-iteration procedure

For iteration `N` (start at 1):

1. **Plan** — dispatch `spec-planner` (fresh) with: iteration `N`, the loop dir
   `loop/<feature>/`, the spec path(s), and (if `N>1`) the paths to
   `iter-<N-1>/3-review.md`, `iter-<N-1>/4-test-review.md`, and
   `iter-<N-1>/2-implement.md`. It writes `iter-N/1-plan.md` and returns the targets
   + test selector. **Read the plan's targets** so you know what this iteration is
   meant to land.
2. **Implement + verify** — dispatch `spec-implementer` (fresh) with the path to
   `iter-N/1-plan.md`. It edits code/tests, then **builds (only if the project needs
   it), runs the targeted tests, and runs the project's lint/type-check on the
   changed files**, self-retrying up to 3 times. It writes `iter-N/2-implement.md`
   with a `VERIFY` block and returns `RESULT: PASS|FAIL`. If `RESULT: FAIL` after its
   retries, stop and surface to the user — do not move on.
3. **Review** (only if `RESULT: PASS`) — dispatch **both reviewers in parallel**
   (one message, two Agent calls), each fresh, with the spec path(s) and the
   `iter-N/` handoff paths:
   - `spec-reviewer` → design / SOLID / conventions → `iter-N/3-review.md`.
   - `spec-test-reviewer` → test coverage / edge cases / uncovered functionality
     → `iter-N/4-test-review.md`.
   Collect both Net verdicts + findings. Fix any **CRITICAL** finding from *either*
   reviewer via one more implement→review cycle within this iteration; carry
   Important/Suggestion items from both into the **next** planner's input (that is
   the loop's whole point).

Keep your running summary terse — one line per stage (the artifact path + verdict).
Do **not** echo agent transcripts; the markdown handoffs are the record.

## Deciding to iterate again

After `3-review.md` and `4-test-review.md`, read their **remaining-requirements**
and **coverage-gap** notes. If spec requirements remain (or open findings warrant
it), run iteration `N+1` — the planner re-plans from both reviews. Stop when both
reviewers report all spec `MUST` requirements proven-by-test and no CRITICAL
findings.

> First run: do **one** iteration, then pause and report so the workflow itself can
> be inspected and improved before iterating further.

## Finalize (when iterating stops)

1. Run a **full** check yourself (the implementer self-verifies per iteration, but
   do one whole-suite pass at the end): rebuild only if source changed since the last
   build, then run the project's **whole** test suite + lint/type-check over all
   changed files. Record the result in the last `iter-N/2-implement.md` (a
   `## Final full check` section) or a short `loop/<feature>/final-check.md`.
2. If green, **stop without committing**. Print:
   - the per-iteration artifact tree (`loop/<feature>/`),
   - the changed-files summary (`git status --short`),
   - the open Important/Suggestion review notes,
   - and: *"Loop green. Review the diff and commit when ready, following the
     project's commit conventions (see its conventions file)."*

## Rules

- **Never commit** from this skill — verify only; the user commits.
- One fresh subagent per stage per iteration (no shared transcript); the **markdown
  handoff is the only channel** between stages.
- Respect the per-iteration attempt cap (5) — escalate, don't loop forever.
- If the spec itself is wrong or underspecified, stop and ask rather than guessing.
