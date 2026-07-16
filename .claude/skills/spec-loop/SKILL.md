---
name: spec-loop
description: Loop-based engineering from a spec — decomposes the spec once into a machine-checkable backlog of thin vertical slices (each with executable acceptance checks), then lands ONE slice per iteration via planner → implementer → risk-tiered review, all fresh subagents handing off markdown under loop/<feature>/. Done = acceptance checks executed with pasted evidence, never "looks done". Verifies but does NOT commit. Use to implement a numbered spec (e.g. plans/<name>-spec.md) incrementally with on-disk, inspectable progress.
---

# Spec Loop (slice-at-a-time, evidence-gated)

Drive a spec to completion in **iterations**: decompose once into a backlog of thin
vertical slices, then each iteration lands **one slice, fully, to green** — proven by
**executed acceptance checks with pasted evidence**, not by assertion. Every stage is a
fresh subagent that reads the previous stage's markdown and writes its own; all
iteration-N artifacts live in `loop/<feature>/iter-N/`.

**Announce at start:** "I'm using the spec-loop skill to implement the spec."

## Design rationale (why these rules exist)

- **Fresh agents have no memory** — the loop's state must live in artifacts (backlog,
  research, decisions), or agents re-derive or contradict earlier work.
- **"Looks done" is the only signal an unchecked agent has** — so done is defined as
  *executable checks passing with evidence*, and deferral inside a slice is illegal.
- **A gap-hunting reviewer always finds gaps** — unbounded review causes oscillation
  and over-engineering; findings must be grounded in the contract or an executable
  check, and review-fix cycles are capped.
- **Loops without terminal conditions invent work** ("overbaking") — the exit is
  machine-checkable, and the loop never idles past it.
- **The loop costs multiples of a solo session** — its value is completeness and
  unattended operation on specs too big for one context, not efficiency. Right-size
  the machinery per slice; don't run the full pipeline on trivial work.

## Living artifacts (`loop/<feature>/`, updated in place)

| File | Owner | Purpose |
|---|---|---|
| `backlog.md` | planner | **Plan-of-record**: ordered slices, each with requirements covered, dependency, **risk tier**, **acceptance checks (commands + expected outcome)**, and `passes: no|yes`. After creation, the only legal mutations are flipping `passes`, inserting a fix-slice, or splitting a slice — **never deleting or weakening a slice's checks**. |
| `research.md` | planner (iter 1) | One-time codebase research: the files, functions, seams, and dataflow this spec touches, with `path:line`. Written once so later agents don't re-derive it; append corrections only. |
| `architecture.md` | planner authors, implementer reconciles | Public-interface + dependency map (mermaid + signatures + key types) of the feature. Target vs shipped. |
| `decisions.md` | anyone (append-only) | ADR-lite log: decision, why, alternatives rejected, open `[NEEDS CLARIFICATION]` items and their resolutions. **No agent may relitigate a recorded decision without a backlog item.** |
| `iter-N/1-plan.md`, `2-implement.md`, `3-review.md`, `4-test-review.md` | per stage | The per-iteration handoffs. |

## Input

`$ARGUMENTS` — path to the spec (e.g. `plans/interface-spec.md`). Derive `<feature>`
from the spec filename (kebab-case, drop `-spec`). Pass companion specs as context too.

## Setup (once)

1. Feature branch (if on `main`, branch first).
2. Read the project's conventions file (`CLAUDE.md` etc.) for build/test/lint commands
   and pre-existing-failure caveats.
3. Ensure a **runnable entry** exists: if the project lacks a one-command way to
   build+smoke-test, have the first slice create `loop/<feature>/init.sh` (idempotent:
   env → build if needed → fast smoke command). Every later agent runs it instead of
   rediscovering the environment.

## Per-iteration procedure

For iteration `N` (start at 1):

0. **Smoke-check the base** (you, cheap): run the fast smoke command / targeted suite
   before planning. If the base is broken, the next slice is a **fix-slice** — never
   plan new work on a red base.
1. **Plan one slice** — dispatch `spec-planner` (fresh) with: iteration `N`, loop dir,
   spec path(s), and (N>1) the previous `3-review.md`/`4-test-review.md`/`2-implement.md`.
   - Iter 1: it writes `research.md`, `backlog.md` (slices **with acceptance checks and
     risk tiers**), `architecture.md`, then `iter-1/1-plan.md` for slice #1.
   - Iter N>1: it flips the finished slice's `passes`, folds review findings (only
     contract-grounded ones — see step 3), and plans the **next single slice**.
   - The plan contains the **contract**: the exact acceptance checks + review rubric
     this slice will be judged by. Read the returned targets + risk tier.
2. **Implement + verify** — dispatch `spec-implementer` (fresh) with `iter-N/1-plan.md`.
   It completes the **whole slice** (no stubs/deferral), executes the slice's
   **acceptance checks**, and writes `2-implement.md` whose `VERIFY` block pastes the
   commands run and their real output. Returns `RESULT: PASS|FAIL`; a too-big slice
   returns `FAIL reason: re-slice` (the planner splits it next iteration) — never
   partial work reported as PASS. On `FAIL` after its retries: stop, surface to user.
3. **Review, tiered by risk** (only on PASS):
   - **trivial** tier (mechanical, ≤1 file + tests, no new public surface): **skip the
     reviewers**. The acceptance-check evidence is the gate. Optionally batch: the
     planner may put 2–3 *adjacent trivial* slices in one plan.
   - **standard** tier: dispatch `spec-reviewer` only.
   - **high** tier (new public surface / core algorithm / cross-module): dispatch
     **both reviewers in parallel** (one message, two Agent calls).
   Reviewers judge **only against the plan's contract + the spec requirements it
   traces**; a finding that cites neither a requirement nor a failing/missing
   executable check is a non-blocking Suggestion. Fix CRITICALs via **at most 2**
   implement→review cycles within the iteration; if still contested, stop and
   escalate to the user (don't grind). Suggestions enter future plans only if the
   planner promotes them to backlog items.

Keep your running summary terse — one line per stage (artifact + verdict + slice k/M).
Do **not** echo agent transcripts.

## Terminal condition (machine-checkable — then STOP)

The loop exits when **all three** hold:
1. every backlog slice has `passes: yes`;
2. the project's **whole** test suite + lint/type-check are green (run it yourself;
   rebuild only if needed);
3. one **final acceptance review**: a single fresh `spec-reviewer` pass scoped to the
   **spec vs the full diff** (not a slice) — per-slice gates can all pass while a
   cross-cutting requirement fell through; this holistic pass is the complement.

Record the full check in `loop/<feature>/final-check.md`. Then **stop — never run
another iteration after the exit condition holds** (a loop past done invents work).
Print: the backlog (all `passes: yes`), `git status --short`, open Suggestions, and
*"Loop green. Review the diff and commit when ready, following the project's commit
conventions."* **Never commit** — the user commits.

## Rules

- **One slice per iteration, finished.** Deferral exists only as a future backlog
  slice. No stubs/TODO/`NotImplementedError`/skipped tests standing in for this
  slice's behavior.
- **Evidence, not assertions.** Every PASS is backed by pasted command+output in the
  handoff. **Deleting or weakening a test or acceptance check to get green is
  forbidden** — that is the one unforgivable move.
- **Ambiguity goes to `decisions.md`, not into code.** A planner that hits an
  underspecified point writes `[NEEDS CLARIFICATION]`; either the user answers or the
  assumption is recorded explicitly. Don't silently guess; don't relitigate recorded
  decisions.
- One fresh subagent per stage; the markdown handoff is the only channel.
- Caps: 3 implementer self-retries; 2 review-fix cycles; 5 attempts per iteration —
  escalate, don't loop forever.
- If the spec itself is wrong, stop and ask.

> First run: do **one** iteration (research + backlog + slice #1), then pause and
> report so the decomposition can be inspected before iterating further.
