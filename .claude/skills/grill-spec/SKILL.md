---
name: grill-spec
description: Interrogate the user to extract a complete, agreed set of requirements BEFORE any spec or design exists — a "grill me" loop that asks one sharp question at a time, refuses vague answers, grounds every question in the real code and docs, and drives toward shared understanding. Covers functional behavior AND the non-functional dimensions (data, interfaces, constraints, error/edge cases, scope boundary, acceptance, priorities, assumptions). Continues until a coverage map is satisfied and the user confirms a played-back restatement. Output is a requirements brief (plans/<name>-requirements.md) that feeds the write-spec skill. Use at the very start of a feature, when the ask is vague, or when a design proposal feels premature ("we're on the wrong track").
---

# Grill the spec out of the user

The fastest way to build the wrong thing is to start designing before you and the user
mean the same thing by the request. This skill is the **front-end to `write-spec`**: it
does not design and it does not write a spec — it **extracts and pins down requirements**
by interrogation, until you both share one understanding you can prove you share.

It works like being grilled: **one sharp question at a time, no vague answer survives,
nothing moves forward until the current point is concrete.** You are the one grilling —
on behalf of the eventual implementer who will have none of this conversation's context.

**Announce at start:** "I'm using the grill-spec skill — I'll ask questions until we
share the same understanding of the requirements, then hand off to write-spec."

## The grilling discipline (what makes this not a normal chat)

1. **One question at a time.** Never batch five questions into a paragraph. Ask the
   single most-load-bearing unknown, get the answer, then ask the next. Depth over
   breadth per turn.
2. **No vague answer survives.** "It should be fast" / "handle errors gracefully" /
   "be configurable" are not answers — they are prompts for the next question. Drill:
   *fast how — what latency, on what input size? Which errors, surfaced how, to whom?
   Configurable over what axis, with what default?* Convert every adjective into a
   number, a name, or a rule.
3. **Pin to something testable before moving on.** A point is "done" only when you could
   hand it to someone and they'd write the same acceptance check you would. If you can't
   name the test, keep grilling.
4. **Challenge, don't transcribe.** Play back the implication of an answer and let the
   user feel the consequence: "So a 600 MB STL would block the whole run — is that
   acceptable, or is there a size ceiling?" Surface contradictions with earlier answers
   out loud.
5. **Hunt the boundary.** Most defects live at the edges. Always ask what's *out* of
   scope, what the non-goals are, what should explicitly *not* happen, and what the
   system does on the bad/empty/huge/concurrent input.
6. **Don't ask what the code or docs already answer.** Read first (below). Spend the
   user's attention only on what the repo can't tell you — intent, priorities, trade-offs,
   unbuilt mechanisms.
7. **Track open threads.** Keep a running list of unresolved points and assumptions in
   view; never silently drop a thread because a new one opened.

## Ground first, then grill

Before the first question, read the real ground so your questions are sharp and you
never ask what's already settled:

- **Code** — the seams the feature touches (`Grep`/`Glob`/`Read`). Know the existing
  signatures, the data shapes, what already exists vs. what's new.
- **Docs** — the repo's documentation (`doc/`, `README`, `CLAUDE.md`, any `specs/` or
  `plans/`), so you inherit established vocabulary and constraints instead of reinventing
  them. Use the project's words for things.
- **Prior art** — adjacent features that already solved a similar problem; their choices
  are defaults you can offer ("the mesh stage is file-driven via `preprocess.yaml` —
  same pattern here, or different?").

Open by **restating the goal in one line and stating your reading of the situation**,
then let the user correct it. A wrong restatement the user fixes is worth more than a
question.

## How to ask

- Use **`AskUserQuestion`** at genuine forks with discrete options (architecture choice,
  scope boundary, priority call) — recommended option first, each option's consequence
  spelled out. It keeps the choice crisp and on the record.
- Use **plain prose questions** for open/unbounded points (numbers, rules, intent) where
  fixed options would bias the answer.
- When an answer has an obvious default, **state the default and ask only for a veto**
  ("I'll assume laminar for the skeleton unless you want turbulence in v1") — don't burn
  a turn on a foregone conclusion.

## The coverage map (grill every row to "pinned")

Hold this map and drive each row from *unknown* → *pinned* (testable). You are not done
while any MUST-level row is still vague. Not every row applies to every feature — mark
the ones that genuinely don't apply as *N/A* out loud, don't skip them silently.

| Dimension | What "pinned" means | Example sharp question |
|-----------|--------------------|------------------------|
| **Goal / user & job** | one sentence on who it's for and what job it does | "Who runs this, and what do they have before vs. after?" |
| **Functional behavior** | the input→output / state transitions, as rules | "Given X in, exactly what comes out?" |
| **Inputs & data** | every input, its type/shape/source, valid ranges | "What forms can the input take — file? object? both?" |
| **Interfaces & deps** | the surfaces it calls / is called by; what it relies on | "Does this reuse the existing tool registry or a new path?" |
| **Constraints (NFRs)** | perf, scale, env, platform, security — as numbers/limits | "What's the largest realistic input, and the time budget?" |
| **Error & edge cases** | bad / empty / huge / concurrent / partial-failure behavior | "What happens when a stage fails halfway — resume or restart?" |
| **Scope boundary / non-goals** | what's explicitly out, what must NOT happen | "What are we deliberately NOT doing in v1?" |
| **Acceptance / verification** | how we'll know it's correct — the demo or test | "What's the one run that proves this works?" |
| **Priorities** | MUST / SHOULD / COULD on each behavior | "If we ship only one of these, which?" |
| **Assumptions & risks** | what we're taking for granted; what could sink it | "What are we assuming is already true that might not be?" |

## Converge on shared understanding (the exit condition)

Grilling ends on **two** conditions, both required:

1. **Coverage** — every applicable map row is pinned (or explicitly deferred to an open
   question, on the record).
2. **Confirmed playback** — you give a **concise, structured restatement** of the whole
   requirement set (not a transcript — the distilled functional + non-functional
   requirements, priorities, scope boundary, and open questions) and the user explicitly
   confirms "yes, that's it." If they correct anything, fix it and play back again.

Play back **periodically**, not only at the end — every few topics, mirror back what you
heard so drift gets caught early. The final playback is the same act, complete.

Do **not** declare convergence yourself. The shared-understanding test is the user
agreeing with your restatement, not you deciding you've heard enough.

## Output — the requirements brief

When converged, write `plans/<name>-requirements.md`:

```markdown
# <Feature> — Requirements (agreed <date>)

## Goal
<one line: who, what job, before→after>

## Functional requirements
- FR1 · MUST · <testable behavior> — *Accept:* <how we'd verify>
- FR2 · SHOULD · …

## Non-functional requirements
- NFR1 · MUST · <constraint as a number/limit/rule> — *Accept:* …

## Data & interfaces
<inputs, shapes, the surfaces it touches — cite real symbols from the code>

## Scope boundary
**In:** … **Out / non-goals:** …

## Acceptance
<the one run / demo / test that proves it works end to end>

## Assumptions
<what we're taking as given>

## Open questions (deferred, not blocking)
<resolved-later items, each with who-decides>
```

Keep it the **agreed requirements**, not a design. No architecture, no chosen
implementation — those are `write-spec`'s and the planner's job. If the grilling
surfaced a design constraint the user insisted on, record it as an NFR, not as a design.

## Hand-off

The brief is the input to **`write-spec`**: its FR/NFR rows become the spec's prioritized
requirements with `Verify` cells, and its scope boundary becomes the spec's *Out of
scope*. When the brief is confirmed, offer to run `write-spec` against it (which in turn
feeds `spec-loop`). Chain: **grill-spec → write-spec → spec-loop**.

## Self-review checklist (before writing the brief)

- [ ] You **grounded in code + docs** first and didn't ask what the repo already answers.
- [ ] You asked **one question at a time** and let **no vague answer** through — every
      adjective became a number, name, or rule.
- [ ] Every applicable **coverage-map row is pinned** or explicitly deferred on the record.
- [ ] You hunted the **scope boundary, non-goals, and edge/error behavior** — not just the
      happy path.
- [ ] Every requirement is **testable** (you could name its acceptance check) and
      **prioritized** (MUST/SHOULD/COULD).
- [ ] You **played back** the full set and the **user confirmed** it — convergence is
      their call, not yours.
- [ ] The brief is **requirements, not design**; open questions are listed, nothing silently
      dropped.
