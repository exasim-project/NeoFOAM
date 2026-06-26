---
name: write-spec
description: Write or revise a software specification with clear, atomic, testable, prioritized (MUST/SHOULD/COULD) requirements, each tied to an acceptance test, with all API references grounded in the real codebase. Use when authoring a new spec (e.g. plans/<name>-spec.md), tightening a vague one, or reconciling a spec with what was actually implemented. The output is the input the spec-loop's planner consumes.
---

# Write a specification

A spec is a contract: it says exactly what the system must do, in statements you can
prove. It is **not** a design doc, a tutorial, or a wishlist. Write it so the
`spec-loop` planner can build a requirement-coverage table directly from it and a
reviewer can check each requirement against a test.

**Announce at start:** "I'm using the write-spec skill."

## The seven traits (every requirement must pass all seven)

1. **Specific & unambiguous** — state exactly what the system must do; no room for
   interpretation. "the fold over no values returns `VGREAT`" beats "handles the empty
   case gracefully."
2. **Testable / verifiable** — you can prove whether it's met. If you can't name the
   test, the requirement is too vague — rewrite it until you can.
3. **Necessary** — it ties back to a real user/business need, not "nice to have."
   Cut requirements that exist only because they're easy to write.
4. **Feasible** — achievable within the technical / time / budget constraints. If it
   depends on an unbuilt mechanism, say so and mark status.
5. **Atomic** — one requirement per statement. **Split every "and."** "Port CFL and
   delete the provider node" is two requirements; they're tracked and tested
   separately.
6. **Consistent** — no requirement contradicts another. After drafting, re-read for
   conflicts (e.g. one says "build-time error", another says "fold-time error").
7. **Prioritized** — tag **MUST** / **SHOULD** / **COULD** so trade-offs are explicit
   when time runs short. MUST = the feature is wrong without it; SHOULD = important but
   deferrable; COULD = opportunistic.

## Ground every API reference in real code

The fastest way to write a wrong spec is to invent the API. Before citing any symbol,
signature, error type, or timing:

- **Read the implementation** (or the upstream spec it depends on). Use `Grep`/`Glob`/
  `Read`. Cite real names and real behavior.
- When the spec **depends on** another mechanism, pin the exact public surface it uses
  and the **constraints that mechanism actually imposes** (e.g. "the consumer module
  must not use `from __future__ import annotations`", "the error is raised at fold time,
  not build time"). These are the details that make a dependent spec correct.
- If you are **reconciling a spec with an implementation**, diff the spec's claims
  against the code and fix every divergence — timing (build vs runtime), error types,
  names, and any "MUST" that turned out partial. Mark what shipped vs. what's deferred.
- Never silently upgrade an aspiration to a fact. If the mechanism isn't built, the
  requirement is `SHOULD`/`COULD` with an explicit **status/open-question** note.

## Recommended structure

1. **Title + Status + Scope** — one line each; note what it supersedes/depends on.
2. **Motivation** — the concrete problems (cite reviews/issues/commits). Why this is
   necessary (trait 3).
3. **Requirements** — a table per group with columns `ID | Priority | Requirement |
   Verify`. Stable IDs (`XX<n>`). The `Verify` cell is the test that proves it — this is
   the column the reviewer and planner key on.
4. **Consumer wiring / API** — short, real-code examples of how the thing is used;
   annotate the non-obvious constraints inline (the gotchas you grounded above).
5. **Acceptance criteria** — each maps to requirement IDs and is a test, not a
   sentiment. `ACn (TC4, TC7) …`.
6. **Out of scope / open questions** — what is explicitly deferred, and every unknown
   that would otherwise leak into implementation as a guess.

## Self-review checklist (run before finishing)

- [ ] Every requirement is **atomic** — grep your draft for " and " in requirement
      cells and split each one.
- [ ] Every requirement has a **priority** and a **Verify** entry you could actually
      write as a test.
- [ ] No two requirements **contradict** (timing, error type, ownership).
- [ ] Every API symbol/signature/error appears in the **real code** (or the upstream
      spec) — you read it, you didn't infer it.
- [ ] Anything depending on an **unbuilt** mechanism is marked SHOULD/COULD with a
      status note, not asserted as done.
- [ ] Acceptance criteria map back to requirement IDs.
- [ ] Out-of-scope lists every open question, so the implementer never guesses.

## Hand-off

A spec written this way drops straight into the `spec-loop`: the planner turns the
`ID | Priority | Verify` rows into its requirement-coverage table, sizing MUSTs first.
If you just authored or revised a spec, offer to run `spec-loop` against it.
