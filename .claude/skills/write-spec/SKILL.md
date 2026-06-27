---
name: write-spec
description: Build a software specification interactively with the user — pick the template format that fits the situation (API/component, behavior change, CLI, bugfix, refactor/migration, data/schema), draft it section-by-section while discussing scope and priorities, and ground every API reference in the real codebase. The API/component template leads with a complete Architecture & API block (mermaid graph + every public interface + core dataclasses) and a handful of behavioral requirements. Use when authoring a new spec (e.g. plans/<name>-spec.md), tightening a vague one, or reconciling a spec with what was implemented. The output is the input the spec-loop's planner consumes.
---

# Write a specification

A spec is a contract: it says exactly what the system must do, in statements you can
prove. It is **not** a design doc, a tutorial, or a wishlist. Write it so the
`spec-loop` planner can build a requirement-coverage table directly from it and a
reviewer can check each requirement against a test.

A spec is **built with the user, not handed to them.** Work in passes, discuss the
shape, confirm priorities — don't dump a finished document and ask for a rubber stamp.

**Announce at start:** "I'm using the write-spec skill — let's build this together."

## Build it interactively — discuss, don't one-shot

Draft in passes; check in at each. Use **`AskUserQuestion`** at genuine forks (template
choice, scope boundary, priority calls, unresolved design decisions) — but skip
anything you can settle by reading the code or that has an obvious default (state the
default and move on). It is a conversation, not an interrogation form.

1. **Frame.** Restate the goal in one line. Establish the **situation** (→ which
   template) and the **scope boundary**. Ask only what the request and code don't
   already answer; propose your reading and let the user redirect.
2. **Ground.** Read the real code for the seams the spec touches (see *Ground every API
   reference* below). Surface what you found — especially any **infeasibility** or
   dependency on an unbuilt mechanism — *before* writing requirements, so the user can
   redirect scope early.
3. **Draft the lead block.** Write the template's normative lead (for API/component: the
   Architecture & API block). Show it and get feedback — this is where most of the
   discussion lives (names, what's public, where the boundary sits).
4. **Draft the behaviors.** Propose the handful of requirements + a **MUST/SHOULD/COULD**
   split. Priorities are the user's call — confirm them, don't assume.
5. **Confirm open questions.** List the unknowns and deferrals. Ask the user to resolve
   the ones that change the design; leave the rest explicitly in *Out of scope / open
   questions*.
6. **Finalize** and offer to run `spec-loop` against it.

For a quick or small spec, collapse passes — but still confirm the template and the
priority split before finalizing.

## Pick a template for the situation

Different work needs different specs. Pick the closest template; **combine** when a
feature spans two (e.g. a CLI command backed by a new API). If the situation is
ambiguous, ask the user with `AskUserQuestion` (recommended option first). Every
template shares the spine — *Title/Status/Scope · Motivation · a normative lead block ·
a handful of prioritized requirements with `Verify` cells · Out of scope / open
questions* — only the **lead block** and the **requirement focus** change.

| Template | Use when | Lead block | Requirements focus |
|----------|----------|------------|--------------------|
| **API / component** *(default)* | adding or changing a code surface others call | the 3-part **Architecture & API** block (below) | behaviors the signatures can't show |
| **Behavior / algorithm** | changing what existing code *does*, little new surface | input→output / state-transition **tables** + the changed signatures only | the rules, edge cases, invariants |
| **CLI / user-facing** | a command, flag, or interaction | command/flag **signatures** + example invocations + the output/exit-code contract | behavior per flag, error messages, exit codes |
| **Bugfix / defect** | fixing a specific defect | a **reproduction** (steps + expected vs actual) + the root cause | the invariant that must now hold + a regression test (often 1–2 rows) |
| **Refactor / migration** | moving/renaming with **no** behavior change | a from→to **move-map** + a removal list | parity/equivalence (behavior unchanged) + what's deleted |
| **Data / schema / protocol** | a config, schema, or wire format | the **schema as types** + serialization forms | validation rules, compatibility, round-trip |

### The API / component lead block (the default, in full)

When the spec defines an API the **signature block is normative** — it already states
the names, types, decorators, and call shape. The lead block must be **complete and
carry the bulk of the spec**: a reader understands *what exists* from it alone. Three
parts:

1. **A `mermaid` dependency graph** of the public components (modules/classes) and the
   edges between them — who owns/calls/imports/contributes-to whom. The map scanned first.
2. **Every public interface** — the full signature of each public function/method/class
   the feature adds or changes (names, params, return types, decorators), each with a
   one-line purpose. Not a sample — all of them.
3. **The core data structures as dataclasses/types** — real `@dataclass` / `Literal` /
   `TypedDict` declarations (fields + types), so the shape is exact, not prose.

Plus one **worked usage example** annotated inline with the non-obvious constraints you
grounded (the gotchas).

## Lead with shape, keep requirements few

Whatever the template, the **lead block carries the shape; requirements carry only the
behavior** the lead can't encode. A requirement that re-narrates a signature is noise —
it triples the word count (lead → requirement → acceptance) without adding a provable
claim.

A requirement earns its row only if it adds at least one of:
- **priority** (MUST/SHOULD/COULD) on a behavior;
- **behavior/algorithm** the shape doesn't determine (the fold is `min`; empty → `VGREAT`);
- **timing** (build vs run vs fold time);
- **error / edge semantics** (what raises, when, with what message);
- **an invariant** (per-case isolation, OCP, idempotence, ordering);
- **a removal / migration** (symbol X is deleted).

**Litmus test:** cover the lead block with your hand — can you still write the
requirement's `Verify` test from it alone? If the shape fully determines it, delete the
requirement and let the lead stand, or compress it to the behavioral delta and **cite
the symbol** instead of re-describing it.

**Keep the requirements few** — a handful, not dozens. 25+ granular rows means shape
leaked into the requirements; fold it back into the lead block. Each row is a
*consolidated* assertion with its own `Verify` test (the `Verify` cell **is** the
acceptance test — no separate acceptance section that re-bundles IDs).

Before (narrates the signature — cut it):
> MI1 — A model declares an interface via `@<model>.interface` decorating the fold;
> the decorated name is the interface name and the function is the single fold…

After (only the part the signature can't show):
> MI1 · MUST · Empty fold returns the declared default. *Verify:* `fold([]) == VGREAT`.

## The seven traits (every requirement must pass all seven)

1. **Specific & unambiguous** — "the fold over no values returns `VGREAT`" beats
   "handles the empty case gracefully."
2. **Testable / verifiable** — if you can't name the test, it's too vague; rewrite it.
3. **Necessary** — ties to a real need, not "nice to have."
4. **Feasible** — achievable within the constraints; if it depends on an unbuilt
   mechanism, say so and mark status.
5. **Atomic** — one requirement per statement. **Split every "and."**
6. **Consistent** — no requirement contradicts another (timing, error type, ownership).
7. **Prioritized** — **MUST** (wrong without it) / **SHOULD** (deferrable) / **COULD**
   (opportunistic).

## Ground every API reference in real code

The fastest way to write a wrong spec is to invent the API. Before citing any symbol,
signature, error type, or timing:

- **Read the implementation** (or the upstream spec it depends on). Use `Grep`/`Glob`/
  `Read`. Cite real names and real behavior.
- When the spec **depends on** another mechanism, pin the exact public surface it uses
  and the **constraints that mechanism imposes** (e.g. "the consumer module must not use
  `from __future__ import annotations`", "the error is raised at fold time, not build
  time"). These details make a dependent spec correct.
- When **reconciling a spec with an implementation**, diff the spec's claims against the
  code and fix every divergence — timing, error types, names, any "MUST" that turned out
  partial. Mark what shipped vs. deferred.
- Never silently upgrade an aspiration to a fact. If the mechanism isn't built, the
  requirement is `SHOULD`/`COULD` with an explicit **status/open-question** note.

## Self-review checklist (run before finishing)

- [ ] The **template fits the situation**, and you **confirmed it + the priority split
      with the user** (this is interactive work, not a one-shot drop).
- [ ] The **lead block is complete** for its template — for API/component: a `mermaid`
      graph, *every* public interface, and the core dataclasses; a reader understands
      what exists from it alone.
- [ ] **Requirements are few** (a handful, not dozens) and **none restates a signature**
      already in the lead block (apply the litmus test).
- [ ] Every requirement is **atomic**, **prioritized**, and has a **`Verify`** test you
      could actually write; no two **contradict**.
- [ ] Every API symbol/signature/error appears in the **real code** (or upstream spec) —
      you read it, you didn't infer it.
- [ ] **No standalone acceptance section that re-bundles requirement IDs** — `Verify`
      cells are the acceptance tests; keep only genuine cross-requirement E2E scenarios.
- [ ] Anything depending on an **unbuilt** mechanism is SHOULD/COULD with a status note.
- [ ] Out-of-scope lists every open question, so the implementer never guesses.

## Hand-off

A spec written this way drops straight into the `spec-loop`: the planner turns the
`ID | Priority | Verify` rows into its requirement-coverage table and seeds its
architecture map from the lead block. If you just authored or revised a spec, offer to
run `spec-loop` against it.
