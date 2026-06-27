# Code guide (spec-loop implementer)

How to write the code for an increment. Project-agnostic — the project's own
conventions file (`CLAUDE.md` / `AGENTS.md` / `CONTRIBUTING.md`) always wins where
it is more specific; this guide fills the gaps.

## Match the project

- Follow the project's established idioms: frameworks, error handling, naming,
  comment density, file/module layout. Read the files you touch **and their
  neighbors** first, then write code that looks like it was already there.
- If the project runs a type checker, annotate fully and fix type errors **in
  code** — never by loosening the config. When a suppression is truly unavoidable,
  use the **correct** suppression code (a wrong code is itself a lint error).
- Mirror the project's test layout: where tests live, free functions vs. classes,
  fixtures, how optional/native dependencies are gated/skipped, and license/SPDX
  headers on new files.

## Scope discipline

- Implement **only** the planned increment. No drive-by refactors, no speculative
  abstractions (YAGNI). Keep diffs minimal and reviewable.
- If the plan looks wrong, note it in the handoff — don't silently diverge.

## No requirement IDs in tests

Tests assert **behavior**; the requirement→test mapping lives **only** in the
plan's coverage table. Do **not** put requirement IDs (`R1`, `FR3`, `IF7`, `TC4`,
…) anywhere in test files — not in per-line comments, section/heading comments,
docstrings, or test names. Name tests and grouping comments for the behavior they
prove. If the plan's task code carries such IDs, strip them as you transcribe it.
