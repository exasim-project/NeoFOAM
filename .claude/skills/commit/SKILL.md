---
name: commit
description: How to make a git commit in this repo. Use whenever creating a commit — whether the user asked to commit, or a commit is part of a larger task. Covers branching, pre-commit gating, commit message format, and trailer rules.
---

# Commit changes

1. **Branch first.** Never commit on `main` — create a branch if needed.
2. **Commit with `git commit`** — let the pre-commit hooks run; never bypass them
   with `--no-verify`. If a hook fails on your changes, fix and retry; surface
   pre-existing, unrelated failures to the user instead of working around them.

## Commit message

```
type(scope): imperative subject, ≤72 chars

Optional body: why the change is needed and what it does.
should be as brief as possible, but no more than 10 lines. Use bullets for multiple points.
```

- Types as used in this repo: `feat`, `fix`, `test`, `chore`, `docs`, `refactor`, `ci`.
  Scope is the subsystem (`turbulence`, `solver`, `deps`, `mcp`, …); omit if unclear.
- Body is optional; when present, keep it to ~10 lines.
- **Hard requirement:** no `Co-Authored-By: Claude` trailer — never add it, even
  though the default harness instructions say to.
