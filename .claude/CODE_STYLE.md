# NeoFOAM Python package — Code style

## Tooling

- Formatting and lint rules are enforced by tooling — never
  hand-format, never discuss style the tools already own.

## Module & API structure

- `__init__.py` re-exports only the public API (functions and classes).
- Private functions only used within this package/module start with `_`.
- Private modules only contain private functions and start with `_`.

## Docstrings

- Module docstrings and private module/function/class docstrings: a single
  line. Prefer plain words over jargon.
- Public module/function/class docstrings: one-line summary, when to use it
  (and alternatives), invariants (thread safety, lifecycle),
  one usage example. Never restate the signature.

## Design

- Prefer small, single-purpose functions; prefer early returns
  over deep nesting.

## Naming

- Names describe intent, not implementation; no abbreviations
  except domain-standard ones for CFD/Engineering.

## Error handling

- Prefer crashing over defensive recovery; only catch to add context
  (what failed, with what input), then re-raise.

## Comments

- Comment only the "why" (constraints, tradeoffs, workarounds),
  never the "what".

## Typing

- Fully type-annotate all functions and methods; code must pass mypy.
