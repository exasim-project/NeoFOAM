# NeoFOAM Python package — Code style

# NeoFOAM Python package — Code style

Formatting and lint rules are enforced by tooling — never
hand-format, never discuss style the tools already own.

- init should only export public/exported function 
- private function only used within this package/module should start with underscore
- private modules only contain private functions and start with _
- Keep module docstrings to a single line. Prefer plain words over jargon.
- Keep private module/function/classes docstrings to a single line. Prefer plain words over jargon.
- Public module/function/classes docstrings: one-line summary, when to use it
  (and alternatives), invariants (thread safety, lifecycle),
  one usage example. Never restate the signature.
- Prefer small, single-purpose functions; prefer early returns
  over deep nesting.
- Don't abstract until there are ≥2 real call sites (no
  speculative interfaces, factories, or config options).
- Names describe intent, not implementation; no abbreviations
  except domain-standard ones.
- Prefer crashing over defensive recovery; only catch to add context (what failed, with what input), then re-raise.
- Comment only the "why" (constraints, tradeoffs, workarounds),
  never the "what".
- write strongly typed code