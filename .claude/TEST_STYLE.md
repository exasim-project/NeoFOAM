# Test rules (`test/`)

## Placement

- Unit tests mirror `src/neofoam/` 1:1: `turbulence/selection.py` → `test/turbulence/test_selection.py`.
- Integration/e2e tests are named for the scenario (`test_neon_turbulence_parity.py`), never reusing a source-module name.
- Prefer the lowest level that can prove the behavior — no solver run for a config check.
- Tests are free functions, not classes. `test/io/` has no `__init__.py` (shadows stdlib `io`); `test/framework/` has one.

## A good test

1. **Proves one named behavior** — `test_<subject>_<behavior>`; if the name needs "and", split it.
2. **Reads as setup → act → assert** — the call to the thing under test is one visible line, not buried in a helper.
3. **Inputs are explicit and on disk** — real case files under `test/<area>/cases/<name>/`, never file content as strings/f-string templates in the test. Vary content via the format's own reader/writer, not text patching.
4. **Never mutates checked-in files** — `shutil.copytree` the case into `tmp_path`.
5. **Builds objects the way production does** — select from the real case dict, don't hand-construct with fabricated arguments.
6. **Expectations are pre-written data** — literals in the test/`parametrize` entry, or the case's `expected.yaml`; never computed by the code under test itself.
7. **Numeric asserts have justified tolerances** — `assert_allclose` with explicit `rtol`/`atol` tied to something physical (linear-solver tol, field peak), and an `err_msg` naming the case.
8. **Deterministic and independent** — seeded RNG, no reliance on test order or leftover state.
9. **Docstring carries the design rationale** (why no walls, why subprocess-per-run, why this tolerance) — once, at module level. SPDX header on every file.
10. **Adding coverage is data, not code** — new case directory or `parametrize` entry, no new test body.

## Hard constraints

- pybFoam/OpenFOAM are hard deps: no skip-if-missing, no `try/except` imports. Only optional extras (telemetry, agent, MCP) use `pytest.importorskip`.
- One `Foam::Time` per process — multiple runs (reference vs subject) go through subprocess workers that write `.npy` artifacts.
- Non-editable install: source changes need `pip install .[all] -v` before pytest sees them.
- Never delete or weaken an existing test to get green.

## Done

`pytest test/<area> -q` passes and you've seen the new test fail once (break the code) so you know it can.
