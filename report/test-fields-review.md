<!--
SPDX-License-Identifier: GPL-3.0-or-later
SPDX-FileCopyrightText: 2026 NeoFOAM authors
-->

# Review: `test/fields/` against `TEST_STYLE.md`

Scope: the test modules under `test/fields/` plus the shared `hotRoom` fixture,
judged against `.claude/TEST_STYLE.md`. Verdict: the *content* of these tests is
strong — real fixture on disk for the IO path, literal expectations, one behavior
per test, good docstrings, SPDX everywhere. The problems are almost all about
**placement / 1:1 mirror** and a few **weak assertions / in-function imports**.

---

## 1. Placement violations (the main issue) — rule "mirror `src/neofoam/` 1:1"

Three modules sat in `test/fields/` but test source that lives under
`src/neofoam/framework/`, and homes for that source *already existed*:

| test file | source it actually tests | correct home (exists) |
|---|---|---|
| `test_spec_field.py` | `framework/model/spec.py` (`Model.field`) | `test/framework/model/test_spec.py` |
| `test_configurations.py` | `framework/solver/configurations.py` | `test/framework/solver/test_configurations.py` |
| `test_auto_synthesis.py` (the `run_build` half) | `framework/model/runtime.py` (`ModelRuntime.run_build`) | `test/framework/model/test_spec.py` |

Consequences:

- **Two homes for one module.** `framework/solver/configurations.py` was tested by
  both `test/fields/test_configurations.py` *and* the pre-existing
  `test/framework/solver/test_configurations.py` — two files with the **same
  basename** for the same source. The field-specific surface (`_is_field_schema`,
  the `.fields` / `.dicts` filters) belongs *added to* the framework file.
- **`run_build` tested in two places.** `test/framework/model/test_spec.py`
  already had `test_run_build_*`; `test_auto_synthesis.py` added four more.
- `Model.field` registration (`test_spec_field.py`) had no test in
  `test/framework/model/test_spec.py` — so this wasn't overlap, it was a test that
  landed in the wrong directory.

## 2. Naming — rule "`turbulence/selection.py` → `test/turbulence/test_selection.py`"

- `test_auto_synthesis.py` mirrors `fields/synthesis.py`, so the file should be
  **`test_synthesis.py`**. As-is the name also advertised a second subject
  (`run_build`) — the smell that flagged the mixed-concern problem in §1.
- `test_in_tree_models.py` is a legitimate integration test ("named for the
  scenario"). Name acceptable; kept.

## 3. Duplicated setup helpers — rule 2 (DRY setup is fine, but this was copy-paste)

- `_SyntheticSolver` duck-type was **defined twice**, verbatim, in
  `test_configurations.py` and `test_loader.py`.
- `FIXTURE` + `_stage(tmp_path)` were **identical** in `test_loader.py` and
  `test_in_tree_models.py`.

## 4. Weak / broad assertions — rule "assertions prove the behavior"

- `test_decl.py::test_field_decl_is_immutable` — `pytest.raises(Exception)`.
- `test_value_types.py::test_field_value_scalar_rejects_vector` —
  `_pytest.raises(Exception)`.
- Several `pytest.raises(ValueError)` / `TypeError` with no `match=`
  (`test_schema.py`, `test_bc.py::test_build_bc_union_empty_raises`).

## 5. In-function imports — CODE_STYLE "imports at top"

- `test_auto_synthesis.py`: `lazy` imported inside two test bodies.
- `test_configurations.py`: `schema_for` inside a test.
- `test_loader.py`: `from pydantic import BaseModel` inside a test.
- `test_value_types.py`: `import pytest as _pytest` inside a test (`pytest` already
  imported at top); `from typing import Any` out of isort order.

## 6. Minor / judgment calls (no change required)

- **Inline dicts in `test_bc.py` / `test_schema.py`.** These pass dicts to
  `model_validate` — the pydantic API's natural input, not OpenFOAM dict-as-text —
  so within bounds. The real disk round-trip is delegated to `test_loader.py`.
- **`test_bc.py` round-trip family** already `parametrize`s topology + wall-function
  arms; value-carrying arms are separate because payloads differ. Reasonable.
- **Overlap `test_loader.py` ↔ `test_in_tree_models.py`** is a deliberate
  unit-vs-integration split (synthetic solver vs real `incompressibleFluid`).

---

## Applied (2026-07-12)

All items above were applied. Baseline and post-change test counts match (147
passed across `test/fields` + the two framework targets), confirming tests were
**relocated, not lost or duplicated**. `pre-commit` (ruff format, ruff, mypy,
typos, reuse) is clean on every touched file.

- **§1 placement / §2 naming:**
  - `test_spec_field.py` → folded into `test/framework/model/test_spec.py`
    ("Cycle 7 — Model.field(...) declarations"); original deleted.
  - `test/fields/test_configurations.py` → field-schema cases folded into
    `test/framework/solver/test_configurations.py` (new "Field-schema surface"
    section); original deleted.
  - `test_auto_synthesis.py` → renamed `test_synthesis.py` (keeps the
    `synthesize_init_step` tests); the four `run_build` cases moved to
    `test/framework/model/test_spec.py` ("Cycle 8 — run_build auto-synthesizes
    field steps"), where `ModelRuntime` already lives.
- **§3 de-dup:** `FIXTURE` + `_stage` replaced by a `staged_hot_room` fixture in a
  new `test/fields/conftest.py`, shared by `test_loader.py` and
  `test_in_tree_models.py`.
- **§4 assertions:** `FrozenInstanceError` in `test_decl.py`; `ValidationError` in
  `test_value_types.py`; `match=` added to the empty-`allowed_bcs`,
  unsupported-value-type (`test_schema.py`) and empty-union (`test_bc.py`) raises.
- **§5 imports:** hoisted `BaseModel` (`test_loader.py`), `schema_for`
  (framework configurations test), `lazy` (relocated with the `run_build` tests);
  dropped the in-body `import pytest as _pytest` and fixed `typing` import order in
  `test_value_types.py`.

**Accepted residual:** the `_SyntheticSolver` duck-type now exists once in
`test_loader.py` and once in `test/framework/solver/test_configurations.py`. These
are different suites with different purposes; hoisting it to a `test/`-root
`conftest.py` would couple the two trees, so it was left duplicated on purpose.
