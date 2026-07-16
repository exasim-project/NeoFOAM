# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Unit tests for the pipe composition core: start states, ``|``, ``.build_at()``, forking.

Proves the two invariants the API rests on: ``|`` only ever composes (a Pipeline is
an immutable, reusable value) and ``.build_at()`` only ever materializes (independent dirs,
each run in isolation). The committed ``cavity`` template is staged into ``tmp_path``.
"""

from pathlib import Path

import pybFoam as pyf
import pytest

from neofoam.tooling.casebuild import block_mesh, empty, from_template, patch
from neofoam.tooling.casebuild.pipeline import pipe

CAVITY = Path(__file__).parent / "cases" / "cavity"


def _end_time(case_path: Path) -> float:
    d = pyf.dictionary.read(str(case_path / "system" / "controlDict"))
    return float(d.get_scalar("endTime"))


def test_empty_creates_bare_skeleton(tmp_path: Path) -> None:
    case = empty().build_at(tmp_path / "c")
    for sub in ("system", "constant", "0"):
        assert (case.path / sub).is_dir()
    assert not any((case.path / "0").iterdir())  # 0/ is empty — no seeded configs


def test_from_template_copies_and_restores_0orig(tmp_path: Path) -> None:
    case = from_template(CAVITY).build_at(tmp_path / "c")
    assert (case.path / "system" / "blockMeshDict").is_file()
    assert (case.path / "0" / "p").is_file()  # restored from 0.orig/


def test_pipe_operator_applies_steps_in_order(tmp_path: Path) -> None:
    case = (
        from_template(CAVITY)
        | patch("system/controlDict", endTime=1.0)
        | patch("system/controlDict", endTime=2.0)
    ).build_at(tmp_path / "c")
    assert _end_time(case.path) == pytest.approx(2.0)  # later step wins


def test_pipe_function_equals_operator(tmp_path: Path) -> None:
    op = (from_template(CAVITY) | patch("system/controlDict", endTime=3.0)).build_at(
        tmp_path / "a"
    )
    fn = pipe(from_template(CAVITY), patch("system/controlDict", endTime=3.0)).build_at(
        tmp_path / "b"
    )
    assert _end_time(op.path) == pytest.approx(3.0)
    assert _end_time(fn.path) == pytest.approx(3.0)


def test_pipeline_is_reusable_value(tmp_path: Path) -> None:
    recipe = from_template(CAVITY) | patch("system/controlDict", endTime=5.0)
    a = recipe.build_at(tmp_path / "a")
    b = recipe.build_at(tmp_path / "b")
    assert a.path != b.path
    assert _end_time(a.path) == pytest.approx(5.0)
    assert _end_time(b.path) == pytest.approx(5.0)


def test_fork_is_independent_and_reuses_mesh(tmp_path: Path) -> None:
    base = (from_template(CAVITY) | block_mesh()).build_at(tmp_path / "base")
    v1 = (base | patch("system/controlDict", endTime=1.0)).build_at(tmp_path / "v1")
    v2 = (base | patch("system/controlDict", endTime=2.0)).build_at(tmp_path / "v2")

    assert _end_time(base.path) == pytest.approx(0.01)  # base on disk untouched
    assert _end_time(v1.path) == pytest.approx(1.0)
    assert _end_time(v2.path) == pytest.approx(2.0)
    # forking copied the generated mesh — variants never re-mesh
    assert (v1.path / "constant" / "polyMesh" / "points").is_file()
