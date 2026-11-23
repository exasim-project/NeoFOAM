# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""setAlphaField tool: parity with the OpenFOAM utility, restart skip, script errors.

**Case.** ``test/preprocess/cases/box`` overlaid with ``cases/set_alpha_field``: a unit
box cut into 16x16x1 cells and a cylinder along z, of radius 0.3 about (0.5, 0.1), so the
interface also crosses the wall y = 0. The walls are ``fixedValue`` because that type
stores its face values in the field file, which makes the wetted fractions comparable
from disk; a ``zeroGradient`` patch writes none.

**Oracle.** The real ``setAlphaField`` utility, run on a second copy of the case with the
equivalent ``system/setAlphaFieldDict``. Both sides write 18 digits and run the same cut,
but compute the level set in different code (numpy, ``implicitFunction``), so they agree
to round-off of the geometry rather than bit for bit: ``atol = 1e-12`` on fractions in
[0, 1].
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose

from neofoam.framework.tools import ToolRuntime
from neofoam.tooling.casebuild import CaseDir, patch
from neofoam.tools import available_tools
from neofoam.tools.run import run_preprocess
from neofoam.tools.set_alpha_field import SetAlphaFieldStep, setAlphaFieldTool

BOX = Path(__file__).parents[1] / "preprocess" / "cases" / "box"
CASES = Path(__file__).parent / "cases"
OVERLAY = CASES / "set_alpha_field"
NO_FUNCTION = CASES / "set_alpha_field_no_function"
RESTART_OVERLAY = CASES / "set_fields_restart"

#: ``cases/set_fields_restart/0.1``: what the earlier run left behind.
_RESTART_ALPHA = 0.75
_RESTART_TIME = "0.1"
_N_CELLS = 256

_NONUNIFORM_VALUE = re.compile(
    r"\bwalls\s*\{.*?value\s+nonuniform\s+List<scalar>\s*\d+\s*\((.*?)\)", re.DOTALL
)


class _FakeTime:
    def timeName(self) -> str:
        return "0"


class _FakeMesh:
    """A mesh that is nothing but its time: the error paths stop before using it."""

    def time(self) -> _FakeTime:
        return _FakeTime()


def _stage(dest: Path) -> Path:
    shutil.copytree(BOX, dest)
    shutil.copytree(OVERLAY, dest, dirs_exist_ok=True)
    return dest


def _openfoam_reference(dest: Path) -> Path:
    """A copy of the case initialised by the real OpenFOAM utility."""
    case = _stage(dest)
    for utility in ("blockMesh", "setAlphaField"):
        subprocess.run([utility, "-case", str(case)], check=True, capture_output=True)
    return case


def _wall_values(case: Path) -> np.ndarray[Any, Any]:
    """The ``walls`` face values of ``0/alpha.water``, as OpenFOAM wrote them."""
    match = _NONUNIFORM_VALUE.search((case / "0" / "alpha.water").read_text())
    assert match is not None, f"{case}: walls hold no nonuniform value list"
    return np.array(match.group(1).split(), dtype=float)


def _step() -> Any:
    runtime = ToolRuntime(
        spec=setAlphaFieldTool,
        name="preprocess.setAlphaField",
        config=SetAlphaFieldStep(tool="setAlphaField"),
    )
    return runtime.run_build()[0]


def test_set_alpha_field_is_registered() -> None:
    by_name = {tool.name: tool for tool in available_tools()}

    assert by_name["setAlphaField"] is setAlphaFieldTool


def test_preprocess_cells_match_the_openfoam_utility(tmp_path: Path) -> None:
    reference = _openfoam_reference(tmp_path / "reference")
    case = _stage(tmp_path / "case")

    run_preprocess(["preprocess", "-case", str(case)])

    assert_allclose(
        CaseDir(case).read_field("alpha.water", time="0"),
        CaseDir(reference).read_field("alpha.water", time="0"),
        rtol=0,
        atol=1e-12,
        err_msg="cases/set_alpha_field: cell volume fractions",
    )


def test_preprocess_patches_match_the_openfoam_utility(tmp_path: Path) -> None:
    reference = _openfoam_reference(tmp_path / "reference")
    case = _stage(tmp_path / "case")

    run_preprocess(["preprocess", "-case", str(case)])

    assert_allclose(
        _wall_values(case),
        _wall_values(reference),
        rtol=0,
        atol=1e-12,
        err_msg="cases/set_alpha_field: wetted area fractions of walls",
    )


def test_preprocess_leaves_a_restarted_runs_alpha_alone(tmp_path: Path) -> None:
    case = CaseDir(_stage(tmp_path / "case"))
    shutil.copytree(RESTART_OVERLAY / _RESTART_TIME, case.path / _RESTART_TIME)
    patch("system/controlDict", startTime=float(_RESTART_TIME))(case)

    run_preprocess(["preprocess", "-case", str(case.path)])

    # read_field stages the requested time as 0/, so the start time has to point back.
    patch("system/controlDict", startTime=0.0)(case)
    assert_allclose(
        case.read_field("alpha.water", time=_RESTART_TIME),
        np.full(_N_CELLS, _RESTART_ALPHA),
        err_msg=f"restart at {_RESTART_TIME}: alpha.water must not be re-initialised",
    )


def test_a_script_without_set_alpha_is_a_named_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    shutil.copytree(NO_FUNCTION, tmp_path, dirs_exist_ok=True)
    (tmp_path / "0").mkdir()
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match=r"must define a function set_alpha\(mesh\)"):
        _step().initializer({"_prev_mesh": _FakeMesh()})


def test_a_case_without_the_script_is_a_named_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "0").mkdir()
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match=r"no system/setAlphaField\.py"):
        _step().initializer({"_prev_mesh": _FakeMesh()})
