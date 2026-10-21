# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""setFields tool: registration, catalog, the restart guard, and the real pipeline.

The unit checks fake the two ``neofoam.preprocess`` entry points and hand the step a
mesh that is nothing but a ``time()``, so no case is read and no mesh is built; the
end-to-end checks run the real ``neofoam preprocess`` pipeline (``blockMesh`` →
``setFields``) over ``test/preprocess/cases/box`` — a unit box cut into 4x4x1 cells
whose ``0/alpha.water`` is deliberately a wrong uniform 0.5, so the assertion fails
unless the tool wrote the field. Cell centres sit at x = 0.125/0.375/0.625/0.875, so
the case's declared region ``x <= 0.5`` is the two left-hand columns — the pre-written
``_IN_REGION``.

Both front doors are exercised: the case as checked in declares YAML only; the
script variant overlays ``cases/set_fields_script/system/setFields.py``, whose
region is applied *before* the YAML's, so the right-hand half keeps the script's
0.25 while the left-hand half takes the YAML's 1.0.

**The restart guard** is what stops the same ``system/preprocess.yaml`` from
re-initialising a continued run, and it is pinned twice: as a unit (a case holding an
older time directory than the one the run starts from is a restart) and end to end,
by overlaying ``cases/set_fields_restart/0.1`` — fields at a deliberately distinct
uniform 0.75 — and starting the run there. The values written at 0.1 must come back
untouched.

The case is copied into ``tmp_path`` first — preprocessing writes ``0/`` and
``constant/polyMesh`` — and the written field is read back through
``CaseDir.read_field``, i.e. in a fresh process: the ``Foam::Time``
``run_preprocess`` built is gone by the time it returns, so reading through the
mesh it published would touch freed memory.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pytest
from numpy.testing import assert_allclose

from neofoam.framework.initialization import InitStepExecutionError
from neofoam.framework.tools import ToolRuntime
from neofoam.io.schema import tool_catalog
from neofoam.solver.incompressibleFluid import incompressibleFluid
from neofoam.tooling.casebuild import CaseDir, patch
from neofoam.tools import available_tools, set_fields
from neofoam.tools.run import run_preprocess
from neofoam.tools.set_fields import SetFieldsStep, setFieldsTool

BOX = Path(__file__).parents[1] / "preprocess" / "cases" / "box"
SCRIPT_OVERLAY = Path(__file__).parent / "cases" / "set_fields_script"
RESTART_OVERLAY = Path(__file__).parent / "cases" / "set_fields_restart"

#: ``test/preprocess/cases/box``: the eight cells the declared region selects.
_IN_REGION = np.array([True, True, False, False] * 4)

#: ``cases/set_fields_restart/0.1``: what the earlier run left behind.
_RESTART_ALPHA = 0.75

#: The time the restart case is continued from.
_RESTART_TIME = "0.1"


class _Declaration:
    """The two attributes the tool reads off a SetFields."""

    def __init__(self) -> None:
        self.defaults: dict[str, Any] = {"alpha.water": 0.0}
        self.regions: list[Any] = []


class _FakeTime:
    """The two accessors the restart guard calls on the mesh's Time.

    ``value`` and ``name`` are separate because OpenFOAM's are: the name is the
    start time rounded to ``timePrecision``, and it is the name that is on disk.
    """

    def __init__(self, value: float, name: Optional[str] = None) -> None:
        self._value = value
        self._name = str(value) if name is None else name

    def value(self) -> float:
        return self._value

    def timeName(self) -> str:
        return self._name


class _FakeMesh:
    """A mesh that is nothing but its time — the step touches nothing else."""

    def __init__(self, time_value: float = 0.0, time_name: Optional[str] = None) -> None:
        self._time = _FakeTime(time_value, time_name)

    def time(self) -> _FakeTime:
        return self._time


def _step() -> Any:
    runtime = ToolRuntime(
        spec=setFieldsTool, name="preprocess.setFields", config=SetFieldsStep(tool="setFields")
    )
    return runtime.run_build()[0]


def _fake_case(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Fake the case read and the write-back; return the recorded apply calls."""
    calls: list[Any] = []
    monkeypatch.setattr(set_fields, "set_fields_for_case", lambda case_dir: _Declaration())
    monkeypatch.setattr(set_fields, "apply_set_fields", lambda *args: calls.append(args))
    return calls


def _time_dirs(case_dir: Path, *names: str) -> Path:
    """A case directory holding only the named (empty) time directories."""
    for name in names:
        (case_dir / name).mkdir()
    return case_dir


def test_set_fields_is_registered() -> None:
    by_name = {tool.name: tool for tool in available_tools()}
    assert by_name["setFields"] is setFieldsTool


def test_the_step_schema_reaches_the_tool_catalog() -> None:
    by_name = {info.name: info for info in tool_catalog(incompressibleFluid)}

    assert by_name["setFields"].step_schema["properties"]["tool"]["const"] == "setFields"


def test_the_step_takes_no_option_but_the_tool_name() -> None:
    assert set(SetFieldsStep.model_fields) == {"tool"}


def test_setfields_passes_the_prior_mesh_through(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_case(monkeypatch)
    monkeypatch.chdir(_time_dirs(tmp_path, "0"))
    prior = _FakeMesh()

    # setFields writes fields, not topology; the pipeline treats it as a
    # mesh-advancing step so it can be the sink.
    assert _step().initializer({"_prev_mesh": prior}) is prior


def test_setfields_applies_at_the_cases_earliest_time(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _fake_case(monkeypatch)
    monkeypatch.chdir(_time_dirs(tmp_path, "0"))
    prior = _FakeMesh(time_value=0.0)

    _step().initializer({"_prev_mesh": prior})

    assert calls == [(prior, {"alpha.water": 0.0}, [])]


def test_setfields_skips_a_run_starting_after_an_existing_time(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _fake_case(monkeypatch)
    monkeypatch.chdir(_time_dirs(tmp_path, "0", "0.1"))
    prior = _FakeMesh(time_value=0.1)

    # A restart: 0/ is older than the 0.1/ the run starts from.
    assert _step().initializer({"_prev_mesh": prior}) is prior
    assert calls == []


def test_setfields_applies_when_the_only_time_directory_is_a_late_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _fake_case(monkeypatch)
    monkeypatch.chdir(_time_dirs(tmp_path, "5"))
    prior = _FakeMesh(time_value=5.0)

    # Not a restart: nothing older exists, so the case starts here.
    _step().initializer({"_prev_mesh": prior})

    assert calls == [(prior, {"alpha.water": 0.0}, [])]


def test_setfields_applies_when_the_start_directory_is_the_earliest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _fake_case(monkeypatch)
    monkeypatch.chdir(_time_dirs(tmp_path, "0.1"))
    # `startTime 0.10000001` at `timePrecision 6` *is* the 0.1/ directory the run
    # starts from, not a time after it: comparing the value against the name
    # would read this as a restart.
    prior = _FakeMesh(time_value=0.10000001, time_name="0.1")

    _step().initializer({"_prev_mesh": prior})

    assert calls == [(prior, {"alpha.water": 0.0}, [])]


@pytest.mark.parametrize("sibling", ["0.orig", "constant", "processor0"])
def test_setfields_ignores_a_directory_that_is_not_a_time(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, sibling: str
) -> None:
    calls = _fake_case(monkeypatch)
    monkeypatch.chdir(_time_dirs(tmp_path, "5", sibling))
    prior = _FakeMesh(time_value=5.0, time_name="5")

    _step().initializer({"_prev_mesh": prior})

    assert calls == [(prior, {"alpha.water": 0.0}, [])]


def test_setfields_wraps_binding_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _fake_case(monkeypatch)
    monkeypatch.chdir(_time_dirs(tmp_path, "0"))
    boom = RuntimeError("binding boom")

    def raise_runtime(*args: Any) -> Any:
        raise boom

    monkeypatch.setattr(set_fields, "apply_set_fields", raise_runtime)

    with pytest.raises(InitStepExecutionError) as excinfo:
        _step().initializer({"_prev_mesh": _FakeMesh()})
    assert excinfo.value.step_name == "preprocess.setFields"
    assert excinfo.value.__cause__ is boom


@pytest.mark.parametrize(
    ("script", "outside_the_region"),
    [(False, 0.0), (True, 0.25)],
    ids=["yaml_only", "script_and_yaml"],
)
def test_preprocess_writes_the_declared_region_into_the_field(
    tmp_path: Path, script: bool, outside_the_region: float
) -> None:
    case_dir = tmp_path / "box"
    shutil.copytree(BOX, case_dir)
    if script:
        shutil.copy(SCRIPT_OVERLAY / "system/setFields.py", case_dir / "system/setFields.py")

    run_preprocess(["preprocess", "-case", str(case_dir)])

    assert_allclose(
        CaseDir(case_dir).read_field("alpha.water", time="0"),
        np.where(_IN_REGION, 1.0, outside_the_region),
        err_msg="cases/box: alpha.water after neofoam preprocess",
    )


def test_preprocess_leaves_a_restarted_runs_fields_alone(tmp_path: Path) -> None:
    case = CaseDir(tmp_path / "box")
    shutil.copytree(BOX, case.path)
    shutil.copytree(RESTART_OVERLAY / _RESTART_TIME, case.path / _RESTART_TIME)
    patch("system/controlDict", startTime=float(_RESTART_TIME))(case)

    run_preprocess(["preprocess", "-case", str(case.path)])

    # read_field stages the requested time directory as 0/ of a temporary case, so
    # the start time has to point back at it before the fields can be read back.
    patch("system/controlDict", startTime=0.0)(case)
    assert_allclose(
        case.read_field("alpha.water", time=_RESTART_TIME),
        np.full(16, _RESTART_ALPHA),
        err_msg=f"cases/box restarted at {_RESTART_TIME}: alpha.water must not be re-initialised",
    )


def test_preprocess_can_be_rerun_on_a_binary_time_directory(tmp_path: Path) -> None:
    case = CaseDir(tmp_path / "box")
    shutil.copytree(BOX, case.path)
    patch("system/controlDict", writeFormat="binary")(case)

    run_preprocess(["preprocess", "-case", str(case.path)])
    # The second run reads back what the first wrote — a binary field file, whose
    # FoamFile header the class pre-check has to read without OpenFOAM.
    run_preprocess(["preprocess", "-case", str(case.path)])

    assert_allclose(
        case.read_field("alpha.water", time="0"),
        np.where(_IN_REGION, 1.0, 0.0),
        err_msg="cases/box with writeFormat binary: alpha.water after a second preprocess",
    )
