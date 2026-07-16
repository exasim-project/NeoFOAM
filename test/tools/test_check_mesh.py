# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""checkMesh tool: raise/pass/wrap/reraise (unit, faked binding) + valid mesh (OF).

The unit checks monkeypatch ``checkMesh`` so no mesh is built; the OF case drives the
real binding against the ``preprocess_case`` fixture.
"""

import os
from pathlib import Path
from typing import Any

import pytest

from neofoam.tooling.casebuild import from_template
from neofoam.framework.initialization import InitStepExecutionError
from neofoam.framework.tools import ToolRuntime
from neofoam.tools import check_mesh
from neofoam.tools.check_mesh import CheckMeshStep, checkMeshTool

CASE = Path(__file__).parents[1] / "solver" / "incompressibleFluid" / "preprocess_case"


def _check_step(**cfg_kwargs: Any) -> Any:
    cfg = CheckMeshStep(tool="checkMesh", **cfg_kwargs)
    runtime = ToolRuntime(spec=checkMeshTool, name="preprocess.checkMesh", config=cfg)
    return runtime.run_build()[0]


def test_checkmesh_raises_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        check_mesh,
        "checkMesh",
        lambda *a, **k: {"passed": False, "total_errors": 3},
    )
    step = _check_step(fail_on_error=True)
    with pytest.raises(InitStepExecutionError) as excinfo:
        step.initializer({"_prev_mesh": object()})
    assert excinfo.value.step_name == "preprocess.checkMesh"


def test_checkmesh_passes_returns_prior_mesh(monkeypatch: pytest.MonkeyPatch) -> None:
    stats = {"passed": True, "total_errors": 0}
    monkeypatch.setattr(check_mesh, "checkMesh", lambda *a, **k: stats)
    step = _check_step(fail_on_error=True)
    prior = object()
    # checkMesh only validates; it passes the prior mesh straight through.
    assert step.initializer({"_prev_mesh": prior}) is prior


def test_checkmesh_does_not_raise_when_fail_on_error_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stats = {"passed": False, "total_errors": 5}
    monkeypatch.setattr(check_mesh, "checkMesh", lambda *a, **k: stats)
    step = _check_step(fail_on_error=False)
    prior = object()
    assert step.initializer({"_prev_mesh": prior}) is prior


def test_checkmesh_wraps_binding_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    boom = RuntimeError("binding boom")

    def raise_runtime(*a: Any, **k: Any) -> Any:
        raise boom

    monkeypatch.setattr(check_mesh, "checkMesh", raise_runtime)
    step = _check_step(fail_on_error=True)
    with pytest.raises(InitStepExecutionError) as excinfo:
        step.initializer({"_prev_mesh": object()})
    assert excinfo.value.step_name == "preprocess.checkMesh"
    assert excinfo.value.__cause__ is boom


@pytest.mark.parametrize("exc_type", [ValueError, TypeError])
def test_checkmesh_reraises_value_or_type_error(
    monkeypatch: pytest.MonkeyPatch, exc_type: type[Exception]
) -> None:
    boom = exc_type("bad checkMesh argument")

    def raise_it(*a: Any, **k: Any) -> Any:
        raise boom

    monkeypatch.setattr(check_mesh, "checkMesh", raise_it)
    step = _check_step(fail_on_error=True)
    with pytest.raises(exc_type) as excinfo:
        step.initializer({"_prev_mesh": object()})
    assert excinfo.value is boom


def test_checkmesh_propagates_check_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def fake_check(m: Any, **kwargs: Any) -> dict[str, Any]:
        seen.update(kwargs)
        return {"passed": True, "total_errors": 0}

    monkeypatch.setattr(check_mesh, "checkMesh", fake_check)
    step = _check_step(all_topology=True, all_geometry=True, check_quality=True)
    step.initializer({"_prev_mesh": object()})
    assert seen == {
        "all_topology": True,
        "all_geometry": True,
        "check_quality": True,
    }


def test_checkmesh_passes_on_valid_mesh(tmp_path: Path) -> None:
    import pybFoam as pyf
    from pybFoam.meshing import checkMesh as real_check
    from pybFoam.meshing import generate_blockmesh

    assert not (CASE / "constant" / "polyMesh").exists()
    case_dir = from_template(CASE).build_at(tmp_path / "case")
    cwd = Path.cwd()
    os.chdir(case_dir.path)
    try:
        time = pyf.Time(pyf.argList(["preprocess"]))
        m = generate_blockmesh(time, pyf.dictionary.read("system/blockMeshDict"))
        stats = real_check(m)
        assert stats["passed"] is True
    finally:
        os.chdir(cwd)
