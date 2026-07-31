# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Default-mesh replacement wiring (pure-Python — built steps inspected only).

Mirrors the BUILD-stage wiring of ``create_fields``: a default disk-read
``mesh`` step plus the detected pipeline steps. When the pipeline is non-empty
its terminal alias (``replaces=["mesh"]``) supersedes the default; an empty
pipeline leaves the default untouched. No mesh is executed.
"""

from pathlib import Path

import pytest

from neofoam.framework.initialization import InitializerBuilder, lazy
from neofoam.framework.solver import configurations
from neofoam.framework.tools import tool_graph_steps
from neofoam.solver.incompressibleFluid import incompressibleFluid
from neofoam.solver.incompressibleFluid.create_fields import create_init
from neofoam.tools.run import detect_tools

CASE = Path(__file__).parent / "preprocess_case"
# A full case with NO ``system/preprocess.yaml`` (the opt-in/OCP boundary): the
# default disk-read mesh must survive the real create_fields wiring untouched.
DISK_MESH_CASE = Path(__file__).parent / "val_pitzDaily"


def _wire(case_dir: Path) -> list:
    builder = InitializerBuilder()
    builder.add(lazy("_foam_time", lambda _ctx: "time"))
    builder.add(lazy("mesh", lambda _ctx: "disk", depends_on=["_foam_time"]))
    builder.extend(tool_graph_steps(detect_tools(case_dir)))
    return builder.build()


def test_pipeline_replaces_default_mesh() -> None:
    steps = _wire(CASE)
    mesh_steps = [s for s in steps if s.name == "mesh"]
    assert len(mesh_steps) == 1
    # the surviving mesh step is the pipeline alias, not the disk-read default
    assert mesh_steps[0].replaces == ["mesh"]
    assert any(s.name.startswith("preprocess.") for s in steps)


def test_absent_enable_file_keeps_default_disk_mesh_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OCP / opt-in guard: no ``preprocess.yaml`` → default disk-read mesh runs.

    Drives the *real* create_fields wiring (not a hand-built stand-in): detection
    yields no preprocess runtimes and the built init-step list still carries the
    original ``lazy("mesh", create_mesh, depends_on=["_foam_time",
    "_foam_arglist"])`` — not a pipeline alias — so an on-disk mesh case is read
    unchanged. No lazy step is executed, so no mesh/pybFoam object is created.
    """
    # detect_and_create reads dictionaries by relative path, so run from the case.
    monkeypatch.chdir(DISK_MESH_CASE)

    assert detect_tools(DISK_MESH_CASE) == []

    runner = create_init(case_dir=DISK_MESH_CASE)
    runner.run_load()
    steps = runner.run_build()

    mesh_steps = [s for s in steps if s.name == "mesh"]
    assert len(mesh_steps) == 1
    # the surviving mesh step is the disk-read default, not a pipeline alias
    assert mesh_steps[0].replaces == []
    # ``_foam_arglist`` is the argList ``dynamicFvMesh::New`` selects through.
    assert mesh_steps[0].depends_on == ["_foam_time", "_foam_arglist"]
    assert not any(s.name.startswith("preprocess.") for s in steps)


def test_pipeline_replaces_default_mesh_through_create_init(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # detect/build read dicts by relative path, so run from the case directory.
    monkeypatch.chdir(CASE)
    runner = create_init(case_dir=CASE)
    runner.argv = ["incompressibleFluid"]
    runner.run_load()
    steps = runner.run_build()  # builds InitSteps; no lazy initializer is executed

    mesh_steps = [s for s in steps if s.name == "mesh"]
    assert len(mesh_steps) == 1
    assert mesh_steps[0].replaces == ["mesh"]  # surviving mesh is the pipeline alias
    assert any(s.name.startswith("preprocess.") for s in steps)


def test_configurations_includes_preprocess_config() -> None:
    assert "PreprocessConfig" in configurations(incompressibleFluid).names
