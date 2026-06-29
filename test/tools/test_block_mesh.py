# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""blockMesh tool: build-step wiring (unit, faked bindings) + in-process build (OF).

The unit checks monkeypatch the module-global bindings so no mesh is built; the
in-process build drives the real bindings against the ``preprocess_case`` fixture
(which carries no ``constant/polyMesh`` on disk).
"""

import os
import shutil
import types
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("pybFoam")

from neofoam.framework.tools import ToolRuntime  # noqa: E402
from neofoam.tools import block_mesh  # noqa: E402
from neofoam.tools.block_mesh import BlockMeshStep, blockMeshTool  # noqa: E402
from neofoam.tools.run import run_preprocess  # noqa: E402

CASE = Path(__file__).parents[1] / "solver" / "incompressibleFluid" / "preprocess_case"


def test_block_step_depends_on_foam_time() -> None:
    rt = ToolRuntime(
        spec=blockMeshTool,
        name="preprocess.blockMesh",
        config=BlockMeshStep(tool="blockMesh"),
    )
    step = rt.run_build()[0]
    assert step.name == "preprocess.blockMesh"
    assert step.depends_on == ["_foam_time"]


def test_block_propagates_verbose(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}
    monkeypatch.setattr(
        block_mesh,
        "pyf",
        types.SimpleNamespace(dictionary=types.SimpleNamespace(read=lambda f: f)),
    )

    def fake_block(time: Any, d: Any, verbose: bool = False) -> Any:
        seen["call"] = (d, verbose)
        return object()

    monkeypatch.setattr(block_mesh, "generate_blockmesh", fake_block)
    rt = ToolRuntime(
        spec=blockMeshTool,
        name="preprocess.blockMesh",
        config=BlockMeshStep(tool="blockMesh", verbose=True),
    )
    rt.run_build()[0].initializer({"_foam_time": "T"})
    assert seen["call"] == ("system/blockMeshDict", True)


def test_blockmesh_builds_in_process(tmp_path: Path) -> None:
    assert not (CASE / "constant" / "polyMesh").exists()
    case = tmp_path / "case"
    shutil.copytree(CASE, case)
    cwd = Path.cwd()
    os.chdir(case)
    try:
        ctx = run_preprocess(["preprocess"])
        assert ctx.mesh.nCells() > 0
    finally:
        os.chdir(cwd)
