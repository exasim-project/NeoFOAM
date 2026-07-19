# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""blockMesh tool: build-step wiring + the ``BlockMeshDictConfig`` reader/writer.

The unit checks monkeypatch the module-global bindings so no mesh is built; the
in-process build drives the real bindings against the ``preprocess_case`` fixture
(which carries no ``constant/polyMesh`` on disk). The config tests cover the
structured ⇄ OpenFOAM round-trip that the case-authoring surface (wizard/MCP)
relies on to read and persist a ``blockMeshDict``.
"""

import os
import types
from pathlib import Path
from typing import Any

import pytest

from neofoam.tooling.casebuild import from_template
from neofoam.framework.tools import ToolRuntime
from neofoam.io import write_configs
from neofoam.tools import block_mesh
from neofoam.tools.block_mesh import (
    Block,
    BlockMeshDictConfig,
    BlockMeshStep,
    BlockPatch,
    blockMeshTool,
)
from neofoam.tools.run import run_preprocess

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


@pytest.mark.slow
def test_blockmesh_builds_in_process(tmp_path: Path) -> None:
    assert not (CASE / "constant" / "polyMesh").exists()
    case_dir = from_template(CASE).build_at(tmp_path / "case")
    cwd = Path.cwd()
    os.chdir(case_dir.path)
    try:
        ctx = run_preprocess(["preprocess"])
        assert ctx.mesh.nCells() > 0
    finally:
        os.chdir(cwd)


# --------------------------------------------------------------------------- #
# BlockMeshDictConfig — structured ⇄ OpenFOAM round-trip (hermetic)           #
# --------------------------------------------------------------------------- #


def _unit_box() -> BlockMeshDictConfig:
    """A minimal one-block config (a graded unit box with two named patches)."""
    return BlockMeshDictConfig(
        vertices=[
            (0, 0, 0),
            (1, 0, 0),
            (1, 1, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 0, 1),
            (1, 1, 1),
            (0, 1, 1),
        ],
        blocks=[Block(vertices=list(range(8)), cells=(4, 5, 6))],
        boundary=[
            BlockPatch(name="inlet", type="patch", faces=[(0, 4, 7, 3)]),
            BlockPatch(name="walls", type="wall", faces=[(0, 1, 5, 4), (2, 3, 7, 6)]),
        ],
    )


def test_config_serialises_to_openfoam_literals() -> None:
    """Under the ``openfoam`` format the compound sections become OF strings."""
    data = _unit_box().model_dump(context={"format": "openfoam"})
    assert data["vertices"].startswith("( (") and data["vertices"].endswith(") )")
    assert (
        data["blocks"]
        == "( hex ( 0 1 2 3 4 5 6 7 ) ( 4 5 6 ) simpleGrading ( 1 1 1 ) )"
    )
    assert "inlet { type patch ; faces ( ( 0 4 7 3 ) ) ; }" in data["boundary"]
    # A plain dump keeps the structured (form/JSON) shape.
    plain = _unit_box().model_dump()
    assert plain["vertices"][0] == (0, 0, 0)
    assert plain["blocks"][0]["cells"] == (4, 5, 6)


def test_config_write_read_round_trip(tmp_path: Path) -> None:
    """Writing then loading the file reconstructs an equal config."""
    cfg = _unit_box()
    write_configs([cfg], tmp_path)
    reloaded = BlockMeshDictConfig.load(case_dir=tmp_path / "system" / "blockMeshDict")
    assert reloaded == cfg
