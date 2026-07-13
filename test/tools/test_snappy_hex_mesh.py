# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""snappyHexMesh tool: build-step wiring + the ``SnappyHexMeshDictConfig`` reader/writer.

The unit checks monkeypatch the module-global bindings so no mesh is built; the OF
case drives the real bindings against the ``preprocess_case`` fixture. The config
test covers the ``castellate_and_snap`` constructor that the case-authoring surface
(wizard/MCP) uses to assemble geometry + refinementSurfaces.
"""

import os
import types
from pathlib import Path
from typing import Any

import pytest

from neofoam.tooling.casebuild import from_template
from neofoam.framework.tools import ToolRuntime
from neofoam.tools import snappy_hex_mesh
from neofoam.tools.snappy_hex_mesh import (
    SnappyHexMeshDictConfig,
    SnappyHexMeshStep,
    SnappySurface,
    snappyHexMeshTool,
)

CASE = Path(__file__).parents[1] / "solver" / "incompressibleFluid" / "preprocess_case"


def test_snappy_reads_prev_mesh_and_returns_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}
    monkeypatch.setattr(
        snappy_hex_mesh,
        "pyf",
        types.SimpleNamespace(dictionary=types.SimpleNamespace(read=lambda f: f)),
    )

    def fake_snappy(
        m: Any, d: Any, overwrite: bool = True, verbose: bool = True
    ) -> None:
        seen["call"] = (m, d, overwrite, verbose)

    monkeypatch.setattr(snappy_hex_mesh, "generate_snappy_hex_mesh", fake_snappy)
    prior = object()
    rt = ToolRuntime(
        spec=snappyHexMeshTool,
        name="preprocess.snappyHexMesh",
        config=SnappyHexMeshStep(tool="snappyHexMesh", overwrite=False, verbose=True),
    )
    result = rt.run_build()[0].initializer({"_prev_mesh": prior})
    # snappyHexMesh mutates the prior mesh in place and threads it onward.
    assert result is prior
    assert seen["call"][0] is prior
    assert seen["call"][1] == "system/snappyHexMeshDict"
    assert seen["call"][2] is False  # overwrite propagated
    assert seen["call"][3] is True  # verbose propagated


@pytest.mark.slow
def test_snappy_refines_prior_mesh(tmp_path: Path) -> None:
    import pybFoam as pyf
    from pybFoam.meshing import generate_blockmesh, generate_snappy_hex_mesh

    assert not (CASE / "constant" / "polyMesh").exists()
    case_dir = from_template(CASE).build_at(tmp_path / "case")
    cwd = Path.cwd()
    os.chdir(case_dir.path)
    try:
        time = pyf.Time(pyf.argList(["preprocess"]))
        block_mesh = generate_blockmesh(
            time, pyf.dictionary.read("system/blockMeshDict")
        )
        block_cells = block_mesh.nCells()

        generate_snappy_hex_mesh(
            block_mesh,
            pyf.dictionary.read("system/snappyHexMeshDict"),
            verbose=False,
        )
        assert block_mesh.nCells() != block_cells
    finally:
        os.chdir(cwd)


# --------------------------------------------------------------------------- #
# SnappyHexMeshDictConfig — construction                                        #
# --------------------------------------------------------------------------- #


def test_castellate_and_snap_builds_geometry_and_refinement() -> None:
    """The constructor assembles geometry + refinementSurfaces + locationInMesh."""
    cfg = SnappyHexMeshDictConfig.castellate_and_snap(
        surfaces=[SnappySurface(name="tubes", file="tubes.stl", level=(1, 2))],
        location_in_mesh=(0.08, 0.08, 0.01),
    )
    assert cfg.geometry == {"tubes": {"type": "triSurfaceMesh", "file": '"tubes.stl"'}}
    cmc = cfg.castellatedMeshControls
    assert cmc["refinementSurfaces"]["tubes"]["level"] == "(1 2)"
    assert cmc["locationInMesh"] == "(0.08 0.08 0.01)"
