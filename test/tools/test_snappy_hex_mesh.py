# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""snappyHexMesh tool: build-step wiring (unit, faked bindings) + refine (OF).

The unit checks monkeypatch the module-global bindings so no mesh is built; the
OF case drives the real bindings against the ``preprocess_case`` fixture.
"""

import os
import shutil
import types
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("pybFoam")

from neofoam.framework.tools import ToolRuntime  # noqa: E402
from neofoam.tools import snappy_hex_mesh  # noqa: E402
from neofoam.tools.snappy_hex_mesh import (  # noqa: E402
    SnappyHexMeshStep,
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


def test_snappy_refines_prior_mesh(tmp_path: Path) -> None:
    import pybFoam as pyf
    from pybFoam.meshing import generate_blockmesh, generate_snappy_hex_mesh

    assert not (CASE / "constant" / "polyMesh").exists()
    case = tmp_path / "case"
    shutil.copytree(CASE, case)
    cwd = Path.cwd()
    os.chdir(case)
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
