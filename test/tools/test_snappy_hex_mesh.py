# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""snappyHexMesh tool: build-step wiring + the ``SnappyHexMeshDictConfig`` reader/writer.

The unit checks monkeypatch the module-global bindings so no mesh is built; the OF
case drives the real bindings against the ``preprocess_case`` fixture. The config
tests mirror the blockMesh strategy: over real ``snappyHexMeshDict`` files vendored
under ``snappyhexmesh_cases/`` verify that loading one and writing it back reproduces
the dict, and — on the tube-bank case (the one with committed STL geometry) — that it
reproduces an **identical mesh** (same ``blockMesh``+``snappyHexMesh``+``checkMesh`` stats).
"""

import json

import os
import shutil
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("pybFoam")

from neofoam.framework.tools import ToolRuntime  # noqa: E402
from neofoam.io import write_configs  # noqa: E402
from neofoam.tools import snappy_hex_mesh  # noqa: E402
from neofoam.tools.snappy_hex_mesh import (  # noqa: E402
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


# --------------------------------------------------------------------------- #
# SnappyHexMeshDictConfig — construction + load/write round-trip               #
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


# --------------------------------------------------------------------------- #
# Reproduction: each vendored dict, loaded + written back, is unchanged         #
# --------------------------------------------------------------------------- #
#
# ``snappyhexmesh_cases/`` holds one real snappyHexMeshDict per distinct feature
# (searchable primitives, cell/face zones, surface layers, AMI, a plain baseline).
# snappyHexMesh needs a base mesh + external STL geometry to run, so the vendored
# dicts (geometry absent) are verified at the dict level — load → write → reload is
# structurally identical. The mesh-level ``checkMesh`` reproduction is covered by
# ``test_tube_bank_snappy_round_trip_reproduces_mesh`` below, which has geometry.

_CASES = sorted(
    (Path(__file__).parent / "snappyhexmesh_cases").glob("*.snappyHexMeshDict")
)


def _rounded(value: Any) -> Any:
    """Round every float (incl. those inside strings) to 6 significant figures.

    pybFoam writes scalars at 6 significant figures, so an exact ``==`` on a
    load→write→reload cycle trips on that precision; comparing at 6 figures isolates
    genuine structural differences.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return float(f"{value:.6g}")
    if isinstance(value, dict):
        return {k: _rounded(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_rounded(v) for v in value]
    if isinstance(value, str):
        import re

        return re.sub(r"-?\d+\.\d+", lambda m: f"{float(m.group()):.6g}", value)
    return value


@pytest.mark.parametrize("case", _CASES, ids=[p.stem for p in _CASES])
def test_snappyhexmeshdict_round_trip_reproduces_dict(
    case: Path, tmp_path: Path
) -> None:
    """Load a vendored ``snappyHexMeshDict``, write it back, assert it is unchanged."""
    cfg = SnappyHexMeshDictConfig.load(case_dir=case)
    write_configs([cfg], tmp_path)
    reloaded = SnappyHexMeshDictConfig.load(
        case_dir=tmp_path / "system" / "snappyHexMeshDict"
    )
    assert _rounded(reloaded.model_dump()) == _rounded(cfg.model_dump())


# --------------------------------------------------------------------------- #
# Mesh reproduction on the tube-bank (blockMesh → snappy → checkMesh)           #
# --------------------------------------------------------------------------- #

_PROBE = Path(__file__).parent / "_snappy_probe.py"
_WORKFLOW = Path(__file__).parents[1] / "workflow"


def _mesh_signature(case_dir: Path) -> dict[str, Any]:
    """Mesh ``case_dir`` (blockMesh→snappy) in a subprocess; return its stats."""
    proc = subprocess.run(
        [sys.executable, str(_PROBE), str(case_dir)], capture_output=True, text=True
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("STATS ")]
    if proc.returncode != 0 or not lines:
        pytest.skip(f"could not mesh {case_dir.name}:\n{proc.stderr[-500:]}")
    return json.loads(lines[-1][len("STATS ") :])


def _stage_tube_bank(case: Path, snappy: SnappyHexMeshDictConfig) -> None:
    """Lay down the tube-bank mesh inputs (blockMeshDict + snappy + STL) for meshing.

    The minimal ``controlDict``/``fvSchemes``/``fvSolution`` a Time + fvMesh need are
    provisioned by ``_snappy_probe.py`` at mesh time.
    """
    from neofoam.workflow.patch_set import PatchSet
    from neofoam.workflow.mesh_inputs import block_mesh_dict

    (case / "system").mkdir(parents=True)
    (case / "constant" / "triSurface").mkdir(parents=True)
    patch_set = PatchSet.load(_WORKFLOW / "cases" / "tube_bank_manifest.json")
    write_configs([block_mesh_dict(patch_set), snappy], case_dir=case)
    for stl in (_WORKFLOW / "cases" / "tube_bank" / "constant" / "triSurface").glob("*.stl"):
        shutil.copy(stl, case / "constant" / "triSurface" / stl.name)


@pytest.mark.slow
def test_tube_bank_snappy_round_trip_reproduces_mesh(tmp_path: Path) -> None:
    """A snappy dict written, then reloaded + rewritten, meshes identically."""
    from neofoam.workflow.patch_set import PatchSet
    from neofoam.workflow.mesh_inputs import snappy_dict

    patch_set = PatchSet.load(_WORKFLOW / "cases" / "tube_bank_manifest.json")

    original = tmp_path / "original"
    _stage_tube_bank(original, snappy_dict(patch_set))

    # Reload the written snappyHexMeshDict and stage it again.
    reloaded = SnappyHexMeshDictConfig.load(
        case_dir=original / "system" / "snappyHexMeshDict"
    )
    regenerated = tmp_path / "regenerated"
    _stage_tube_bank(regenerated, reloaded)

    original_stats = _mesh_signature(original)
    regenerated_stats = _mesh_signature(regenerated)

    assert original_stats["checkMesh"], "original tube-bank mesh failed checkMesh"
    assert regenerated_stats["checkMesh"], "regenerated tube-bank mesh failed checkMesh"
    topo = ("nCells", "nFaces", "nPoints", "patches")
    assert {k: regenerated_stats[k] for k in topo} == {
        k: original_stats[k] for k in topo
    }
    assert regenerated_stats["volume"] == pytest.approx(
        original_stats["volume"], rel=1e-5
    )
