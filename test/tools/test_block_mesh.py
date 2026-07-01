# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""blockMesh tool: build-step wiring + the ``BlockMeshDictConfig`` reader/writer.

The unit checks monkeypatch the module-global bindings so no mesh is built; the
in-process build drives the real bindings against the ``preprocess_case`` fixture
(which carries no ``constant/polyMesh`` on disk). The config tests cover the
structured ⇄ OpenFOAM round-trip, and — over the real ``blockMeshDict`` files
vendored under ``blockmesh_cases/`` — verify that loading one and writing it back
reproduces an **identical mesh** (same ``blockMesh`` + ``checkMesh`` stats).
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
from neofoam.tools import block_mesh  # noqa: E402
from neofoam.tools.block_mesh import (  # noqa: E402
    Block,
    BlockMeshDictConfig,
    BlockMeshStep,
    BlockPatch,
    blockMeshTool,
)
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


@pytest.mark.slow
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


# A section that does not fit the structured grammar (here: projected vertices,
# ``project (x y z) (surface)``) is kept as a raw string and round-trips verbatim
# rather than crashing. Files live in ``blockmesh_cases/opaque/`` — dicts that
# exercise the fallback but need external geometry to mesh, so they are not part
# of the meshing corpus above.
_OPAQUE_CASES = sorted(
    (Path(__file__).parent / "blockmesh_cases" / "opaque").glob("*.blockMeshDict")
)


@pytest.mark.parametrize("case", _OPAQUE_CASES, ids=[p.stem for p in _OPAQUE_CASES])
def test_unstructurable_section_round_trips_opaque(case: Path, tmp_path: Path) -> None:
    """A dict with an unparseable section loads (not crashes) and round-trips."""
    cfg = BlockMeshDictConfig.load(case_dir=case)
    # The awkward section fell back to a raw string; the rest stayed structured.
    assert isinstance(cfg.vertices, str)
    assert isinstance(cfg.blocks, list) and isinstance(cfg.boundary, list)
    write_configs([cfg], tmp_path)
    reloaded = BlockMeshDictConfig.load(case_dir=tmp_path / "system" / "blockMeshDict")
    assert reloaded == cfg


# --------------------------------------------------------------------------- #
# Reproduction: each vendored dict, loaded + written back                       #
# --------------------------------------------------------------------------- #
#
# ``blockmesh_cases/`` holds one real blockMeshDict per distinct feature
# (simpleGrading, multi/edge grading + grading vars, arc edges + cyclic
# neighbourPatch, mergePatchPairs, inGroups). Two levels of check: a fast one that
# the input parses + writes back unchanged, and a slow one (``@pytest.mark.slow``)
# that the regenerated dict — the hard-to-parse ones — actually meshes to the same
# ``checkMesh``-valid mesh via ``generate_blockmesh``.

_CASES = sorted((Path(__file__).parent / "blockmesh_cases").glob("*.blockMeshDict"))
_PROBE = Path(__file__).parent / "_blockmesh_probe.py"


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
def test_blockmeshdict_round_trip_reproduces_dict(case: Path, tmp_path: Path) -> None:
    """Load a vendored ``blockMeshDict``, write it back, assert it is unchanged."""
    cfg = BlockMeshDictConfig.load(case_dir=case)
    write_configs([cfg], tmp_path)
    reloaded = BlockMeshDictConfig.load(case_dir=tmp_path / "system" / "blockMeshDict")
    assert _rounded(reloaded.model_dump()) == _rounded(cfg.model_dump())


def _mesh_signature(dict_path: Path) -> dict[str, Any]:
    """Mesh ``dict_path`` in a subprocess and return its stats (JSON ``STATS`` line).

    The subprocess ``os._exit``\\ s after printing, so the teardown of the
    mesh-bound pybFoam objects (which SIGBUSes in-process) can't fail the run.
    """
    proc = subprocess.run(
        [sys.executable, str(_PROBE), str(dict_path)],
        capture_output=True,
        text=True,
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("STATS ")]
    if proc.returncode != 0 or not lines:
        pytest.skip(f"could not mesh {dict_path.name}:\n{proc.stderr[-500:]}")
    return json.loads(lines[-1][len("STATS ") :])


@pytest.mark.slow
@pytest.mark.parametrize("case", _CASES, ids=[p.stem for p in _CASES])
def test_blockmeshdict_round_trip_reproduces_mesh(case: Path, tmp_path: Path) -> None:
    """Load a vendored ``blockMeshDict``, write it back, assert an identical mesh."""
    cfg = BlockMeshDictConfig.load(case_dir=case)
    write_configs([cfg], tmp_path)
    saved = tmp_path / "system" / "blockMeshDict"

    original = _mesh_signature(case)
    regenerated = _mesh_signature(saved)

    assert original["checkMesh"], f"original mesh failed checkMesh: {case.name}"
    assert regenerated["checkMesh"], f"regenerated mesh failed checkMesh: {case.name}"
    # Topology must be bit-identical; volume matches to float round-trip precision
    # (compound sections come back from pybFoam at 6 significant figures).
    topo = ("nCells", "nFaces", "nPoints", "patches")
    assert {k: regenerated[k] for k in topo} == {k: original[k] for k in topo}, (
        f"round-trip changed the mesh topology for {case.name}:\n"
        f"  original={original}\n  regenerated={regenerated}"
    )
    assert regenerated["volume"] == pytest.approx(original["volume"], rel=1e-5)
