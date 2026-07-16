# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The MCP-agent driver: geometry is staged, the model fills the physics via the MCP.

The fast test checks the deterministic staging (mesh dicts + manifest) and that the
MCP ``case_patches`` tool can read the staged manifest — the geometry-discovery seam
the agent relies on. The slow test drives the live MCP agent (gated on an API key +
OpenFOAM) and asserts the authored case meshes + solves.
"""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

from neofoam.tooling.workflow.patch_set import PatchSet  # noqa: E402
from neofoam.mcp.tools import case_patches  # noqa: E402
from neofoam.solver.incompressibleFluid.configs import ControlDictConfig  # noqa: E402

_DRIVER_PATH = Path(__file__).parent / "cases" / "fill_tube_bank_mcp.py"
_spec = importlib.util.spec_from_file_location("fill_tube_bank_mcp", _DRIVER_PATH)
assert _spec and _spec.loader
fill_tube_bank_mcp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fill_tube_bank_mcp)


def test_staging_lays_down_geometry_and_manifest(tmp_path: Path) -> None:
    """Geometry (mesh dicts + STLs), preprocess.yaml and the manifest are staged."""
    patch_set = PatchSet.load(fill_tube_bank_mcp.MANIFEST)
    fill_tube_bank_mcp.stage_geometry_and_manifest(tmp_path, patch_set)
    for rel in (
        "system/blockMeshDict",
        "system/snappyHexMeshDict",
        "system/preprocess.yaml",
        "Allrun",
        "manifest.json",
        "constant/triSurface/tubes.stl",
    ):
        assert (tmp_path / rel).is_file(), f"missing {rel}"


def test_mcp_case_patches_sees_the_staged_geometry(tmp_path: Path) -> None:
    """The agent's geometry-discovery tool reads the staged manifest."""
    patch_set = PatchSet.load(fill_tube_bank_mcp.MANIFEST)
    fill_tube_bank_mcp.stage_geometry_and_manifest(tmp_path, patch_set)
    assert {p.name for p in case_patches(str(tmp_path))} == {
        "inlet",
        "outlet",
        "walls",
        "tubes",
        "frontBack",
    }


@pytest.mark.slow
@pytest.mark.skipif(
    not (os.environ.get("ANTHROPIC_API_KEY") and os.environ.get("WM_PROJECT_DIR")),
    reason="needs ANTHROPIC_API_KEY + a sourced OpenFOAM",
)
def test_live_mcp_agent_fills_and_case_solves(tmp_path: Path) -> None:
    """The live MCP agent authors the case via the tools; it then meshes + solves."""
    case = fill_tube_bank_mcp.build_tube_bank(tmp_path)  # live LLM + MCP toolset
    # Shorten the run so the smoke test is quick (write every step, stop after a few).
    ControlDictConfig(
        application="pimpleFoam",
        endTime=0.006,
        deltaT=0.002,
        writeControl="timeStep",
        writeInterval=1,
    ).save(case_dir=case)

    proc = subprocess.run(
        ["./Allrun"],
        cwd=case,
        capture_output=True,
        text=True,
        env={**os.environ, "NEOFOAM_PYTHON": sys.executable},
    )
    time_dirs = [
        child.name
        for child in case.iterdir()
        if child.is_dir()
        and child.name.replace(".", "", 1).isdigit()
        and float(child.name) > 0
    ]
    assert time_dirs, (
        f"no time directory written\nSTDOUT:\n{proc.stdout[-1000:]}\n"
        f"STDERR:\n{proc.stderr[-1000:]}"
    )
