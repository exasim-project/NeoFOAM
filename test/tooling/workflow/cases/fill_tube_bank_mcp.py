# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Autonomous MCP variant: describe the problem, the model fills the case via the MCP.

The model is handed a **problem description** (loaded by name from ``prompts.json``,
so more scenarios can be added without touching code) and the **neofoam MCP server**
as a toolset, and authors the case itself:

* ``case_patches`` — discover the boundary patches (name + role);
* ``model_catalog`` / ``list_configs`` / ``config_schema`` — discover the physics
  models + config shapes;
* ``save_case`` — write the configs to the case directory.

Geometry is a given: the mesh dicts, STLs, a ``system/preprocess.yaml`` and the
manifest are staged deterministically first (the model is told not to author the
mesh). A single ``scripts/Allrun`` then meshes (blockMesh → snappyHexMesh, in-process
from ``preprocess.yaml``) and solves.

Run ``python test/workflow/cases/fill_tube_bank_mcp.py <case_dir> --prompt heat_exchanger``
(needs an Anthropic API key + a sourced OpenFOAM), or ``--no-run`` to only author.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from typing import Any, Optional

from neofoam.tooling.workflow.mesh_inputs import block_mesh_dict, snappy_dict
from neofoam.tooling.workflow.patch_set import PatchSet
from neofoam.framework.tools import PreprocessConfig
from neofoam.io import write_configs

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "tube_bank_manifest.json"
TRI_SURFACE = (
    HERE / "tube_bank" / "constant" / "triSurface"
)  # test/workflow/cases/tube_bank/constant/triSurface
PROMPTS_FILE = HERE / "prompts.json"
# Mesh + solve run in one solver invocation via the minimal Allrun (in-process
# preprocess); see the module docstring.
ALLRUN = HERE.resolve().parents[3] / "scripts" / "Allrun"

DEFAULT_MODEL = "claude-sonnet-4-5"


def load_prompt(name: str) -> str:
    """Load a named problem description from ``prompts.json``."""
    prompts = json.loads(PROMPTS_FILE.read_text())
    if name not in prompts:
        raise KeyError(f"unknown prompt {name!r}; available: {sorted(prompts)}")
    return prompts[name]["prompt"]


def build_mcp_agent(model_name: str = DEFAULT_MODEL) -> Any:
    """A pydantic-ai agent wired to the neofoam MCP server (in-memory toolset)."""
    from pydantic_ai import Agent
    from pydantic_ai.mcp import MCPToolset

    from neofoam.mcp.server import mcp

    return Agent(f"anthropic:{model_name}", toolsets=[MCPToolset(mcp)])


def _stage_stls(case: Path, patch_set: PatchSet) -> None:
    """Copy the patch_set's declared STL surfaces into ``<case>/constant/triSurface``."""
    dst = case / "constant" / "triSurface"
    dst.mkdir(parents=True, exist_ok=True)
    for patch in patch_set.patches:
        name = Path(patch.stl).name
        src = TRI_SURFACE / name
        if not src.is_file():
            raise FileNotFoundError(f"geometry surface not found: {src}")
        shutil.copy2(src, dst / name)


def stage_geometry_and_manifest(case: Path, patch_set: PatchSet) -> None:
    """Lay down the deterministic geometry: STLs, mesh dicts, preprocess.yaml + manifest.

    ``preprocess.yaml`` wires the in-process mesh pipeline so a single ``scripts/Allrun``
    builds the mesh (blockMesh → snappyHexMesh → checkMesh) before solving.
    """
    _stage_stls(case, patch_set)
    write_configs([block_mesh_dict(patch_set), snappy_dict(patch_set)], case_dir=case)
    PreprocessConfig(
        tools=[
            {"tool": "blockMesh"},
            {"tool": "snappyHexMesh", "depends_on": ["blockMesh"]},
            {"tool": "checkMesh", "depends_on": ["snappyHexMesh"]},
        ]
    ).save(case_dir=case)
    shutil.copy2(ALLRUN, case / "Allrun")  # OpenFOAM-style: run ./Allrun in the case
    patch_set.save(case / "manifest.json")  # so the MCP `case_patches` tool can read it


def _findings_text(report: Any) -> str:
    """Render a ValidationReport's findings as a fix-list for the agent."""
    return "\n".join(
        f"- [{f.level}] {f.file}: {f.message}" + (f"  (fix: {f.fix})" if f.fix else "")
        for f in report.findings
    )


def build_tube_bank(
    case_dir: str | Path,
    *,
    prompt: str = "tube_bank",
    patch_set: Optional[PatchSet] = None,
    agent: Optional[Any] = None,
    max_fix_iterations: int = 3,
) -> Path:
    """Stage geometry, let the MCP agent author the physics, then drive a fix loop.

    ``prompt`` is a name in ``prompts.json`` (e.g. ``tube_bank`` or ``heat_exchanger``).
    After the agent authors the case, the driver runs ``validate_case`` itself and — if
    it is not ``ok`` — feeds the findings back and asks the agent to fix them, up to
    ``max_fix_iterations`` times (the conversation is kept, so the agent keeps its
    context). Returns the case directory; a single ``scripts/Allrun`` then meshes
    and solves.
    """
    from neofoam.mcp.registry import resolve_solver
    from neofoam.mcp.tools import validate_case

    case = Path(case_dir)
    patch_set = patch_set or PatchSet.load(MANIFEST)
    problem = load_prompt(prompt)
    solver = resolve_solver("incompressibleFluid")

    stage_geometry_and_manifest(case, patch_set)
    agent = agent or build_mcp_agent()

    async def _author() -> None:
        async with agent:  # open the MCP toolset for the run
            result = await agent.run(f"{problem}\n\nCase directory: {case}")
            for _ in range(max_fix_iterations):
                report = validate_case(solver, str(case))  # independent re-check
                if report.ok:
                    return
                result = await agent.run(
                    "`validate_case` still reports problems with the case you saved:\n"
                    f"{_findings_text(report)}\n\nApply each fix and call `save_case` "
                    "again for the affected configs.",
                    message_history=result.all_messages(),
                )

    asyncio.run(_author())
    return case


def main(argv: Optional[list[str]] = None) -> int:
    """Author the case via the MCP agent and (unless ``--no-run``) mesh + solve it."""
    import argparse
    import os
    import subprocess
    import sys

    parser = argparse.ArgumentParser(description="MCP-agent-fill + run a CFD case.")
    parser.add_argument("case_dir", help="Target case directory to author.")
    parser.add_argument(
        "--prompt",
        default="tube_bank",
        help="Prompt name from prompts.json (e.g. tube_bank, heat_exchanger).",
    )
    parser.add_argument(
        "--no-run", action="store_true", help="Only fill + save the case (for review)."
    )
    args = parser.parse_args(argv)

    case = build_tube_bank(args.case_dir, prompt=args.prompt)
    print(f"case authored via MCP ({args.prompt}): {case}")
    if args.no_run:
        return 0
    # Mesh + solve in one solver run via the case-local Allrun (OpenFOAM-style:
    # run ./Allrun from inside the case; in-process preprocess meshes then solves).
    return subprocess.run(
        ["./Allrun"], cwd=case, env={**os.environ, "NEOFOAM_PYTHON": sys.executable}
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
