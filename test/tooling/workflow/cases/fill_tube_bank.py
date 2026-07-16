# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Agent-driven driver: an LLM fills the case configs, we save them and verify it runs.

This is the point of the workflow: given the geometry patch_set, an agent fills
the solver's ``BaseConfig`` models (``configurations(incompressibleFluid)`` via
:func:`build_case_agent`), we ``write_configs`` them to disk, and launch the solver
to prove the authored case actually meshes + solves.

Split of responsibility:

* **geometry is a given** — the two mesh dicts (``block_mesh_dict`` / ``snappy_dict``)
  are derived deterministically from the patch_set and the STLs are staged as-is;
  a ``system/preprocess.yaml`` wires the in-process mesh pipeline so a single solver
  run (``scripts/Allrun``) meshes (blockMesh → snappyHexMesh) then solves;
* **the agent decides the physics** — boundary conditions (from the patch_set patch
  roles), transport / turbulence, time controls, and the PIMPLE fvSchemes/fvSolution.

The ``agent`` is injectable: production uses a live LLM (``build_case_agent``); tests
pass a stub returning a recorded ``CaseSpec`` so the fill→save→run path is
deterministic without a network call.

Run ``python test/workflow/cases/fill_tube_bank.py <case_dir>`` to fill + solve (needs an
Anthropic API key + a sourced OpenFOAM), or ``--no-run`` to only author.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Optional

from neofoam.agent.case_fill import build_case_agent, case_spec_to_configs
from neofoam.tooling.workflow.patch_set import PatchSet
from neofoam.tooling.workflow.mesh_inputs import block_mesh_dict, snappy_dict
from neofoam.framework.tools import PreprocessConfig
from neofoam.io import BaseConfig, write_configs
from neofoam.solver.incompressibleFluid.incompressibleFluid import incompressibleFluid
from neofoam.tools.block_mesh import BlockMeshDictConfig
from neofoam.tools.snappy_hex_mesh import SnappyHexMeshDictConfig

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "tube_bank_manifest.json"
TRI_SURFACE = (
    HERE / "tube_bank" / "constant" / "triSurface"
)  # test/workflow/cases/tube_bank/constant/triSurface
# Mesh + solve are run by the minimal OpenFOAM-convention Allrun (just the solver;
# meshing happens in-process from preprocess.yaml).
ALLRUN = HERE.resolve().parents[3] / "scripts" / "Allrun"

# Mesh dicts are geometry, not physics — the agent must not author them.
_GEOMETRY_CONFIGS = (BlockMeshDictConfig, SnappyHexMeshDictConfig)


def case_prompt(patch_set: PatchSet) -> str:
    """The natural-language brief the agent fills the case from.

    Carries the geometry facts the agent cannot invent — the boundary patch names
    and their roles — plus the physics of a laminar tube-bank run.
    """
    patches = "\n".join(f"  - {p.name}: {p.role.value}" for p in patch_set.patches)
    return (
        "Author a LAMINAR incompressibleFluid (PIMPLE) tube-bank case.\n\n"
        "Fill 0/U and 0/p with a boundaryField entry for EVERY patch below:\n"
        f"{patches}\n\n"
        "Boundary conditions by role (this is a thin quasi-2D slab, so the 'empty'"
        " patch is realised as a symmetry patch):\n"
        "  inlet:  U type fixedValue, value uniform (1 0 0); p type zeroGradient\n"
        "  outlet: U type zeroGradient;                      p type fixedValue, value uniform 0\n"
        "  wall:   U type noSlip;                            p type zeroGradient\n"
        "  empty:  U type symmetry;                          p type symmetry\n"
        "internalField: U uniform (0 0 0); p uniform 0.\n\n"
        "Transport: transportModel Newtonian, nu 1.5e-5 (air). Turbulence:"
        " simulationType laminar.\n"
        "Control: application pimpleFoam; endTime 0.5; deltaT 0.002; writeControl"
        " timeStep; writeInterval 50.\n"
        "fvSchemes: ddt(U) Euler; div(phi,U) Gauss linearUpwind grad(U);"
        " div((nuEff*dev2(T(grad(U))))) Gauss linear; laplacian(nuEff,U) Gauss linear"
        " corrected; sensible defaults for the rest.\n"
        "fvSolution solvers (every solver entry needs a smoother):\n"
        "  p and pFinal: solver GAMG, smoother GaussSeidel, tolerance 1e-6"
        " (relTol 0.05 for p, 0 for pFinal)\n"
        "  U and UFinal: solver smoothSolver, smoother symGaussSeidel, tolerance 1e-8"
        " (relTol 0.1 for U, 0 for UFinal)\n"
        "  also give p_rgh / p_rghFinal the same settings as p / pFinal.\n"
        "PIMPLE: nCorrectors 2, nNonOrthogonalCorrectors 1.\n"
        "Do NOT author blockMeshDict or snappyHexMeshDict — the mesh is handled"
        " separately."
    )


def author_configs(
    patch_set: PatchSet, *, agent: Optional[Any] = None
) -> list[BaseConfig]:
    """Have the agent fill the physics configs (drops any geometry it filled)."""
    agent = agent or build_case_agent(solver=incompressibleFluid)
    result = agent.run_sync(case_prompt(patch_set))
    return [
        cfg
        for cfg in case_spec_to_configs(result.output)
        if not isinstance(cfg, _GEOMETRY_CONFIGS)
    ]


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


def _stage_allrun(case: Path) -> None:
    """Copy ``scripts/Allrun`` into the case dir (run there, OpenFOAM-style)."""
    shutil.copy2(ALLRUN, case / "Allrun")


def preprocess_chain() -> PreprocessConfig:
    """The in-process blockMesh → snappyHexMesh → checkMesh pipeline (system/preprocess.yaml).

    Staged so a single solver run (``scripts/Allrun``) builds the mesh from the
    dicts before solving — no separate mesh binaries to orchestrate.
    """
    return PreprocessConfig(
        tools=[
            {"tool": "blockMesh"},
            {"tool": "snappyHexMesh", "depends_on": ["blockMesh"]},
            {"tool": "checkMesh", "depends_on": ["snappyHexMesh"]},
        ]
    )


def build_tube_bank(
    case_dir: str | Path,
    *,
    patch_set: Optional[PatchSet] = None,
    agent: Optional[Any] = None,
) -> Path:
    """Fill a laminar tube-bank case (agent physics + deterministic geometry), on disk.

    Stages the STLs, has the agent author the physics configs, adds the
    patch_set-derived mesh dicts + the in-process preprocess pipeline, and copies in
    the ``Allrun`` script. Returns the case directory; running ``./Allrun`` there then
    meshes (from ``preprocess.yaml``) and solves in one solver invocation.
    """
    case = Path(case_dir)
    patch_set = patch_set or PatchSet.load(MANIFEST)

    _stage_stls(case, patch_set)
    write_configs(
        [
            block_mesh_dict(patch_set),
            snappy_dict(patch_set),
            *author_configs(patch_set, agent=agent),
        ],
        case_dir=case,
    )
    preprocess_chain().save(case_dir=case)
    _stage_allrun(case)
    return case


def main(argv: Optional[list[str]] = None) -> int:
    """Fill the case with a live agent and (unless ``--no-run``) mesh + solve it."""
    import argparse
    import os
    import subprocess
    import sys

    parser = argparse.ArgumentParser(description="Agent-fill + run the tube-bank case.")
    parser.add_argument("case_dir", help="Target case directory to author.")
    parser.add_argument(
        "--no-run", action="store_true", help="Only fill + save the case (for review)."
    )
    args = parser.parse_args(argv)

    case = build_tube_bank(args.case_dir)
    print(f"case authored by agent: {case}")
    if args.no_run:
        return 0

    # Mesh + solve in one solver run via the case-local Allrun (OpenFOAM-style:
    # run ./Allrun from inside the case; in-process preprocess meshes then solves).
    return subprocess.run(
        ["./Allrun"], cwd=case, env={**os.environ, "NEOFOAM_PYTHON": sys.executable}
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
