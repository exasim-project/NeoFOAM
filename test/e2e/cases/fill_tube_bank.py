# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Agent-driven driver: an LLM fills the case configs, we save them and verify it runs.

This is the point of the e2e workflow: given the geometry manifest, an agent fills
the solver's ``BaseConfig`` models (``configurations(incompressibleFluid)`` via
:func:`build_case_agent`), we ``write_configs`` them to disk, and launch the solver
to prove the authored case actually meshes + solves.

Split of responsibility:

* **geometry is a given** — the two mesh dicts (``block_mesh_dict`` / ``snappy_dict``)
  and the preprocessing chain are derived deterministically from the manifest, and
  the STLs are staged as-is;
* **the agent decides the physics** — boundary conditions (from the manifest patch
  roles), transport / turbulence, time controls, and the PIMPLE fvSchemes/fvSolution.

The ``agent`` is injectable: production uses a live LLM (``build_case_agent``); tests
pass a stub returning a recorded ``CaseSpec`` so the fill→save→run path is
deterministic without a network call.

Run ``python test/e2e/cases/fill_tube_bank.py <case_dir>`` to fill + solve (needs an
Anthropic API key + a sourced OpenFOAM), or ``--no-run`` to only author.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from neofoam.agent.case_fill import build_case_agent, case_spec_to_configs
from neofoam.e2e.config import require_configs
from neofoam.e2e.geometry import stage_geometry
from neofoam.e2e.manifest import PatchManifest
from neofoam.e2e.mesh_inputs import block_mesh_dict, snappy_dict
from neofoam.framework.tools import PreprocessConfig
from neofoam.io import BaseConfig, write_configs
from neofoam.solver.incompressibleFluid.incompressibleFluid import incompressibleFluid
from neofoam.tools.block_mesh import BlockMeshDictConfig
from neofoam.tools.snappy_hex_mesh import SnappyHexMeshDictConfig

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "tube_bank_manifest.json"
TRI_SURFACE = HERE.parent / "constant" / "triSurface"  # test/e2e/constant/triSurface

# Mesh dicts are geometry, not physics — the agent must not author them.
_GEOMETRY_CONFIGS = (BlockMeshDictConfig, SnappyHexMeshDictConfig)


def case_prompt(manifest: PatchManifest) -> str:
    """The natural-language brief the agent fills the case from.

    Carries the geometry facts the agent cannot invent — the boundary patch names
    and their roles — plus the physics of a laminar tube-bank run.
    """
    patches = "\n".join(f"  - {p.name}: {p.role.value}" for p in manifest.patches)
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
    manifest: PatchManifest, *, agent: Optional[Any] = None
) -> list[BaseConfig]:
    """Have the agent fill the physics configs (drops any geometry it filled)."""
    agent = agent or build_case_agent(solver=incompressibleFluid)
    result = agent.run_sync(case_prompt(manifest))
    return [
        cfg
        for cfg in case_spec_to_configs(result.output)
        if not isinstance(cfg, _GEOMETRY_CONFIGS)
    ]


def preprocess_chain() -> PreprocessConfig:
    """The blockMesh → snappyHexMesh → checkMesh pipeline (fills the existing config)."""
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
    manifest: Optional[PatchManifest] = None,
    agent: Optional[Any] = None,
) -> Path:
    """Fill a laminar tube-bank case (agent physics + deterministic geometry), on disk.

    Stages the geometry, has the agent author the physics configs, adds the
    manifest-derived mesh dicts + preprocessing chain, writes everything, and gates
    on :func:`require_configs`. Returns the case directory.
    """
    case = Path(case_dir)
    manifest = manifest or PatchManifest.load(MANIFEST)

    stage_geometry(TRI_SURFACE, case, manifest=manifest)
    write_configs(
        [
            block_mesh_dict(manifest),
            snappy_dict(manifest),
            *author_configs(manifest, agent=agent),
        ],
        case_dir=case,
    )
    preprocess_chain().save(case_dir=case)

    require_configs(case)
    return case


def main(argv: Optional[list[str]] = None) -> int:
    """Fill the case with a live agent and (unless ``--no-run``) mesh + solve it."""
    import argparse
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

    # Normal launch: neofoam.e2e.solve runs the preprocess DAG (mesh) then solves, in
    # an isolated process (in-process pybFoam SIGBUSes at GC teardown).
    return subprocess.run(
        [sys.executable, "-m", "neofoam.e2e.solve", str(case)]
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
