# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Hardcoded (no-LLM) driver: author a runnable laminar tube-bank case, config-driven.

This is the near-term stand-in for the agent: it fills the **real** ``BaseConfig``
models the solver owns (``configurations(incompressibleFluid)``) and writes them —
filling the models *is* writing the case files. Nothing renders OpenFOAM text by
hand.

Pipeline:

* geometry is a given input — :func:`stage_geometry` copies the checked-in STLs into
  ``constant/triSurface`` (no geometry is generated);
* the two mesh dicts come from the manifest (:func:`block_mesh_dict` /
  :func:`snappy_dict`);
* the preprocessing chain is the existing :class:`PreprocessConfig`
  (blockMesh → snappyHexMesh → checkMesh);
* the physics are hardcoded laminar values, with each patch's boundary condition
  chosen from its manifest role.

``write_configs`` persists the OpenFOAM-strategy configs (the YAML ``preprocess.yaml``
is saved via its own strategy); :func:`require_configs` gates completeness. Running is
a normal solver launch — ``python -m neofoam.e2e.solve <case>`` runs the preprocess DAG
(mesh) then solves, in an isolated process.

Run ``python test/e2e/cases/fill_tube_bank.py <case_dir>`` to author + solve, or
``--no-run`` to only author (for review / check-in).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from neofoam.e2e.config import require_configs
from neofoam.e2e.geometry import stage_geometry
from neofoam.e2e.manifest import PatchManifest, PatchRole
from neofoam.e2e.mesh_inputs import block_mesh_dict, snappy_dict
from neofoam.framework.solver.configurations import configurations
from neofoam.framework.tools import PreprocessConfig
from neofoam.io import BaseConfig, write_configs
from neofoam.solver.incompressibleFluid.configs import ControlDictConfig
from neofoam.solver.incompressibleFluid.incompressibleFluid import incompressibleFluid
from neofoam.turbulence.config import TurbulencePropertiesConfig
from neofoam.viscosity.config import TransportPropertiesConfig

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "tube_bank_manifest.json"
TRI_SURFACE = HERE.parent / "constant" / "triSurface"  # test/e2e/constant/triSurface

# --- hardcoded laminar physics (the knobs an agent would otherwise pick) ----------
INLET_VELOCITY = 1.0
NU = 1.5e-5
END_TIME = 0.5
DELTA_T = 2.0e-3
WRITE_INTERVAL = 50

# Boundary condition per (field, manifest role). ``frontBack`` is role ``empty`` but is
# realised as a ``symmetry`` mesh patch (see mesh_inputs._PATCH_TYPE), so its BC matches.
_U_BC: dict[PatchRole, dict[str, Any]] = {
    PatchRole.inlet: {"type": "fixedValue", "value": [INLET_VELOCITY, 0.0, 0.0]},
    PatchRole.outlet: {"type": "zeroGradient"},
    PatchRole.wall: {"type": "noSlip"},
    PatchRole.symmetry: {"type": "symmetry"},
    PatchRole.empty: {"type": "symmetry"},
}
_P_BC: dict[PatchRole, dict[str, Any]] = {
    PatchRole.inlet: {"type": "zeroGradient"},
    PatchRole.outlet: {"type": "fixedValue", "value": 0.0},
    PatchRole.wall: {"type": "zeroGradient"},
    PatchRole.symmetry: {"type": "symmetry"},
    PatchRole.empty: {"type": "symmetry"},
}

# fvSchemes / fvSolution for a laminar PIMPLE run. The Pimple config classes accumulate
# keys across every family bound to the solver (incl. the optional Boussinesq slice), so
# each declared key must be present; the extra p_rgh/rhok entries are inert for laminar.
_FV_SCHEMES: dict[str, Any] = {
    "ddtSchemes": {"ddt(U)": "Euler"},
    "gradSchemes": {
        "grad(U)": "Gauss linear",
        "grad(p)": "Gauss linear",
        "grad(p_rgh)": "Gauss linear",
    },
    "divSchemes": {
        "div(phi,U)": "Gauss linearUpwind grad(U)",
        "div((nuEff*dev2(T(grad(U)))))": "Gauss linear",
    },
    "laplacianSchemes": {
        "laplacian(nuEff,U)": "Gauss linear corrected",
        "laplacian(rAU,p)": "Gauss linear corrected",
        "laplacian(rAUf,p_rgh)": "Gauss linear corrected",
    },
    "interpolationSchemes": {
        "flux(HbyA)": "linear",
        "interpolate(rAU)": "linear",
        "dotInterpolate(S,U_0)": "linear",
        "flux(U)": "linear",
    },
    "snGradSchemes": {
        "snGrad(p)": "corrected",
        "snGrad(rhok)": "corrected",
        "snGrad(p_rgh)": "corrected",
    },
}
_P_SOLVER = {
    "solver": "GAMG",
    "smoother": "GaussSeidel",
    "tolerance": 1e-6,
    "relTol": 0.05,
}
_U_SOLVER = {
    "solver": "smoothSolver",
    "smoother": "symGaussSeidel",
    "tolerance": 1e-8,
    "relTol": 0.1,
}
_FV_SOLUTION: dict[str, Any] = {
    "solvers": {
        # PIMPLE solves ``<field>`` on inner correctors and ``<field>Final`` (relTol 0)
        # on the last corrector, so both must be present.
        "p": _P_SOLVER,
        "pFinal": {**_P_SOLVER, "relTol": 0},
        "U": _U_SOLVER,
        "UFinal": {**_U_SOLVER, "relTol": 0},
        "p_rgh": _P_SOLVER,
        "p_rghFinal": {**_P_SOLVER, "relTol": 0},
    },
    "PIMPLE": {
        "nCorrectors": 2,
        "nNonOrthogonalCorrectors": 1,
        "pRefCell": 0,
        "pRefValue": 0,
    },
}


def preprocess_chain() -> PreprocessConfig:
    """The blockMesh → snappyHexMesh → checkMesh pipeline (fills the existing config)."""
    return PreprocessConfig(
        tools=[
            {"tool": "blockMesh"},
            {"tool": "snappyHexMesh", "depends_on": ["blockMesh"]},
            {"tool": "checkMesh", "depends_on": ["snappyHexMesh"]},
        ]
    )


def _boundary_field(
    manifest: PatchManifest, bc_by_role: dict[PatchRole, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """A BC per patch (incl. the snappy-created ``tubes``), keyed off its role."""
    return {patch.name: dict(bc_by_role[patch.role]) for patch in manifest.patches}


def physics_configs(manifest: PatchManifest) -> list[BaseConfig]:
    """The solver-owned physics configs, filled with the hardcoded laminar values."""
    cfgs = configurations(incompressibleFluid)
    return [
        cfgs["UFieldConfig"](boundaryField=_boundary_field(manifest, _U_BC)),
        cfgs["pFieldConfig"](boundaryField=_boundary_field(manifest, _P_BC)),
        cfgs["Pimple_fvSchemes"].model_validate(_FV_SCHEMES),
        cfgs["Pimple_fvSolution"].model_validate(_FV_SOLUTION),
        TransportPropertiesConfig(transportModel="Newtonian", nu=NU),
        TurbulencePropertiesConfig(simulationType="laminar"),
        ControlDictConfig(
            application="pimpleFoam",
            endTime=END_TIME,
            deltaT=DELTA_T,
            writeInterval=WRITE_INTERVAL,
            writeControl="timeStep",
        ),
    ]


def build_tube_bank(
    case_dir: str | Path, *, manifest: Optional[PatchManifest] = None
) -> Path:
    """Author a complete, runnable laminar tube-bank case in ``case_dir``.

    Stages the geometry, fills every solver config, writes them, and gates on
    :func:`require_configs`. Returns the case directory.
    """
    case = Path(case_dir)
    manifest = manifest or PatchManifest.load(MANIFEST)

    stage_geometry(TRI_SURFACE, case, manifest=manifest)

    # OpenFOAM-strategy configs go through write_configs (merges co-owners, injects
    # headers); the YAML preprocess enable-file is saved via its own strategy.
    write_configs(
        [block_mesh_dict(manifest), snappy_dict(manifest), *physics_configs(manifest)],
        case_dir=case,
    )
    preprocess_chain().save(case_dir=case)

    require_configs(case)
    return case


def main(argv: Optional[list[str]] = None) -> int:
    """Author the case and (unless ``--no-run``) mesh + solve it in an isolated process."""
    import argparse
    import subprocess
    import sys

    parser = argparse.ArgumentParser(description="Author + run the tube-bank case.")
    parser.add_argument("case_dir", help="Target case directory to author.")
    parser.add_argument(
        "--no-run", action="store_true", help="Only author the case (for review)."
    )
    args = parser.parse_args(argv)

    case = build_tube_bank(args.case_dir)
    print(f"case authored: {case}")
    if args.no_run:
        return 0

    # Normal launch: neofoam.e2e.solve runs the preprocess DAG (mesh) then solves, in an
    # isolated process (in-process pybFoam SIGBUSes at GC teardown).
    return subprocess.run(
        [sys.executable, "-m", "neofoam.e2e.solve", str(case)]
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
