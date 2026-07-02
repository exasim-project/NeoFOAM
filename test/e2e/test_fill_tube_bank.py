# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The agent-driven driver: an agent fills the configs, we save + verify they run.

Two levels. The fast test drives a **stub agent** returning a recorded ``CaseSpec``
so the fill→save→gate path is deterministic (no network): it proves that whatever
the agent returns is written to a complete case. The slow test drives the **live
agent** (gated on an API key + a sourced OpenFOAM) and asserts the authored case
actually meshes + solves.
"""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("pybFoam")

from neofoam.agent.case_fill import build_case_output_model  # noqa: E402
from neofoam.e2e.config import REQUIRED_FILES  # noqa: E402
from neofoam.e2e.manifest import PatchManifest, PatchRole  # noqa: E402
from neofoam.framework.solver.configurations import (  # noqa: E402
    _snake_case,
    configurations,
)
from neofoam.solver.incompressibleFluid.configs import ControlDictConfig  # noqa: E402
from neofoam.solver.incompressibleFluid.incompressibleFluid import (  # noqa: E402
    incompressibleFluid,
)
from neofoam.turbulence.config import TurbulencePropertiesConfig  # noqa: E402
from neofoam.viscosity.config import TransportPropertiesConfig  # noqa: E402

# Load the driver, which lives under cases/ (not an importable package).
_DRIVER_PATH = Path(__file__).parent / "cases" / "fill_tube_bank.py"
_spec = importlib.util.spec_from_file_location("fill_tube_bank", _DRIVER_PATH)
assert _spec and _spec.loader
fill_tube_bank = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fill_tube_bank)


# --- a recorded "good" agent output (the determinism boundary) -------------------
_U_BC = {
    PatchRole.inlet: {"type": "fixedValue", "value": [1.0, 0.0, 0.0]},
    PatchRole.outlet: {"type": "zeroGradient"},
    PatchRole.wall: {"type": "noSlip"},
    PatchRole.symmetry: {"type": "symmetry"},
    PatchRole.empty: {"type": "symmetry"},
}
_P_BC = {
    PatchRole.inlet: {"type": "zeroGradient"},
    PatchRole.outlet: {"type": "fixedValue", "value": 0.0},
    PatchRole.wall: {"type": "zeroGradient"},
    PatchRole.symmetry: {"type": "symmetry"},
    PatchRole.empty: {"type": "symmetry"},
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
_FV_SCHEMES = {
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
_FV_SOLUTION = {
    "solvers": {
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


def _recorded_spec(manifest: PatchManifest) -> Any:
    """A filled ``CaseSpec`` a stub agent hands back (a known-good laminar case)."""
    cfgs = configurations(incompressibleFluid)
    configs = [
        cfgs["UFieldConfig"](
            boundaryField={p.name: dict(_U_BC[p.role]) for p in manifest.patches}
        ),
        cfgs["pFieldConfig"](
            boundaryField={p.name: dict(_P_BC[p.role]) for p in manifest.patches}
        ),
        cfgs["Pimple_fvSchemes"].model_validate(_FV_SCHEMES),
        cfgs["Pimple_fvSolution"].model_validate(_FV_SOLUTION),
        TransportPropertiesConfig(transportModel="Newtonian", nu=1.5e-5),
        TurbulencePropertiesConfig(simulationType="laminar"),
        ControlDictConfig(
            application="pimpleFoam",
            endTime=0.5,
            deltaT=0.002,
            writeInterval=50,
            writeControl="timeStep",
        ),
    ]
    case_spec_cls = build_case_output_model(solver=incompressibleFluid)
    return case_spec_cls(**{_snake_case(type(c).__name__): c for c in configs})


def _stub_agent(manifest: PatchManifest) -> Any:
    """An agent whose ``run_sync`` returns the recorded spec (no network)."""
    output = _recorded_spec(manifest)
    return SimpleNamespace(run_sync=lambda _prompt: SimpleNamespace(output=output))


def test_stub_agent_output_is_written_to_a_complete_case(tmp_path: Path) -> None:
    """Whatever the agent returns is saved into a complete, gate-passing case."""
    manifest = PatchManifest.load(fill_tube_bank.MANIFEST)
    case = fill_tube_bank.build_tube_bank(
        tmp_path, manifest=manifest, agent=_stub_agent(manifest)
    )
    for rel in REQUIRED_FILES:
        assert (case / rel).is_file(), f"missing required file: {rel}"
    u_text = (case / "0" / "U").read_text()
    for patch in manifest.patches:
        assert patch.name in u_text, f"0/U missing patch {patch.name}"


def test_geometry_is_deterministic_not_from_the_agent(tmp_path: Path) -> None:
    """The agent's geometry is ignored; the mesh dicts come from the manifest."""
    manifest = PatchManifest.load(fill_tube_bank.MANIFEST)
    fill_tube_bank.build_tube_bank(
        tmp_path, manifest=manifest, agent=_stub_agent(manifest)
    )
    # locationInMesh in the written snappy dict is the manifest point.
    snappy = (tmp_path / "system" / "snappyHexMeshDict").read_text()
    x, y, z = manifest.location_in_mesh
    assert f"( {x} {y} {z} )" in snappy or f"({x} {y} {z})" in snappy


@pytest.mark.slow
@pytest.mark.skipif(
    not (os.environ.get("ANTHROPIC_API_KEY") and os.environ.get("WM_PROJECT_DIR")),
    reason="needs ANTHROPIC_API_KEY + a sourced OpenFOAM",
)
def test_live_agent_fills_and_case_solves(tmp_path: Path) -> None:
    """The live agent fills the configs and the authored case meshes + solves."""
    case = fill_tube_bank.build_tube_bank(tmp_path)  # live LLM
    # Shorten the run so the smoke test is quick (write every step, stop after a few).
    ControlDictConfig(
        application="pimpleFoam",
        endTime=0.006,
        deltaT=0.002,
        writeControl="timeStep",
        writeInterval=1,
    ).save(case_dir=case)

    proc = subprocess.run(
        [sys.executable, "-m", "neofoam.e2e.solve", str(case)],
        capture_output=True,
        text=True,
    )
    time_dirs = [
        child.name
        for child in case.iterdir()
        if child.is_dir()
        and child.name.replace(".", "", 1).isdigit()
        and float(child.name) > 0
    ]
    assert time_dirs, (
        f"no time directory written\nSTDOUT:\n{proc.stdout[-1000:]}\nSTDERR:\n{proc.stderr[-1000:]}"
    )
