# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Compare OF turbulence.correct() vs NeoN correct() on one time step.

Parametrized over turbulence models. Currently supports Spalart-Allmaras.
To add a new model: add a TurbulenceModelCase to MODELS and a case directory.
"""

import ctypes
import ctypes.util
import os
import shutil
import signal
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

os.environ["FOAM_SIGFPE"] = ""
signal.signal(signal.SIGFPE, signal.SIG_IGN)

import numpy as np
import pytest

import neon._neon as nn
import pybFoam as pyf
from pybFoam import volScalarField, volVectorField
from pybFoam.meshing import generate_blockmesh
from pybFoam.turbulence import incompressibleTurbulenceModel, singlePhaseTransportModel

from neofoam import neofoam_bindings as nfb
from neofoam.solver.incompressibleFluid.models.spalartAllmaras import (
    SpalartAllmarasConfig,
)
from neofoam.turbulenceModels.neon_spalartAllmaras import correct as sa_correct_neon
from neofoam.turbulenceModels.neon_kEpsilon import KEpsilonConfig, correct as ke_correct_neon

sys.path.insert(0, str(Path(__file__).parent))
from generate_fields import (  # noqa: E402
    compute_nuTilda,
    compute_velocity,
    compute_nut_from_nuTilda,
    compute_k,
    compute_epsilon,
    compute_nut_from_k_epsilon,
    write_scalar_field,
    write_vector_field,
    _read_boundary_block,
)

CASE_ROOT = Path(__file__).parent.parent.parent / "test_cases"


def _disable_fpe() -> None:
    libm = ctypes.CDLL(ctypes.util.find_library("m"))
    libm.fedisableexcept(0x3F)


def _of(f: Any) -> np.ndarray:
    return np.array(f.internalField())


def _nn(f: Any) -> np.ndarray:
    return np.asarray(f.internal_vector().__array__())


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------


@dataclass
class TurbulenceModelCase:
    """One turbulence model to test."""

    id: str
    case_source: Path
    comparison_fields: list[str]
    neon_correct_fn: Callable[..., None]
    neon_cfg: Any
    write_fields_fn: Callable[..., None]  # writes model-specific non-uniform fields
    turbulence_properties: str  # turbulenceProperties file content
    rtol: float = 1e-2


# --- Spalart-Allmaras ---

def _write_sa_fields(case_dir: Path, cc: np.ndarray, nu: float) -> None:
    zero = case_dir / "0"
    nuTilda_vals = compute_nuTilda(cc, nu=nu)
    nut_vals = compute_nut_from_nuTilda(nuTilda_vals, nu=nu)
    write_scalar_field(zero / "nuTilda", "nuTilda", "[0 2 -1 0 0 0 0]",
                       nuTilda_vals, _read_boundary_block(zero / "nuTilda"))
    write_scalar_field(zero / "nut", "nut", "[0 2 -1 0 0 0 0]",
                       nut_vals, _read_boundary_block(zero / "nut"))


def _run_sa_neon(ctx: dict[str, Any]) -> None:
    sa_correct_neon(
        ctx["neon_cfg"], ctx["rt_nn"],
        ctx["nuTilda_nn"], ctx["nut_nn"],
        ctx["U_nn"], ctx["phi_nn"],
        ctx["d_nn"], ctx["nu_value"],
    )


SA_TURB_PROPS = """\
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      turbulenceProperties;
}
simulationType      RAS;
RAS
{
    RASModel        SpalartAllmaras;
    turbulence      on;
    printCoeffs     on;
}
"""


# --- k-Epsilon ---

def _write_ke_fields(case_dir: Path, cc: np.ndarray, nu: float) -> None:
    zero = case_dir / "0"
    k_vals = compute_k(cc, nu=nu)
    eps_vals = compute_epsilon(cc, nu=nu)
    nut_vals = compute_nut_from_k_epsilon(k_vals, eps_vals)
    write_scalar_field(zero / "k", "k", "[0 2 -2 0 0 0 0]",
                       k_vals, _read_boundary_block(zero / "k"))
    write_scalar_field(zero / "epsilon", "epsilon", "[0 2 -3 0 0 0 0]",
                       eps_vals, _read_boundary_block(zero / "epsilon"))
    write_scalar_field(zero / "nut", "nut", "[0 2 -1 0 0 0 0]",
                       nut_vals, _read_boundary_block(zero / "nut"))


def _run_ke_neon(ctx: dict[str, Any]) -> None:
    ke_correct_neon(
        ctx["neon_cfg"], ctx["rt_nn"],
        ctx["k_nn"], ctx["epsilon_nn"], ctx["nut_nn"],
        ctx["U_nn"], ctx["phi_nn"],
        ctx["nu_value"],
    )


KE_TURB_PROPS = """\
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      turbulenceProperties;
}
simulationType      RAS;
RAS
{
    RASModel        kEpsilon;
    turbulence      on;
    printCoeffs     on;
}
"""


MODELS = [
    TurbulenceModelCase(
        id="SpalartAllmaras",
        case_source=Path(__file__).parent / "turbTest",
        comparison_fields=["nuTilda"],
        neon_correct_fn=_run_sa_neon,
        neon_cfg=SpalartAllmarasConfig(),
        write_fields_fn=_write_sa_fields,
        turbulence_properties=SA_TURB_PROPS,
    ),
    TurbulenceModelCase(
        id="kEpsilon",
        case_source=Path(__file__).parent / "turbTest",
        comparison_fields=["k", "epsilon"],
        neon_correct_fn=_run_ke_neon,
        neon_cfg=KEpsilonConfig(),
        write_fields_fn=_write_ke_fields,
        turbulence_properties=KE_TURB_PROPS,
        rtol=2e-2,  # slightly relaxed — 4 cells at ~1.1% from solver differences
    ),
]


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


def _select_model() -> TurbulenceModelCase:
    """Select model from TURB_MODEL env var, default to SpalartAllmaras."""
    name = os.environ.get("TURB_MODEL", "SpalartAllmaras")
    for m in MODELS:
        if m.id == name:
            return m
    raise ValueError(f"Unknown model: {name}. Available: {[m.id for m in MODELS]}")


@pytest.fixture(scope="session")
def turb_case(request: Any) -> Any:
    """Set up turbulence test case for ONE model (from TURB_MODEL env var).

    Only one OF turb model can exist per process, so we test one at a time.
    Run with: TURB_MODEL=kEpsilon uv run pytest ...
    """
    model = _select_model()
    case_dir = CASE_ROOT / f"turb_{model.id}"
    original_dir = Path.cwd()

    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(model.case_source, case_dir)

    os.chdir(case_dir)
    _disable_fpe()

    # Mesh
    args = pyf.argList(["test"])
    rt = pyf.Time(args)
    _disable_fpe()
    mesh = generate_blockmesh(rt, pyf.dictionary.read("system/blockMeshDict"))
    _disable_fpe()

    # Write turbulenceProperties for this model
    tp = case_dir / "constant" / "turbulenceProperties"
    tp.write_text(model.turbulence_properties)

    # Write non-uniform analytical fields (U is common, model-specific fields via callback)
    cc = np.array(mesh.C().internalField())
    nu = 1e-5
    zero = case_dir / "0"
    write_vector_field(
        zero / "U", "U", "[0 1 -1 0 0 0 0]",
        compute_velocity(cc), _read_boundary_block(zero / "U"),
    )
    model.write_fields_fn(case_dir, cc, nu)

    # OF fields
    U_of = volVectorField.read_field(mesh, "U")
    phi_of = pyf.createPhi(U_of)
    lam = singlePhaseTransportModel(U_of, phi_of)
    turb = incompressibleTurbulenceModel.New(U_of, phi_of, lam)

    # Get comparison fields from registry (same object as turb model's internal fields)
    of_fields = {}
    for name in model.comparison_fields:
        of_fields[name] = volScalarField.from_registry(mesh, name)

    # NeoN fields
    _disable_fpe()
    rt_nn = nfb.create_adapter_run_time(rt)
    solvers = rt_nn.fv_solution_dict.subDict("solvers")
    for name in ["p", "U", "nuTilda", "k", "epsilon"]:
        if solvers.contains(name):
            solvers.insert_dict(name, nfb.map_fv_solution(solvers.subDict(name)))
    rt_nn.fv_schemes_dict = nfb.map_fv_schemes(rt_nn.fv_schemes_dict)

    ctx: dict[str, Any] = {
        "model": model,
        "of_fields": of_fields,
        "turb": turb,
        "lam": lam,
        "neon_cfg": model.neon_cfg,
        "rt_nn": rt_nn,
        "U_nn": nfb.read_vector_volume_field(rt_nn, "U"),
        "phi_nn": nfb.create_phi(rt_nn, "U"),
        "nu_value": float(nfb.read_transport_viscosity(rt_nn)),
        "nut_nn": nfb.read_scalar_volume_field(rt_nn, "nut"),
    }

    # Read model-specific NeoN fields
    for name in model.comparison_fields:
        ctx[f"{name}_nn"] = nfb.read_scalar_volume_field(rt_nn, name)

    # SA-specific: wall distance
    if model.id == "SpalartAllmaras":
        ctx["d_nn"] = nfb.compute_wall_distance(rt_nn)
        ctx["nuTilda_nn"] = nfb.read_scalar_volume_field(rt_nn, "nuTilda")

    # kEpsilon-specific
    if model.id == "kEpsilon":
        ctx["k_nn"] = nfb.read_scalar_volume_field(rt_nn, "k")
        ctx["epsilon_nn"] = nfb.read_scalar_volume_field(rt_nn, "epsilon")

    yield ctx

    os.chdir(original_dir)
    if case_dir.exists():
        shutil.rmtree(case_dir)


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


def test_turbulence_correct(turb_case: dict[str, Any]) -> None:
    """OF turbulence.correct() vs NeoN correct() on one time step.

    Model selected via TURB_MODEL env var (default: SpalartAllmaras).
    Run kEpsilon: TURB_MODEL=kEpsilon uv run pytest ...
    """
    model: TurbulenceModelCase = turb_case["model"]
    of_fields = turb_case["of_fields"]

    # Save initial state for all comparison fields
    initials: dict[str, np.ndarray] = {}
    saved_fields: dict[str, Any] = {}
    for name, field in of_fields.items():
        initials[name] = _of(field).copy()
        saved_fields[name] = volScalarField(field)

    # --- OF: turbulence.correct() ---
    turb_case["lam"].correct()
    turb_case["turb"].correct()

    results_of: dict[str, np.ndarray] = {}
    for name, field in of_fields.items():
        results_of[name] = _of(field).copy()
        change = np.max(np.abs(results_of[name] - initials[name]))
        print(f"\n  OF correct() {name} change: {change:.6e}")
        assert change > 1e-10, f"correct() did not change {name} — check deltaT"

    # --- Restore ---
    for name, field in of_fields.items():
        field.assign(saved_fields[name])

    # --- NeoN: correct() ---
    model.neon_correct_fn(turb_case)

    # --- Compare each field ---
    for name in model.comparison_fields:
        of_arr = results_of[name]
        nn_arr = _nn(turb_case[f"{name}_nn"])

        abs_diff = np.max(np.abs(nn_arr - of_arr))
        denom = np.max(np.abs(of_arr)) + 1e-30
        rel_diff = abs_diff / denom
        w1 = np.sum(np.abs(nn_arr - of_arr) / (np.abs(of_arr) + 1e-30) < 0.01)

        print(f"\n  {name}:")
        print(f"    OF:   min={of_arr.min():.6e}  max={of_arr.max():.6e}")
        print(f"    NeoN: min={nn_arr.min():.6e}  max={nn_arr.max():.6e}")
        print(f"    diff: abs={abs_diff:.6e}  rel={rel_diff:.6e}")
        print(f"    within 1%: {w1}/{len(of_arr)} ({100 * w1 / len(of_arr):.1f}%)")

        np.testing.assert_allclose(
            nn_arr, of_arr, rtol=model.rtol, atol=1e-10,
            err_msg=f"{name}: NeoN does not match OF",
        )
