# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Compare OF turbulence.correct() vs NeoN correct() on one time step.

Uses the same NeonTurbulenceModel.detect() dispatch as the solver.
Parametrized over turbulence models via TURB_MODEL env var.

Run:
    uv run pytest test/turbulence/test_turbulence_correct.py -v -s
    TURB_MODEL=kEpsilon uv run pytest test/turbulence/test_turbulence_correct.py -v -s
"""

import ctypes
import ctypes.util
import os
import shutil
import signal
import sys
from dataclasses import dataclass
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
from neofoam.turbulenceModels import NeonTurbulenceModel

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
    return np.asarray(f.internal_vector().copy_to_host())


# ---------------------------------------------------------------------------
# Model configs — what each model needs for the test
# ---------------------------------------------------------------------------

_TURB_PROPS_TEMPLATE = """\
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      turbulenceProperties;
}}
simulationType      RAS;
RAS
{{
    RASModel        {ras_model};
    turbulence      on;
    printCoeffs     on;
}}
"""


def _write_turb_props(path: Path, ras_model: str) -> None:
    path.write_text(_TURB_PROPS_TEMPLATE.format(ras_model=ras_model))


@dataclass
class ModelConfig:
    """Configuration for testing one turbulence model."""

    name: str
    ras_model: str
    comparison_fields: list[str]
    write_fields: Callable[[Path, np.ndarray, float], None]
    rtol: float = 1e-2


def _write_sa_fields(case_dir: Path, cc: np.ndarray, nu: float) -> None:
    zero = case_dir / "0"
    nuTilda_vals = compute_nuTilda(cc, nu=nu)
    nut_vals = compute_nut_from_nuTilda(nuTilda_vals, nu=nu)
    write_scalar_field(
        zero / "nuTilda", "nuTilda", "[0 2 -1 0 0 0 0]",
        nuTilda_vals, _read_boundary_block(zero / "nuTilda"),
    )
    write_scalar_field(
        zero / "nut", "nut", "[0 2 -1 0 0 0 0]",
        nut_vals, _read_boundary_block(zero / "nut"),
    )


def _write_ke_fields(case_dir: Path, cc: np.ndarray, nu: float) -> None:
    zero = case_dir / "0"
    k_vals = compute_k(cc, nu=nu)
    eps_vals = compute_epsilon(cc, nu=nu)
    nut_vals = compute_nut_from_k_epsilon(k_vals, eps_vals)
    write_scalar_field(
        zero / "k", "k", "[0 2 -2 0 0 0 0]",
        k_vals, _read_boundary_block(zero / "k"),
    )
    write_scalar_field(
        zero / "epsilon", "epsilon", "[0 2 -3 0 0 0 0]",
        eps_vals, _read_boundary_block(zero / "epsilon"),
    )
    write_scalar_field(
        zero / "nut", "nut", "[0 2 -1 0 0 0 0]",
        nut_vals, _read_boundary_block(zero / "nut"),
    )


MODELS: dict[str, ModelConfig] = {
    "SpalartAllmaras": ModelConfig(
        name="SpalartAllmaras",
        ras_model="SpalartAllmaras",
        comparison_fields=["nuTilda"],
        write_fields=_write_sa_fields,
    ),
    "kEpsilon": ModelConfig(
        name="kEpsilon",
        ras_model="kEpsilon",
        comparison_fields=["k", "epsilon"],
        write_fields=_write_ke_fields,
        rtol=2e-2,
    ),
}


# ---------------------------------------------------------------------------
# Context for the test
# ---------------------------------------------------------------------------


@dataclass
class TurbContext:
    """All data needed for the turbulence comparison test."""

    config: ModelConfig
    # OF side
    of_turb: Any
    of_lam: Any
    of_fields: dict[str, Any]  # comparison fields from OF registry
    # NeoN side — uses the same interface as the solver
    neon_model: Any  # NeonTurbulenceModel instance
    neon_fields: dict[str, Any]  # NeoN fields by name
    neon_rt: Any
    nu_value: float


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def turb_context() -> TurbContext:
    """Set up OF + NeoN turbulence test from TURB_MODEL env var."""
    model_name = os.environ.get("TURB_MODEL", "SpalartAllmaras")
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODELS)}")
    cfg = MODELS[model_name]

    case_dir = CASE_ROOT / f"turb_{cfg.name}"
    case_source = Path(__file__).parent / "turbTest"
    original_dir = Path.cwd()

    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(case_source, case_dir)
    os.chdir(case_dir)
    _disable_fpe()

    # --- Mesh ---
    args = pyf.argList(["test"])
    rt = pyf.Time(args)
    _disable_fpe()
    mesh = generate_blockmesh(rt, pyf.dictionary.read("system/blockMeshDict"))
    _disable_fpe()

    # --- Write turbulenceProperties + analytical fields ---
    _write_turb_props(case_dir / "constant" / "turbulenceProperties", cfg.ras_model)
    cc = np.array(mesh.C().internalField())
    nu = 1e-5
    write_vector_field(
        case_dir / "0" / "U", "U", "[0 1 -1 0 0 0 0]",
        compute_velocity(cc), _read_boundary_block(case_dir / "0" / "U"),
    )
    cfg.write_fields(case_dir, cc, nu)

    # --- OF setup ---
    U_of = volVectorField.read_field(mesh, "U")
    phi_of = pyf.createPhi(U_of)
    lam = singlePhaseTransportModel(U_of, phi_of)
    turb = incompressibleTurbulenceModel.New(U_of, phi_of, lam)

    of_fields: dict[str, Any] = {}
    for name in cfg.comparison_fields:
        of_fields[name] = volScalarField.from_registry(mesh, name)

    # --- NeoN setup — same dispatch as solver ---
    _disable_fpe()
    rt_nn = nfb.create_adapter_run_time(rt)
    solvers = rt_nn.fv_solution_dict.subDict("solvers")
    for name in ["p", "U", "nuTilda", "k", "epsilon"]:
        if solvers.contains(name):
            solvers.insert_dict(name, nfb.map_fv_solution(solvers.subDict(name)))
    rt_nn.fv_schemes_dict = nfb.map_fv_schemes(rt_nn.fv_schemes_dict)

    # Detect model using PluginSystem — same as solver
    neon_model = NeonTurbulenceModel.detect()
    assert type(neon_model).__name__ != "NeonLaminar", (
        f"Expected turbulence model for {cfg.name}, got laminar fallback"
    )

    nu_value = float(nfb.read_transport_viscosity(rt_nn))

    # Read NeoN fields
    neon_fields: dict[str, Any] = {
        "U": nfb.read_vector_volume_field(rt_nn, "U"),
        "phi": nfb.create_phi(rt_nn, "U"),
        "nut": nfb.read_scalar_volume_field(rt_nn, "nut"),
    }
    for name in cfg.comparison_fields:
        neon_fields[name] = nfb.read_scalar_volume_field(rt_nn, name)

    # SA-specific fields
    if cfg.name == "SpalartAllmaras":
        neon_fields["d"] = nfb.compute_wall_distance(rt_nn)

    ctx = TurbContext(
        config=cfg,
        of_turb=turb,
        of_lam=lam,
        of_fields=of_fields,
        neon_model=neon_model,
        neon_fields=neon_fields,
        neon_rt=rt_nn,
        nu_value=nu_value,
    )

    yield ctx

    os.chdir(original_dir)
    if case_dir.exists():
        shutil.rmtree(case_dir)


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


def test_turbulence_correct(turb_context: TurbContext) -> None:
    """Compare OF turbulence.correct() vs NeoN model.correct(ctx).

    Uses the same NeonTurbulenceModel interface as the solver.
    """
    tc = turb_context
    cfg = tc.config

    # Save initial OF field values
    initials: dict[str, np.ndarray] = {}
    saved: dict[str, Any] = {}
    for name, field in tc.of_fields.items():
        initials[name] = _of(field).copy()
        saved[name] = volScalarField(field)

    # --- OF: turbulence.correct() ---
    tc.of_lam.correct()
    tc.of_turb.correct()

    of_results: dict[str, np.ndarray] = {}
    for name, field in tc.of_fields.items():
        of_results[name] = _of(field).copy()
        change = np.max(np.abs(of_results[name] - initials[name]))
        print(f"  OF {name} change: {change:.6e}")
        assert change > 1e-10, f"OF correct() did not change {name}"

    # Restore OF fields
    for name, field in tc.of_fields.items():
        field.assign(saved[name])

    # --- NeoN: model.correct(ctx) — same interface as solver ---
    from neofoam.framework.context import Context

    neon_ctx = Context(
        fields=dict(tc.neon_fields),
        models={
            "neon_runtime": tc.neon_rt,
            "nu_laminar_value": tc.nu_value,
        },
    )

    # SA needs sa_config in models
    if cfg.name == "SpalartAllmaras":
        from neofoam.solver.incompressibleFluid.models.spalartAllmaras import (
            SpalartAllmarasConfig,
        )
        neon_ctx.models["sa_config"] = SpalartAllmarasConfig()
    elif cfg.name == "kEpsilon":
        from neofoam.turbulenceModels.kEpsilon import KEpsilonConfig
        neon_ctx.models["ke_config"] = KEpsilonConfig()

    updates = tc.neon_model.correct(neon_ctx)

    # --- Compare ---
    print(f"\n{'='*60}")
    print(f"  {cfg.name}: OF vs NeoN comparison")
    print(f"{'='*60}")

    for name in cfg.comparison_fields:
        of_arr = of_results[name]
        nn_arr = _nn(tc.neon_fields[name])

        abs_diff = float(np.max(np.abs(nn_arr - of_arr)))
        scale = float(np.max(np.abs(of_arr))) + 1e-30
        rel_diff = abs_diff / scale
        pct_within_1 = float(
            np.mean(np.abs(nn_arr - of_arr) / (np.abs(of_arr) + 1e-30) < 0.01)
        ) * 100

        print(f"  {name:12s}  abs={abs_diff:.6e}  rel={rel_diff:.6e}  "
              f"scale={scale:.6e}  within_1%={pct_within_1:.1f}%")

        np.testing.assert_allclose(
            nn_arr, of_arr, rtol=cfg.rtol,
            err_msg=f"{name}: NeoN vs OF mismatch (rtol={cfg.rtol})",
        )
