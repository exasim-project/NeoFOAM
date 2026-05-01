# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""
Tests for the dummy solver — verifies all config registration mechanisms.

The solver is defined in test/validation/dummy_solver/ (separate package).
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from neofoam.foam.requirements import SchemeRequirement, SolverRequirement
from neofoam.foam.verification import collect_requirements, verify_fvschemes, verify_fvsolution
from neofoam.framework.model import ModelSpec
from neofoam.framework.types import OperationDef

from .dummy_solver import (
    CoreAlgorithmConfig,
    FluidPropertiesConfig,
    ScalarTransportConfig,
    TimeConfig,
    WallModelConfig,
    core_algorithm,
    energy_equation,
    momentum,
    pressure_correction,
    scalar_transport,
    wall_model,
    wall_solve,
)


# ============================================================================
# Model registration
# ============================================================================


def test_core_model_is_model_spec() -> None:
    assert isinstance(core_algorithm, ModelSpec)
    assert core_algorithm.name == "core_algorithm"


def test_optional_models_are_model_specs() -> None:
    assert isinstance(wall_model, ModelSpec)
    assert isinstance(scalar_transport, ModelSpec)


def test_core_model_has_load() -> None:
    assert core_algorithm._load_func is not None


def test_optional_models_have_detect_and_resolve() -> None:
    assert wall_model._detect_func is not None
    assert wall_model._resolve_func is not None
    assert scalar_transport._detect_func is not None
    assert scalar_transport._resolve_func is not None


def test_operations_registered() -> None:
    assert {op.name for op in core_algorithm._operations} == {"momentum", "pressure_correction"}
    assert {op.name for op in wall_model._operations} == {"wall_solve"}
    assert {op.name for op in scalar_transport._operations} == {"energy_equation"}


# ============================================================================
# Config lifecycle
# ============================================================================


def test_core_instantiate() -> None:
    rt = core_algorithm.instantiate(case_dir=Path("."), entry={"name": "core"})
    assert isinstance(rt.config, CoreAlgorithmConfig)
    assert rt.config.nCorrectors == 2


def test_wall_model_instantiate() -> None:
    rt = wall_model.instantiate(case_dir=Path("."), entry={"name": "wall"})
    assert isinstance(rt.config, WallModelConfig)
    assert rt.config.sigma == pytest.approx(2.0 / 3.0)


def test_scalar_transport_instantiate() -> None:
    rt = scalar_transport.instantiate(case_dir=Path("."), entry={"name": "scalar"})
    assert isinstance(rt.config, ScalarTransportConfig)
    assert rt.config.Pr == pytest.approx(0.7)


def test_detect() -> None:
    assert wall_model.run_detect(Path(".")).detected is True
    assert scalar_transport.run_detect(Path(".")).detected is True


def test_core_config_constraint() -> None:
    with pytest.raises(ValidationError, match="nCorrectors"):
        CoreAlgorithmConfig(nCorrectors=0)


def test_wall_model_config_constraint() -> None:
    with pytest.raises(ValidationError, match="sigma"):
        WallModelConfig(sigma=0)


def test_scalar_transport_config_constraint() -> None:
    with pytest.raises(ValidationError, match="Pr"):
        ScalarTransportConfig(Pr=0)


def test_time_config_constraint() -> None:
    with pytest.raises(ValidationError, match="endTime"):
        TimeConfig(endTime=-1.0, deltaT=0.01)


def test_fluid_properties_constraint() -> None:
    with pytest.raises(ValidationError, match="nu"):
        FluidPropertiesConfig(nu=0)


# ============================================================================
# Scheme requirements
# ============================================================================


def test_momentum_schemes() -> None:
    reqs = getattr(momentum, "_scheme_requirements", [])
    assert SchemeRequirement("ddtSchemes", "ddt(U)") in reqs
    assert SchemeRequirement("divSchemes", "div(phi,U)") in reqs
    assert SchemeRequirement("gradSchemes", "grad(U)") in reqs
    assert SchemeRequirement("laplacianSchemes", "laplacian(nuEff,U)") in reqs


def test_pressure_correction_schemes() -> None:
    reqs = getattr(pressure_correction, "_scheme_requirements", [])
    assert SchemeRequirement("laplacianSchemes", "laplacian(rAU,p)") in reqs
    assert SchemeRequirement("snGradSchemes", "snGrad(p)") in reqs
    assert SchemeRequirement("interpolationSchemes", "flux(HbyA)") in reqs


def test_wall_model_schemes() -> None:
    reqs = getattr(wall_solve, "_scheme_requirements", [])
    assert SchemeRequirement("wallDist", "method") in reqs
    assert SchemeRequirement("divSchemes", "div(phi,nuTilda)") in reqs


def test_scalar_transport_schemes() -> None:
    reqs = getattr(energy_equation, "_scheme_requirements", [])
    assert SchemeRequirement("divSchemes", "div(phi,T)") in reqs


def test_solver_requirements() -> None:
    assert getattr(momentum, "_solver_requirements", []) == [SolverRequirement("U")]
    assert getattr(pressure_correction, "_solver_requirements", []) == [SolverRequirement("p")]
    assert getattr(wall_solve, "_solver_requirements", []) == [SolverRequirement("nuTilda")]
    assert getattr(energy_equation, "_solver_requirements", []) == [SolverRequirement("T")]


# ============================================================================
# Collect and verify
# ============================================================================

VALID_FV_SCHEMES = {
    "ddtSchemes": {"ddt(U)": "Euler", "ddt(nuTilda)": "Euler", "ddt(T)": "Euler"},
    "divSchemes": {
        "div(phi,U)": "Gauss upwind",
        "div(phi,nuTilda)": "Gauss upwind",
        "div(phi,T)": "Gauss upwind",
    },
    "gradSchemes": {"grad(U)": "Gauss linear", "grad(p)": "Gauss linear", "grad(nuTilda)": "Gauss linear"},
    "laplacianSchemes": {
        "laplacian(nuEff,U)": "Gauss linear corrected",
        "laplacian(rAU,p)": "Gauss linear corrected",
        "laplacian(DnuTildaEff,nuTilda)": "Gauss linear corrected",
        "laplacian(alphaEff,T)": "Gauss linear corrected",
    },
    "snGradSchemes": {"snGrad(p)": "corrected"},
    "interpolationSchemes": {"flux(HbyA)": "linear", "interpolate(rAU)": "linear"},
    "wallDist": {"method": "meshWave"},
}

VALID_FV_SOLUTION = {"solvers": {"p": {}, "U": {}, "nuTilda": {}, "T": {}}}


def _make_op(func: object, name: str) -> OperationDef:
    return OperationDef(
        func=func,
        name=name,
        scheme_requirements=getattr(func, "_scheme_requirements", []),
        solver_requirements=getattr(func, "_solver_requirements", []),
    )


def _all_ops() -> list[OperationDef]:
    return [
        _make_op(momentum, "momentum"),
        _make_op(pressure_correction, "pressure_correction"),
        _make_op(wall_solve, "wall_solve"),
        _make_op(energy_equation, "energy_equation"),
    ]


def test_valid_case_passes() -> None:
    schemes, solvers = collect_requirements(_all_ops())
    assert verify_fvschemes(VALID_FV_SCHEMES, schemes) == []
    assert verify_fvsolution(VALID_FV_SOLUTION, solvers) == []


def test_missing_wall_entries() -> None:
    schemes, solvers = collect_requirements(_all_ops())
    fv_no_wall = {k: v for k, v in VALID_FV_SCHEMES.items() if k != "wallDist"}
    errors = verify_fvschemes(fv_no_wall, schemes)
    assert any("wallDist" in e.field for e in errors)


def test_core_only_fewer_requirements() -> None:
    core_only = [_make_op(momentum, "momentum"), _make_op(pressure_correction, "pressure_correction")]
    schemes, solvers = collect_requirements(core_only)
    scheme_keys = {(r.section, r.key) for r in schemes}
    assert ("divSchemes", "div(phi,nuTilda)") not in scheme_keys
    assert ("divSchemes", "div(phi,T)") not in scheme_keys
    assert ("wallDist", "method") not in scheme_keys
    assert {r.field for r in solvers} == {"U", "p"}
