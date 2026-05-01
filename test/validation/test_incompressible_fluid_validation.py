# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""
Validation tests for the real incompressibleFluid solver operations.

Verifies that @fvSchemes.add / @fvSolution.add decorators on the real solver
operations produce the correct requirements, and that verification catches
missing/invalid entries.
"""

from neofoam.foam.requirements import SchemeRequirement, SolverRequirement
from neofoam.foam.verification import (
    collect_requirements,
    collect_requirements_from_models,
    verify_fvschemes,
    verify_fvsolution,
    verify_solver_setup,
)
from neofoam.framework.types import OperationDef

# Import real solver operations
from neofoam.solver.incompressibleFluid.models.pressure_velocity.pimpleAlgorithm import (
    continuity as pimple_continuity,
    continuity_boussinesq as pimple_continuity_boussinesq,
    momentum as pimple_momentum,
    momentum_boussinesq as pimple_momentum_boussinesq,
    pimple,
)
from neofoam.solver.incompressibleFluid.models.pressure_velocity.simpleAlgorithm import (
    continuity as simple_continuity,
    momentum as simple_momentum,
)
from neofoam.solver.incompressibleFluid.models.boussinesq import (
    solve_energy,
)
from neofoam.solver.incompressibleFluid.models.spalartAllmaras import (
    spalart_allmaras,
    turbulence_correction,
)


# ============================================================================
# Helpers
# ============================================================================


def _op(func: object, name: str) -> OperationDef:
    return OperationDef(
        func=func,
        name=name,
        scheme_requirements=getattr(func, "_scheme_requirements", []),
        solver_requirements=getattr(func, "_solver_requirements", []),
    )


PITZ_DAILY_SA_SCHEMES = {
    "ddtSchemes": {"default": "Euler"},
    "divSchemes": {
        "default": "none",
        "div(phi,U)": "Gauss linearUpwind grad(U)",
        "div(phiHbyA)": "Gauss linear",
        "div(phi,nuTilda)": "Gauss upwind",
    },
    "gradSchemes": {"default": "Gauss linear"},
    "laplacianSchemes": {"default": "Gauss linear corrected"},
    "snGradSchemes": {"default": "corrected"},
    "interpolationSchemes": {"default": "linear"},
    "wallDist": {"method": "meshWave"},
}

PITZ_DAILY_SA_SOLUTION = {
    "solvers": {
        "p": {"solver": "GAMG", "tolerance": 1e-7},
        "U": {"solver": "smoothSolver", "tolerance": 1e-5},
        "nuTilda": {"solver": "smoothSolver", "tolerance": 1e-8},
    },
}

HOT_ROOM_SCHEMES = {
    "ddtSchemes": {"default": "Euler"},
    "divSchemes": {
        "default": "none",
        "div(phi,U)": "Gauss upwind",
        "div(phiHbyA)": "Gauss linear",
        "div(phi,T)": "Gauss upwind",
    },
    "gradSchemes": {"default": "Gauss linear"},
    "laplacianSchemes": {"default": "Gauss linear corrected"},
    "snGradSchemes": {"default": "corrected"},
    "interpolationSchemes": {"default": "linear"},
}

HOT_ROOM_SOLUTION = {
    "solvers": {
        "p_rgh": {"solver": "GAMG", "tolerance": 1e-7},
        "U": {"solver": "smoothSolver", "tolerance": 1e-5},
        "T": {"solver": "PBiCGStab", "tolerance": 1e-6},
    },
}


# ============================================================================
# Decorators attached correctly to real operations
# ============================================================================


def test_pimple_momentum_has_schemes() -> None:
    reqs = getattr(pimple_momentum, "_scheme_requirements", [])
    assert SchemeRequirement("ddtSchemes", "ddt(U)") in reqs
    assert SchemeRequirement("divSchemes", "div(phi,U)") in reqs
    assert SchemeRequirement("gradSchemes", "grad(U)") in reqs
    assert SchemeRequirement("laplacianSchemes", "laplacian(nuEff,U)") in reqs


def test_pimple_continuity_has_schemes() -> None:
    reqs = getattr(pimple_continuity, "_scheme_requirements", [])
    assert SchemeRequirement("laplacianSchemes", "laplacian(rAU,p)") in reqs
    assert SchemeRequirement("snGradSchemes", "snGrad(p)") in reqs
    assert SchemeRequirement("interpolationSchemes", "flux(HbyA)") in reqs


def test_simple_momentum_no_ddt() -> None:
    reqs = getattr(simple_momentum, "_scheme_requirements", [])
    sections = {r.section for r in reqs}
    assert "ddtSchemes" not in sections
    assert "divSchemes" in sections


def test_sa_has_wallDist_and_nuTilda() -> None:
    reqs = getattr(turbulence_correction, "_scheme_requirements", [])
    assert SchemeRequirement("divSchemes", "div(phi,nuTilda)") in reqs
    assert SchemeRequirement("wallDist", "method") in reqs
    assert getattr(turbulence_correction, "_solver_requirements", []) == [
        SolverRequirement("nuTilda")
    ]


def test_boussinesq_has_div_phi_T() -> None:
    reqs = getattr(solve_energy, "_scheme_requirements", [])
    assert SchemeRequirement("divSchemes", "div(phi,T)") in reqs
    assert getattr(solve_energy, "_solver_requirements", []) == [SolverRequirement("T")]


def test_boussinesq_continuity_solves_p_rgh() -> None:
    reqs = getattr(pimple_continuity_boussinesq, "_solver_requirements", [])
    assert reqs == [SolverRequirement("p_rgh")]


def test_requirements_stored_on_operation_def() -> None:
    """BaseSpec.operation() copies decorator metadata to OperationDef."""
    pimple_ops = {op.name: op for op in pimple._operations}
    assert (
        len(pimple_ops["momentum"].scheme_requirements) == 4
    )  # ddt, div, grad, laplacian
    assert len(pimple_ops["momentum"].solver_requirements) == 1

    sa_ops = {op.name: op for op in spalart_allmaras._operations}
    assert (
        len(sa_ops["turbulence_correction"].scheme_requirements) == 5
    )  # ddt, div, grad, laplacian, wallDist
    assert len(sa_ops["turbulence_correction"].solver_requirements) == 1


# ============================================================================
# pitzDaily_SA: PIMPLE + Spalart-Allmaras
# ============================================================================


def _pitzDaily_SA_ops() -> list[OperationDef]:
    return [
        _op(pimple_momentum, "momentum"),
        _op(pimple_continuity, "continuity"),
        _op(turbulence_correction, "turbulence_correction"),
    ]


def test_pitzDaily_SA_valid() -> None:
    schemes, solvers = collect_requirements(_pitzDaily_SA_ops())
    assert verify_fvschemes(PITZ_DAILY_SA_SCHEMES, schemes) == []
    assert verify_fvsolution(PITZ_DAILY_SA_SOLUTION, solvers) == []


def test_pitzDaily_SA_missing_nuTilda_scheme() -> None:
    schemes, solvers = collect_requirements(_pitzDaily_SA_ops())
    fv_no_sa = {
        "ddtSchemes": {"default": "Euler"},
        "divSchemes": {"div(phi,U)": "Gauss linear", "div(phiHbyA)": "Gauss linear"},
        "gradSchemes": {"default": "Gauss linear"},
        "laplacianSchemes": {"default": "Gauss linear corrected"},
        "snGradSchemes": {"default": "corrected"},
        "interpolationSchemes": {"default": "linear"},
    }
    errors = verify_fvschemes(fv_no_sa, schemes)
    error_fields = {e.field for e in errors}
    assert "divSchemes.div(phi,nuTilda)" in error_fields
    assert "wallDist.method" in error_fields


def test_pitzDaily_SA_missing_nuTilda_solver() -> None:
    _, solvers = collect_requirements(_pitzDaily_SA_ops())
    errors = verify_fvsolution({"solvers": {"p": {}, "U": {}}}, solvers)
    assert any("nuTilda" in e.field for e in errors)


def test_pitzDaily_SA_invalid_ddt_value() -> None:
    from neofoam.foam.schemes import DdtScheme

    schemes, _ = collect_requirements(_pitzDaily_SA_ops())
    fv_bad_ddt = dict(PITZ_DAILY_SA_SCHEMES)
    fv_bad_ddt["ddtSchemes"] = {"ddt(U)": "invalidScheme"}
    errors = verify_fvschemes(
        fv_bad_ddt, schemes, scheme_type_map={"ddtSchemes": DdtScheme}
    )
    assert any(e.error_type == "invalid_scheme" for e in errors)


# ============================================================================
# hotRoom: PIMPLE + Boussinesq (buoyant variant)
# ============================================================================


def _hotRoom_ops() -> list[OperationDef]:
    return [
        _op(pimple_momentum_boussinesq, "momentum"),
        _op(pimple_continuity_boussinesq, "continuity"),
        _op(solve_energy, "solve_energy"),
    ]


def test_hotRoom_valid() -> None:
    schemes, solvers = collect_requirements(_hotRoom_ops())
    assert verify_fvschemes(HOT_ROOM_SCHEMES, schemes) == []
    assert verify_fvsolution(HOT_ROOM_SOLUTION, solvers) == []


def test_hotRoom_needs_p_rgh_not_p() -> None:
    _, solvers = collect_requirements(_hotRoom_ops())
    fields = {r.field for r in solvers}
    assert "p_rgh" in fields
    assert "p" not in fields


def test_hotRoom_needs_T_solver() -> None:
    _, solvers = collect_requirements(_hotRoom_ops())
    assert SolverRequirement("T") in solvers


def test_hotRoom_needs_div_phi_T() -> None:
    schemes, _ = collect_requirements(_hotRoom_ops())
    keys = {(r.section, r.key) for r in schemes}
    assert ("divSchemes", "div(phi,T)") in keys


def test_hotRoom_no_sa_requirements() -> None:
    schemes, solvers = collect_requirements(_hotRoom_ops())
    keys = {(r.section, r.key) for r in schemes}
    assert ("divSchemes", "div(phi,nuTilda)") not in keys
    assert ("wallDist", "method") not in keys
    assert "nuTilda" not in {r.field for r in solvers}


# ============================================================================
# pitzDaily (plain PIMPLE, no optional models)
# ============================================================================


def _pitzDaily_ops() -> list[OperationDef]:
    return [
        _op(pimple_momentum, "momentum"),
        _op(pimple_continuity, "continuity"),
    ]


def test_pitzDaily_no_sa_no_boussinesq() -> None:
    schemes, solvers = collect_requirements(_pitzDaily_ops())
    keys = {(r.section, r.key) for r in schemes}
    assert ("divSchemes", "div(phi,nuTilda)") not in keys
    assert ("divSchemes", "div(phi,T)") not in keys
    assert ("wallDist", "method") not in keys
    assert {r.field for r in solvers} == {"U", "p"}


# ============================================================================
# pitzDaily_steady (SIMPLE, no ddt)
# ============================================================================


def _pitzDaily_steady_ops() -> list[OperationDef]:
    return [
        _op(simple_momentum, "momentum"),
        _op(simple_continuity, "continuity"),
    ]


def test_pitzDaily_steady_no_ddt() -> None:
    schemes, _ = collect_requirements(_pitzDaily_steady_ops())
    sections = {r.section for r in schemes}
    assert "ddtSchemes" not in sections


def test_pitzDaily_steady_valid() -> None:
    schemes, solvers = collect_requirements(_pitzDaily_steady_ops())
    fv_schemes = {
        "divSchemes": {
            "div(phi,U)": "Gauss linearUpwind grad(U)",
            "div(phiHbyA)": "Gauss linear",
        },
        "gradSchemes": {"default": "Gauss linear"},
        "laplacianSchemes": {"default": "Gauss linear corrected"},
        "snGradSchemes": {"default": "corrected"},
        "interpolationSchemes": {"default": "linear"},
    }
    fv_solution = {"solvers": {"p": {}, "U": {}}}
    assert verify_fvschemes(fv_schemes, schemes) == []
    assert verify_fvsolution(fv_solution, solvers) == []


# ============================================================================
# collect_requirements_from_models — pass ModelSpec directly
# ============================================================================


def test_collect_from_model_specs() -> None:
    """Pass real ModelSpec objects — collects from all their _operations."""
    schemes, solvers = collect_requirements_from_models([pimple, spalart_allmaras])
    scheme_keys = {(r.section, r.key) for r in schemes}
    assert ("ddtSchemes", "ddt(U)") in scheme_keys
    assert ("divSchemes", "div(phi,U)") in scheme_keys
    assert ("divSchemes", "div(phi,nuTilda)") in scheme_keys
    assert ("wallDist", "method") in scheme_keys
    assert {r.field for r in solvers} >= {"U", "p", "nuTilda"}


def test_collect_from_model_specs_pimple_only() -> None:
    schemes, solvers = collect_requirements_from_models([pimple])
    scheme_keys = {(r.section, r.key) for r in schemes}
    assert ("divSchemes", "div(phi,nuTilda)") not in scheme_keys
    assert "nuTilda" not in {r.field for r in solvers}


# ============================================================================
# verify_solver_setup — full pipeline
# ============================================================================


def test_verify_solver_setup_valid() -> None:
    """When collecting from ModelSpecs, ALL registered operations are included
    (both standard and buoyant variants). The fvSolution must cover all of them."""
    full_solution = {
        "solvers": {
            "p": {},
            "U": {},
            "nuTilda": {},
            "p_rgh": {},  # needed by continuity_boussinesq variant
        },
    }
    errors = verify_solver_setup(
        fv_schemes=PITZ_DAILY_SA_SCHEMES,
        fv_solution=full_solution,
        models=[pimple, spalart_allmaras],
    )
    assert errors == []


def test_verify_solver_setup_missing_sa() -> None:
    fv_no_sa = {
        "ddtSchemes": {"default": "Euler"},
        "divSchemes": {"div(phi,U)": "Gauss linear", "div(phiHbyA)": "Gauss linear"},
        "gradSchemes": {"default": "Gauss linear"},
        "laplacianSchemes": {"default": "Gauss linear corrected"},
        "snGradSchemes": {"default": "corrected"},
        "interpolationSchemes": {"default": "linear"},
    }
    errors = verify_solver_setup(
        fv_schemes=fv_no_sa,
        fv_solution={"solvers": {"p": {}, "U": {}, "p_rgh": {}}},
        models=[pimple, spalart_allmaras],
    )
    error_fields = {e.field for e in errors}
    assert "divSchemes.div(phi,nuTilda)" in error_fields
    assert "wallDist.method" in error_fields
    assert "solvers.nuTilda" in error_fields


def test_verify_solver_setup_with_invalid_config() -> None:
    """Layer 3: model config constraints caught alongside scheme errors."""
    from neofoam.solver.incompressibleFluid.models.boussinesq import BoussinesqConfig

    bad_config = BoussinesqConfig.model_construct(
        Pr=-1.0, Prt=0.85, beta=3e-3, TRef=300.0, hRef=0.0
    )
    errors = verify_solver_setup(
        fv_schemes=PITZ_DAILY_SA_SCHEMES,
        fv_solution=PITZ_DAILY_SA_SOLUTION,
        models=[pimple],
        model_configs=[bad_config],
    )
    assert len(errors) >= 1
    assert any("Pr" in e.field for e in errors)
