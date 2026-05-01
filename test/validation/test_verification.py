# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""
Plan 08 — Verification Logic.

Target: collect_requirements, verify_fvschemes, verify_fvsolution
from neofoam.foam.verification

Files:
  src/neofoam/foam/verification.py (new)
  src/neofoam/foam/requirements.py (SchemeRequirement, SolverRequirement, fvSchemes, fvSolution)
"""

from neofoam.foam.requirements import SchemeRequirement, SolverRequirement
from neofoam.foam.verification import (
    collect_requirements,
    verify_fvschemes,
    verify_fvsolution,
)
from neofoam.framework.types import OperationDef


def _noop() -> None:
    pass


def test_collect_deduplicates() -> None:
    ops = [
        OperationDef(
            func=_noop,
            name="op1",
            scheme_requirements=[SchemeRequirement("ddtSchemes", "default")],
        ),
        OperationDef(
            func=_noop,
            name="op2",
            scheme_requirements=[SchemeRequirement("ddtSchemes", "default")],
        ),
    ]
    schemes, _ = collect_requirements(ops)
    assert len(schemes) == 1


def test_collect_pitzDaily_SA() -> None:
    momentum = OperationDef(
        func=_noop,
        name="momentum",
        scheme_requirements=[
            SchemeRequirement("ddtSchemes", "default"),
            SchemeRequirement("divSchemes", "div(phi,U)"),
            SchemeRequirement("gradSchemes", "default"),
        ],
        solver_requirements=[SolverRequirement("U")],
    )
    continuity = OperationDef(
        func=_noop,
        name="continuity",
        scheme_requirements=[SchemeRequirement("laplacianSchemes", "default")],
        solver_requirements=[SolverRequirement("p")],
    )
    sa = OperationDef(
        func=_noop,
        name="turbulence_correction",
        scheme_requirements=[
            SchemeRequirement("divSchemes", "div(phi,nuTilda)"),
            SchemeRequirement("wallDist", "method"),
        ],
        solver_requirements=[SolverRequirement("nuTilda")],
    )

    schemes, solvers = collect_requirements([momentum, continuity, sa])
    scheme_keys = {(r.section, r.key) for r in schemes}
    assert ("divSchemes", "div(phi,U)") in scheme_keys
    assert ("divSchemes", "div(phi,nuTilda)") in scheme_keys
    assert ("wallDist", "method") in scheme_keys
    assert {r.field for r in solvers} == {"U", "p", "nuTilda"}


def test_verify_fvschemes_valid() -> None:
    fv_schemes = {
        "ddtSchemes": {"default": "Euler"},
        "divSchemes": {"div(phi,U)": "Gauss linear"},
    }
    errors = verify_fvschemes(
        fv_schemes,
        [
            SchemeRequirement("ddtSchemes", "default"),
            SchemeRequirement("divSchemes", "div(phi,U)"),
        ],
    )
    assert errors == []


def test_verify_fvschemes_missing_entry() -> None:
    errors = verify_fvschemes(
        {"divSchemes": {}}, [SchemeRequirement("divSchemes", "div(phi,nuTilda)")]
    )
    assert len(errors) == 1
    assert errors[0].error_type == "missing_entry"


def test_verify_fvschemes_missing_wallDist() -> None:
    errors = verify_fvschemes({}, [SchemeRequirement("wallDist", "method")])
    assert len(errors) == 1
    assert "wallDist" in errors[0].message


def test_verify_invalid_scheme_value() -> None:
    from neofoam.foam.schemes import DdtScheme

    errors = verify_fvschemes(
        {"ddtSchemes": {"default": "invalidDdt"}},
        [SchemeRequirement("ddtSchemes", "default")],
        scheme_type_map={"ddtSchemes": DdtScheme},
    )
    assert len(errors) == 1
    assert errors[0].error_type == "invalid_scheme"


def test_verify_valid_scheme_value() -> None:
    from neofoam.foam.schemes import DdtScheme

    errors = verify_fvschemes(
        {"ddtSchemes": {"default": "Euler"}},
        [SchemeRequirement("ddtSchemes", "default")],
        scheme_type_map={"ddtSchemes": DdtScheme},
    )
    assert errors == []


def test_verify_fvsolution_missing_solver() -> None:
    errors = verify_fvsolution({"solvers": {}}, [SolverRequirement("nuTilda")])
    assert len(errors) == 1
    assert "nuTilda" in errors[0].message


def test_all_errors_collected() -> None:
    scheme_errors = verify_fvschemes(
        {"divSchemes": {}},
        [
            SchemeRequirement("divSchemes", "div(phi,nuTilda)"),
            SchemeRequirement("wallDist", "method"),
        ],
    )
    solver_errors = verify_fvsolution({"solvers": {}}, [SolverRequirement("nuTilda")])
    all_errors = scheme_errors + solver_errors
    assert len(all_errors) == 3


def test_default_fallback_valid() -> None:
    """If specific key is missing but 'default' exists with a real scheme, it's valid."""
    fv = {"divSchemes": {"default": "Gauss linear"}}
    errors = verify_fvschemes(fv, [SchemeRequirement("divSchemes", "div(phi,U)")])
    assert errors == []


def test_default_none_not_a_valid_fallback() -> None:
    """'default: none' means no unlisted schemes allowed — should NOT be a valid fallback."""
    fv = {"divSchemes": {"default": "none"}}
    errors = verify_fvschemes(fv, [SchemeRequirement("divSchemes", "div(phi,nuTilda)")])
    assert len(errors) == 1
    assert "div(phi,nuTilda)" in errors[0].field


def test_regex_solver_key_match() -> None:
    """OpenFOAM regex key (U|nuTilda) matches requirements for U and nuTilda."""
    fv_sol = {"solvers": {"p": {}, "(U|nuTilda)": {}}}
    errors = verify_fvsolution(
        fv_sol, [SolverRequirement("U"), SolverRequirement("nuTilda")]
    )
    assert errors == []


def test_buoyant_variant_requires_p_rgh() -> None:
    std = OperationDef(
        func=_noop, name="continuity", solver_requirements=[SolverRequirement("p")]
    )
    bouss = OperationDef(
        func=_noop,
        name="continuity_boussinesq",
        solver_requirements=[SolverRequirement("p_rgh")],
    )

    _, solvers_std = collect_requirements([std])
    assert {r.field for r in solvers_std} == {"p"}

    _, solvers_bouss = collect_requirements([bouss])
    assert {r.field for r in solvers_bouss} == {"p_rgh"}
