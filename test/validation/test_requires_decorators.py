# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""
Plan 04 — fvSchemes.add / fvSolution.add decorators.

Target API:
    from neofoam.foam import fvSchemes, fvSolution

    @fvSchemes.add(ddt="default", div="div(phi,U)")
    @fvSolution.add("p", "U")

File: src/neofoam/foam/__init__.py (exports fvSchemes, fvSolution)
"""

from neofoam.foam import fvSchemes, fvSolution
from neofoam.foam.requirements import SchemeRequirement, SolverRequirement


def test_fvSchemes_add_kwargs() -> None:
    @fvSchemes.add(ddt="default", div="div(phi,U)")
    def my_op() -> None:
        pass

    reqs = my_op._scheme_requirements  # type: ignore[attr-defined]
    assert SchemeRequirement("ddtSchemes", "default") in reqs
    assert SchemeRequirement("divSchemes", "div(phi,U)") in reqs


def test_fvSolution_add() -> None:
    @fvSolution.add("p", "U")
    def my_op() -> None:
        pass

    assert my_op._solver_requirements == [
        SolverRequirement("p"),
        SolverRequirement("U"),
    ]  # type: ignore[attr-defined]


def test_decorators_stack() -> None:
    @fvSchemes.add(ddt="default")
    @fvSolution.add("p")
    def my_op() -> None:
        pass

    assert len(my_op._scheme_requirements) == 1  # type: ignore[attr-defined]
    assert len(my_op._solver_requirements) == 1  # type: ignore[attr-defined]


def test_momentum_realistic() -> None:
    @fvSchemes.add(ddt="default", div="div(phi,U)", grad="default")
    @fvSolution.add("U")
    def momentum() -> None:
        pass

    schemes = momentum._scheme_requirements  # type: ignore[attr-defined]
    assert SchemeRequirement("ddtSchemes", "default") in schemes
    assert SchemeRequirement("divSchemes", "div(phi,U)") in schemes
    assert SchemeRequirement("gradSchemes", "default") in schemes


def test_steady_state_no_ddt() -> None:
    @fvSchemes.add(div="div(phi,U)", grad="default")
    def momentum() -> None:
        pass

    sections = {r.section for r in momentum._scheme_requirements}  # type: ignore[attr-defined]
    assert "ddtSchemes" not in sections


def test_wallDist_passes_through() -> None:
    @fvSchemes.add(div="div(phi,nuTilda)", wallDist="method")
    def turbulence() -> None:
        pass

    schemes = turbulence._scheme_requirements  # type: ignore[attr-defined]
    assert SchemeRequirement("wallDist", "method") in schemes
