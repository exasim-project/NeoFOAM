# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Tests for the viscous-stress model (``divDevReff`` assembly + dispatch).

These run without pybFoam: the finite-volume operator surface is injected as a
symbolic recorder (:class:`Sym`), so the assembled stress *expression* and the
``nuEff = nu + nut`` combination can be asserted directly. Only the optional
``default_operators`` smoke test touches pybFoam.
"""

from types import SimpleNamespace
from typing import Any

from neofoam.turbulence.stress import (
    LinearViscoStress,
    StressOperators,
    ViscoStress,
    default_operators,
)


class Sym:
    """Symbolic value recording the operator expression applied to it."""

    def __init__(self, expr: str) -> None:
        self.expr = expr

    def __add__(self, other: object) -> "Sym":
        return Sym(f"({self.expr} + {_expr(other)})")

    def __sub__(self, other: object) -> "Sym":
        return Sym(f"({self.expr} - {_expr(other)})")

    def __mul__(self, other: object) -> "Sym":
        return Sym(f"({self.expr} * {_expr(other)})")

    def __neg__(self) -> "Sym":
        return Sym(f"(-{self.expr})")

    def __repr__(self) -> str:
        return self.expr


def _expr(value: object) -> str:
    return value.expr if isinstance(value, Sym) else str(value)


def _symbolic_operators() -> StressOperators:
    return StressOperators(
        grad=lambda u: Sym(f"grad({_expr(u)})"),
        div=lambda x: Sym(f"div({_expr(x)})"),
        laplacian=lambda g, u: Sym(f"laplacian({_expr(g)}, {_expr(u)})"),
        dev2=lambda x: Sym(f"dev2({_expr(x)})"),
        transpose=lambda x: Sym(f"T({_expr(x)})"),
        neg=lambda x: Sym(f"(-{_expr(x)})"),
        name_of=lambda f: _expr(f),
        name_tensor=lambda name, field: Sym(name),
    )


def _viscosity(nu: Any) -> Any:
    return SimpleNamespace(nu=lambda: nu)


def _turbulence(nut: Any) -> Any:
    return SimpleNamespace(nut=lambda: nut)


def test_linear_stress_satisfies_protocol() -> None:
    stress = LinearViscoStress(_viscosity(Sym("nu")), _turbulence(0.0))
    assert isinstance(stress, ViscoStress)


def test_nuEff_is_molecular_viscosity_when_laminar() -> None:
    """nut == 0 (laminar) ⇒ nuEff is the molecular field itself, untouched."""
    nu = Sym("nu")
    stress = LinearViscoStress(_viscosity(nu), _turbulence(0.0))
    assert stress.nuEff() is nu


def test_nuEff_combines_viscosity_and_turbulence() -> None:
    """A real eddy-viscosity model (nut field) ⇒ nuEff = nu + nut."""
    stress = LinearViscoStress(_viscosity(Sym("nu")), _turbulence(Sym("nut")))
    assert repr(stress.nuEff()) == "(nu + nut)"


def test_divDevReff_assembles_linear_stress_term() -> None:
    """divDevReff = -div(<canonical>) - laplacian(nuEff, U); the div argument
    carries OpenFOAM's canonical scheme key so the case's divScheme resolves."""
    stress = LinearViscoStress(
        _viscosity(Sym("nu")), _turbulence(0.0), operators=_symbolic_operators()
    )

    term = repr(stress.divDevReff(Sym("U")))

    assert term == "(laplacian((-nu), U) - div((nuEff*dev2(T(grad(U))))))"


def test_divDevReff_uses_combined_nuEff_in_laplacian_for_eddy_viscosity() -> None:
    """With a non-zero nut the implicit term is built from nu + nut; the div key
    stays the canonical nuEff form regardless of closure."""
    stress = LinearViscoStress(
        _viscosity(Sym("nu")), _turbulence(Sym("nut")), operators=_symbolic_operators()
    )

    term = repr(stress.divDevReff(Sym("U")))

    assert "laplacian((-(nu + nut)), U)" in term
    assert "div((nuEff*dev2(T(grad(U)))))" in term


def test_divDevReff_invokes_the_full_operator_chain() -> None:
    """grad → transpose → dev2 → (× nuEff) → name → laplacian(neg) → div."""
    seen: list[str] = []

    def record(tag: str, value: Any = None) -> Any:
        seen.append(tag)
        return Sym(tag)

    ops = StressOperators(
        grad=lambda u: record("grad"),
        transpose=lambda x: record("transpose"),
        dev2=lambda x: record("dev2"),
        div=lambda x: record("div"),
        laplacian=lambda g, u: record("laplacian"),
        neg=lambda x: record("neg"),
        name_of=lambda f: "U",
        name_tensor=lambda name, field: record("name_tensor"),
    )
    LinearViscoStress(
        _viscosity(Sym("nu")), _turbulence(0.0), operators=ops
    ).divDevReff(Sym("U"))

    assert seen == [
        "grad",
        "transpose",
        "dev2",
        "name_tensor",
        "neg",
        "laplacian",
        "div",
    ]


def test_stress_reads_nu_from_viscosity_model() -> None:
    """The stress model is given access to the viscosity model for nu."""
    calls = {"nu": 0}

    def nu() -> Any:
        calls["nu"] += 1
        return Sym("nu")

    viscosity = SimpleNamespace(nu=nu)
    stress = LinearViscoStress(
        viscosity, _turbulence(0.0), operators=_symbolic_operators()
    )

    stress.divDevReff(Sym("U"))

    assert calls["nu"] >= 1


def test_default_operators_are_pybfoam_backed() -> None:
    """Smoke test: the lazy default operator surface resolves (needs pybFoam)."""
    import pytest

    pytest.importorskip("pybFoam")
    ops = default_operators()
    assert isinstance(ops, StressOperators)
    assert all(
        callable(fn)
        for fn in (ops.grad, ops.div, ops.laplacian, ops.dev2, ops.transpose)
    )
