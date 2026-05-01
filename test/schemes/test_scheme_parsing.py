# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Unit tests for OpenFOAM scheme string parsing and round-trip serialization."""

import pytest
from pydantic import TypeAdapter, ValidationError

from neofoam.schemes import (
    # Interpolation
    InterpolationScheme,
    Linear,
    Upwind,
    LinearUpwind,
    LimitedLinear,
    VanLeer,
    Minmod,
    SuperBee,
    MUSCL,
    QUICK,
    # SnGrad
    SnGradScheme,
    Corrected,
    Uncorrected,
    Orthogonal,
    LimitedSnGrad,
    # DDT
    DdtScheme,
    Euler,
    Backward,
    SteadyState,
    LocalEuler,
    CrankNicolson,
    # Grad
    GradScheme,
    GaussGrad,
    LeastSquaresGrad,
    # Div
    DivScheme,
    NoneDiv,
    GaussDiv,
    BoundedGaussDiv,
    # Laplacian
    LaplacianScheme,
    GaussLaplacian,
)


# ============================================================================
# Interpolation schemes
# ============================================================================


@pytest.mark.parametrize(
    "raw, expected_type",
    [
        ("linear", Linear),
        ("upwind", Upwind),
        ("vanLeer", VanLeer),
        ("Minmod", Minmod),
        ("SuperBee", SuperBee),
        ("MUSCL", MUSCL),
        ("QUICK", QUICK),
    ],
)
def test_interpolation_simple(raw, expected_type):
    s = TypeAdapter(InterpolationScheme).validate_python(raw)
    assert isinstance(s, expected_type)


def test_interpolation_linear_upwind():
    s = TypeAdapter(InterpolationScheme).validate_python("linearUpwind grad(U)")
    assert isinstance(s, LinearUpwind)
    assert s.grad_field == "grad(U)"


def test_interpolation_linear_upwind_missing_grad_field():
    with pytest.raises(ValidationError):
        TypeAdapter(InterpolationScheme).validate_python("linearUpwind")


def test_interpolation_limited_linear():
    s = TypeAdapter(InterpolationScheme).validate_python("limitedLinear 1")
    assert isinstance(s, LimitedLinear)
    assert s.coefficient == 1.0


def test_interpolation_limited_linear_rejects_out_of_range():
    with pytest.raises(ValidationError):
        TypeAdapter(InterpolationScheme).validate_python("limitedLinear 2.0")


def test_interpolation_unknown_rejected():
    with pytest.raises(ValidationError):
        TypeAdapter(InterpolationScheme).validate_python("invalidScheme")


# ============================================================================
# Surface-normal gradient schemes
# ============================================================================


@pytest.mark.parametrize(
    "raw, expected_type",
    [
        ("corrected", Corrected),
        ("uncorrected", Uncorrected),
        ("orthogonal", Orthogonal),
    ],
)
def test_sn_grad_simple(raw, expected_type):
    s = TypeAdapter(SnGradScheme).validate_python(raw)
    assert isinstance(s, expected_type)


def test_sn_grad_limited():
    s = TypeAdapter(SnGradScheme).validate_python("limited 0.5")
    assert isinstance(s, LimitedSnGrad)
    assert s.coefficient == 0.5


def test_sn_grad_limited_rejects_zero():
    with pytest.raises(ValidationError):
        TypeAdapter(SnGradScheme).validate_python("limited 0")


def test_sn_grad_unknown_rejected():
    with pytest.raises(ValidationError):
        TypeAdapter(SnGradScheme).validate_python("invalidSnGrad")


# ============================================================================
# DDT schemes
# ============================================================================


@pytest.mark.parametrize(
    "raw, expected_type",
    [
        ("Euler", Euler),
        ("backward", Backward),
        ("steadyState", SteadyState),
        ("localEuler", LocalEuler),
    ],
)
def test_ddt_simple(raw, expected_type):
    s = TypeAdapter(DdtScheme).validate_python(raw)
    assert isinstance(s, expected_type)


def test_ddt_crank_nicolson():
    s = TypeAdapter(DdtScheme).validate_python("CrankNicolson 0.9")
    assert isinstance(s, CrankNicolson)
    assert s.coefficient == 0.9


def test_ddt_crank_nicolson_rejects_out_of_range():
    with pytest.raises(ValidationError):
        TypeAdapter(DdtScheme).validate_python("CrankNicolson 1.5")


def test_ddt_unknown_rejected():
    with pytest.raises(ValidationError):
        TypeAdapter(DdtScheme).validate_python("InvalidScheme")


# ============================================================================
# Gradient schemes
# ============================================================================


def test_grad_gauss_linear():
    s = TypeAdapter(GradScheme).validate_python("Gauss linear")
    assert isinstance(s, GaussGrad)
    assert isinstance(s.interpolation, Linear)


def test_grad_least_squares():
    s = TypeAdapter(GradScheme).validate_python("pointCellsLeastSquares")
    assert isinstance(s, LeastSquaresGrad)


def test_grad_gauss_linear_upwind():
    s = TypeAdapter(GradScheme).validate_python("Gauss linearUpwind grad(U)")
    assert isinstance(s, GaussGrad)
    assert isinstance(s.interpolation, LinearUpwind)
    assert s.interpolation.grad_field == "grad(U)"


def test_grad_unknown_rejected():
    with pytest.raises(ValidationError):
        TypeAdapter(GradScheme).validate_python("invalidGrad")


# ============================================================================
# Divergence schemes
# ============================================================================


def test_div_none():
    s = TypeAdapter(DivScheme).validate_python("none")
    assert isinstance(s, NoneDiv)


def test_div_gauss_linear():
    s = TypeAdapter(DivScheme).validate_python("Gauss linear")
    assert isinstance(s, GaussDiv)
    assert isinstance(s.interpolation, Linear)


def test_div_gauss_linear_upwind():
    s = TypeAdapter(DivScheme).validate_python("Gauss linearUpwind grad(U)")
    assert isinstance(s, GaussDiv)
    assert isinstance(s.interpolation, LinearUpwind)
    assert s.interpolation.grad_field == "grad(U)"


def test_div_bounded_gauss():
    s = TypeAdapter(DivScheme).validate_python("bounded Gauss limitedLinear 1")
    assert isinstance(s, BoundedGaussDiv)
    assert isinstance(s.interpolation, LimitedLinear)
    assert s.interpolation.coefficient == 1.0


def test_div_unknown_rejected():
    with pytest.raises(ValidationError):
        TypeAdapter(DivScheme).validate_python("invalidDiv")


# ============================================================================
# Laplacian schemes
# ============================================================================


def test_laplacian_gauss_linear_corrected():
    s = TypeAdapter(LaplacianScheme).validate_python("Gauss linear corrected")
    assert isinstance(s, GaussLaplacian)
    assert isinstance(s.interpolation, Linear)
    assert isinstance(s.sn_grad, Corrected)


def test_laplacian_gauss_linear_limited():
    s = TypeAdapter(LaplacianScheme).validate_python("Gauss linear limited 0.5")
    assert isinstance(s, GaussLaplacian)
    assert isinstance(s.interpolation, Linear)
    assert isinstance(s.sn_grad, LimitedSnGrad)
    assert s.sn_grad.coefficient == 0.5


def test_laplacian_unknown_type_rejected():
    with pytest.raises(ValidationError):
        TypeAdapter(LaplacianScheme).validate_python("invalid linear corrected")


# ============================================================================
# Round-trip serialization
# ============================================================================


@pytest.mark.parametrize(
    "scheme_type, raw_string",
    [
        # DDT
        (DdtScheme, "Euler"),
        (DdtScheme, "backward"),
        (DdtScheme, "steadyState"),
        (DdtScheme, "localEuler"),
        (DdtScheme, "CrankNicolson 0.9"),
        # Interpolation
        (InterpolationScheme, "linear"),
        (InterpolationScheme, "upwind"),
        (InterpolationScheme, "vanLeer"),
        (InterpolationScheme, "linearUpwind grad(U)"),
        (InterpolationScheme, "limitedLinear 1"),
        # SnGrad
        (SnGradScheme, "corrected"),
        (SnGradScheme, "uncorrected"),
        (SnGradScheme, "orthogonal"),
        (SnGradScheme, "limited 0.5"),
        # Grad
        (GradScheme, "Gauss linear"),
        (GradScheme, "pointCellsLeastSquares"),
        # Div
        (DivScheme, "none"),
        (DivScheme, "Gauss linear"),
        (DivScheme, "Gauss linearUpwind grad(U)"),
        (DivScheme, "bounded Gauss linearUpwind grad(U)"),
        (DivScheme, "bounded Gauss limitedLinear 1"),
        # Laplacian
        (LaplacianScheme, "Gauss linear corrected"),
        (LaplacianScheme, "Gauss linear limited 0.5"),
    ],
)
def test_round_trip(scheme_type, raw_string):
    adapter = TypeAdapter(scheme_type)
    parsed = adapter.validate_python(raw_string)
    serialized = parsed.model_dump(mode="python")
    assert serialized == raw_string
