# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Parsing and write-back of the ``neofoam.foam.schemes`` unions.

The unions are the value types ``fvSchemes`` fields carry, so a scheme spelling
they reject makes a valid OpenFOAM case unloadable. Two regressions are pinned
here:

- ``limitedSnGrad`` accepts an optional ``corrected`` sub-scheme token
  (``limited corrected 0.33`` == ``limited 0.33``); the two forms are the same
  scheme, so write-back normalises to the terse one. Any *other* sub-scheme is a
  validation error, matching NeoN's ``LimitedCorrected::readLimitCoeff``.
- ``gradSchemes`` accepts ``cellLimited <inner grad scheme> <k>``, the most
  common gradient spec in the OpenFOAM tutorials after plain ``Gauss linear``.

The interpolation/ddt/snGrad names NeoN cannot construct (``vanLeer`` … ) stay
accepted on purpose: these models are shared with the pybFoam solver families,
where OpenFOAM does the discretisation and every one of them is valid. The
regression guard at the bottom keeps them that way.

TypeAdapter is the lowest level that proves the behaviour — the specs under test
are single dictionary *values*, not file content, so no case directory is needed.
"""

from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError

from neofoam.foam.schemes import (
    CellLimitedGrad,
    DdtScheme,
    DivScheme,
    GaussGrad,
    GradScheme,
    InterpolationScheme,
    LaplacianScheme,
    LimitedSnGrad,
    Linear,
    SnGradScheme,
)

DDT = TypeAdapter(DdtScheme)
DIV = TypeAdapter(DivScheme)
GRAD = TypeAdapter(GradScheme)
INTERPOLATION = TypeAdapter(InterpolationScheme)
LAPLACIAN = TypeAdapter(LaplacianScheme)
SN_GRAD = TypeAdapter(SnGradScheme)


# ---------------------------------------------------------------------------
# limitedSnGrad — the optional ``corrected`` sub-scheme
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spec", ["limited 0.33", "limited corrected 0.33"])
def test_limited_sn_grad_accepts_both_spellings(spec: str) -> None:
    assert SN_GRAD.validate_python(spec) == LimitedSnGrad(coefficient=0.33)


@pytest.mark.parametrize("spec", ["limited 0.33", "limited corrected 0.33"])
def test_limited_sn_grad_serializes_to_the_terse_form(spec: str) -> None:
    assert SN_GRAD.dump_python(SN_GRAD.validate_python(spec)) == "limited 0.33"


@pytest.mark.parametrize("spec", ["limited uncorrected 0.33", "limited orthogonal 0.33"])
def test_limited_sn_grad_rejects_an_unsupported_sub_scheme(spec: str) -> None:
    with pytest.raises(ValidationError):
        SN_GRAD.validate_python(spec)


def test_laplacian_accepts_the_verbose_limited_sn_grad() -> None:
    """The verbose form reaches the snGrad parser through laplacianSchemes too."""
    scheme = LAPLACIAN.validate_python("Gauss linear limited corrected 0.33")
    assert LAPLACIAN.dump_python(scheme) == "Gauss linear limited 0.33"


# ---------------------------------------------------------------------------
# cellLimited gradient
# ---------------------------------------------------------------------------


def test_cell_limited_grad_parses_its_inner_scheme_as_a_model() -> None:
    scheme = GRAD.validate_python("cellLimited Gauss linear 1")
    assert scheme == CellLimitedGrad(inner_scheme=GaussGrad(interpolation=Linear()), coefficient=1)


@pytest.mark.parametrize(
    "spec",
    [
        "cellLimited Gauss linear 1",
        "cellLimited Gauss linearUpwind grad(U) 1",
        "cellLimited Gauss linear 0.5",
        "cellLimited pointCellsLeastSquares 1",
    ],
)
def test_cell_limited_grad_round_trips(spec: str) -> None:
    assert GRAD.dump_python(GRAD.validate_python(spec)) == spec


@pytest.mark.parametrize("spec", ["cellLimited Gauss linear 2", "cellLimited Gauss linear -0.5"])
def test_cell_limited_grad_rejects_a_coefficient_outside_zero_to_one(spec: str) -> None:
    with pytest.raises(ValidationError):
        GRAD.validate_python(spec)


@pytest.mark.parametrize("spec", ["cellLimited 1", "cellLimited Gauss linear"])
def test_cell_limited_grad_rejects_a_missing_inner_scheme_or_coefficient(spec: str) -> None:
    with pytest.raises(ValidationError):
        GRAD.validate_python(spec)


# ---------------------------------------------------------------------------
# Regression guard — schemes OpenFOAM supports but NeoN cannot construct
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "adapter, spec",
    [
        (INTERPOLATION, "vanLeer"),
        (INTERPOLATION, "Minmod"),
        (INTERPOLATION, "SuperBee"),
        (INTERPOLATION, "MUSCL"),
        (INTERPOLATION, "QUICK"),
        (DIV, "Gauss vanLeer"),
        (SN_GRAD, "orthogonal"),
        (DDT, "localEuler"),
        (DDT, "CrankNicolson 0.9"),
    ],
)
def test_openfoam_only_schemes_stay_accepted(adapter: TypeAdapter[Any], spec: str) -> None:
    """pybFoam cases discretise in OpenFOAM, where these are all valid."""
    assert adapter.dump_python(adapter.validate_python(spec)) == spec


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
