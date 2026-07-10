# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Divergence-scheme variants and the staged ``system/fvSchemes``.

Div schemes are selected per call in the tests (``scheme="upwind"``): the
pybFoam workers pass the expanded scheme string to ``fvc.div``/``fvm.div``
(with named ``divT_<scheme>`` fvSchemes entries for the scalar-convection
flux, whose ``fvc.flux`` overload only takes a lookup key), and the neon
workers hand the tokens straight to the operator's ``read``. Everything else
(grad, laplacian, interpolation, snGrad) has a single variant and comes from
the one static ``fvSchemes`` below, which both backends read.
"""

from __future__ import annotations

DIV_SCHEMES: dict[str, str] = {
    "linear": "Gauss linear",
    "upwind": "Gauss upwind",
    "linearUpwind": "Gauss linearUpwind grad({field})",
}


def div_scheme(scheme: str, field: str) -> str:
    """Expand a scheme id to its full OpenFOAM scheme string for ``field``."""
    return DIV_SCHEMES[scheme].format(field=field)


FV_SCHEMES = """\
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSchemes;
}

ddtSchemes
{
    default         Euler;
}

gradSchemes
{
    default         none;
    grad(T)         Gauss linear;
    grad(U)         Gauss linear;
}

divSchemes
{
    default         none;
    div(phi,T)      Gauss linear;
    div(phi,U)      Gauss linear;
    divT_linear         Gauss linear;
    divT_upwind         Gauss upwind;
    divT_linearUpwind   Gauss linearUpwind grad(T);
}

laplacianSchemes
{
    default         none;
    laplacian(Gamma,T) Gauss linear uncorrected;
    laplacian(Gamma,U) Gauss linear uncorrected;
}

interpolationSchemes
{
    default         linear;
}

snGradSchemes
{
    default         uncorrected;
}

// ************************************************************************* //
"""
