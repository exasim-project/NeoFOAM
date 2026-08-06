# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Divergence- and gradient-scheme variants shared by both workers.

Div schemes are selected per call in the tests (``scheme="upwind"``): the
pybFoam workers pass the expanded scheme string to ``fvc.div``/``fvm.div``
(with named ``divT_<scheme>`` fvSchemes entries for the scalar-convection
flux, whose ``fvc.flux`` overload only takes a lookup key), and the neon
workers hand the tokens straight to the operator's ``read``.

Grad schemes are selected per call too, but only through fvSchemes on the
neon side: ``nfb.GradScheme`` takes a field *name* and resolves
``grad(<name>)`` from the dictionary, and a limiter coefficient only survives
as a numeric token when it comes from there (a hand-built TokenList carries
its words as strings, so ``cellLimited Gauss linear 0`` would silently fall
back to the default k=1). Each variant is therefore staged under its own
``grad(U_<scheme>)`` key, like ``divT_<scheme>`` above.

Everything else (laplacian, interpolation, snGrad) has a single variant and
comes from the checked-in ``cases/common/system/fvSchemes``, which both
backends read.
"""

from __future__ import annotations

DIV_SCHEMES: dict[str, str] = {
    "linear": "Gauss linear",
    "upwind": "Gauss upwind",
    "linearUpwind": "Gauss linearUpwind grad({field})",
    # The steady-state tutorials use the "bounded" convection wrapper, which subtracts
    # Sp(surfaceIntegrate(faceFlux), psi) from the inner scheme; the test flux is not
    # divergence free, so that term is what the parity check sees.
    "boundedUpwind": "bounded Gauss upwind",
}


def div_scheme(scheme: str | None, field: str) -> str | None:
    """Expand a scheme id to its full OpenFOAM scheme string for ``field``.

    ``None`` means "no scheme string": both backends then look the scheme up
    under ``div(phi,<field>)`` in the case's own ``system/fvSchemes`` — for
    neon that is the dictionary ``map_fv_schemes`` returned.
    """
    if scheme is None:
        return None
    return DIV_SCHEMES[scheme].format(field=field)


GRAD_SCHEMES: dict[str, str] = {
    "cellLimited": "cellLimited Gauss linear 1",
    # the same scheme with limiting switched off by its coefficient
    "cellLimitedOff": "cellLimited Gauss linear 0",
}


def grad_scheme(scheme: str) -> str:
    """Expand a grad-scheme id to its full OpenFOAM scheme string."""
    return GRAD_SCHEMES[scheme]


def grad_key(scheme: str, field: str) -> str:
    """The staged field name whose ``grad(<name>)`` fvSchemes entry holds ``scheme``."""
    return f"{field}_{scheme}"
