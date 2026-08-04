# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Divergence-scheme variants shared by both workers.

Div schemes are selected per call in the tests (``scheme="upwind"``): the
pybFoam workers pass the expanded scheme string to ``fvc.div``/``fvm.div``
(with named ``divT_<scheme>`` fvSchemes entries for the scalar-convection
flux, whose ``fvc.flux`` overload only takes a lookup key), and the neon
workers hand the tokens straight to the operator's ``read``. Everything else
(grad, laplacian, interpolation, snGrad) has a single variant and comes from
the checked-in ``cases/common/system/fvSchemes``, which both backends read.
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


def div_scheme(scheme: str, field: str) -> str:
    """Expand a scheme id to its full OpenFOAM scheme string for ``field``."""
    return DIV_SCHEMES[scheme].format(field=field)
