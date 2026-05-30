# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Momentum-stress computers — how a model assembles ``divDevReff(U)``.

A momentum-transport model **defines how** its contribution to the momentum
equation, the deviatoric-stress divergence ``divDevReff(U)``, is built. This
module provides the shared **linear eddy-viscosity** (Boussinesq) assembly that
every linear closure reuses by registering it; non-linear / viscoelastic models
register a different function, so the choice stays with the model.

``divDevReff(U, nu, nut) = -laplacian(nuEff, U) - div(nuEff*dev2(T(grad U)))``
with ``nuEff = nu + nut`` — ``nu`` (molecular) and ``nut`` (eddy) are the Context
fields the viscosity and turbulence models own. Assembled from *visible* pybFoam
operators (imported lazily, so this module stays importable without OpenFOAM).
"""

from typing import Any

__all__ = ["linear_viscous_stress"]


def _add_viscosity(nu: Any, nut: Any) -> Any:
    """Effective viscosity ``nuEff = nu + nut``.

    When ``nut`` is a field (RAS/LES) the sum is a ``tmp`` field; it is
    materialised into a concrete ``volScalarField`` named ``nuEff`` because the
    assembly reads it twice (a held ``tmp`` would be deallocated → use-after-free)
    and the name matches the ``laplacian(nuEff,U)`` scheme. When both are
    ``dimensionedScalar`` (laminar, ``nut = 0``) pybFoam binds no scalar+scalar
    add, so the sum is rebuilt as a (reusable) ``dimensionedScalar``.
    """
    import pybFoam

    try:
        total = nu + nut
    except TypeError:
        return pybFoam.dimensionedScalar(
            pybFoam.Word("nuEff"), nu.dimensions(), nu.value() + nut.value()
        )
    return pybFoam.volScalarField(pybFoam.Word("nuEff"), total)


def _negate(nuEff: Any) -> Any:
    """Negated ``nuEff`` for the implicit ``-laplacian(nuEff, U)``.

    OpenFOAM negates the laplacian *matrix*; pybFoam binds no unary ``-`` on a
    matrix, so the coefficient is negated instead. The negated coefficient keeps
    the name ``nuEff`` so the case's ``laplacian(nuEff,U)`` scheme still resolves
    (negating the coefficient, not the matrix, is mathematically identical — the
    operator is linear in it). A ``dimensionedScalar`` binds no unary ``-`` so it
    is rebuilt; a field is negated then materialised under the ``nuEff`` name.
    """
    import pybFoam

    try:
        negated = -nuEff
    except TypeError:
        return pybFoam.dimensionedScalar(
            pybFoam.Word("nuEff"), nuEff.dimensions(), -nuEff.value()
        )
    return pybFoam.volScalarField(pybFoam.Word("nuEff"), negated)


def linear_viscous_stress(U: Any, nu: Any, nut: Any) -> Any:
    """Boussinesq (linear eddy-viscosity) momentum stress term.

    ``-laplacian(nuEff, U) - div(nuEff*dev2(T(grad U)))``. The tensor handed to
    ``div`` is stamped with OpenFOAM's canonical key ``(nuEff*dev2(T(grad(U))))``
    so the case's ``divScheme`` resolves it. Operator ordering follows pybFoam's
    bound operators (``matrix - field``; ``dev2(...) * nuEff``).
    """
    import pybFoam
    from pybFoam import fvc, fvm

    nuEff = _add_viscosity(nu, nut)
    scheme_name = f"(nuEff*dev2(T(grad({U.name()}))))"
    stress = pybFoam.volTensorField(
        pybFoam.Word(scheme_name), pybFoam.dev2(pybFoam.T(fvc.grad(U))) * nuEff
    )
    return fvm.laplacian(_negate(nuEff), U) - fvc.div(stress)
