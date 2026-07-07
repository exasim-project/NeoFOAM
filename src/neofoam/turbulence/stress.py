# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Viscous-stress objects — the momentum equation's stress collaborator.

A ``viscousStress`` owns *how* the momentum stress term ``divDevReff(U)`` is
assembled from the molecular ``nu`` and eddy ``nut`` viscosities. Its lifecycle:

* ``update(ctx)`` runs **before** the PIMPLE loop — it *reads* ``nu``/``nut``
  (Context fields the viscosity and momentum-transport models own) and refreshes
  the effective viscosity ``nuEff``; it does not modify ``nu``/``nut``. The
  *operation* that calls ``update`` is owned by the momentum-transport model (e.g.
  ``laminar``'s ``update_viscous_stress``), not by this object.
* ``divDevReff(U)`` is called **by the momentum equation** and uses the
  refreshed ``nuEff``.

**The momentum-transport model defines which stress it uses** — a native model
registers its stress via its ``@build``; these classes are the reusable assembly.
A linear eddy-viscosity closure (laminar, kEpsilon, …) uses
:class:`LinearViscousStress`; the OpenFOAM fallback uses :class:`OpenFOAMStress`,
which delegates to the pybFoam model. pybFoam is imported lazily, so this module
stays importable without a built OpenFOAM environment.
"""

from typing import Any

__all__ = [
    "LinearViscousStress",
    "OpenFOAMStress",
]


def _add_viscosity(nu: Any, nut: Any) -> Any:
    """Effective viscosity ``nuEff = nu + nut``.

    When ``nut`` is a field (RAS/LES) the sum is a ``tmp`` field, materialised
    into a concrete ``volScalarField`` named ``nuEff`` (read twice by the
    assembly; a held ``tmp`` would be deallocated → use-after-free; the name
    matches the ``laplacian(nuEff,U)`` scheme). When both are ``dimensionedScalar``
    (laminar, ``nut = 0``) pybFoam binds no scalar+scalar add, so the sum is
    rebuilt as a (reusable) ``dimensionedScalar``.
    """
    import pybFoam

    try:
        total = nu + nut
    except TypeError:
        return pybFoam.dimensionedScalar(
            pybFoam.Word("nuEff"), nu.dimensions(), nu.value() + nut.value()
        )
    return pybFoam.volScalarField(pybFoam.Word("nuEff"), total)


def _materialize(nu: Any) -> Any:
    """``nuEff`` when there is no eddy viscosity — ``nu`` renamed to ``nuEff``.

    Renaming matters: the assembly looks the coefficient up under ``nuEff`` (the
    ``laplacian(nuEff,U)`` scheme). A field is wrapped as ``volScalarField``; a
    ``dimensionedScalar`` (laminar) is rebuilt (it cannot seed a ``volScalarField``
    without a mesh).
    """
    import pybFoam

    try:
        return pybFoam.volScalarField(pybFoam.Word("nuEff"), nu)
    except TypeError:
        return pybFoam.dimensionedScalar(
            pybFoam.Word("nuEff"), nu.dimensions(), nu.value()
        )


def _negate(nuEff: Any) -> Any:
    """Negated ``nuEff`` for the implicit ``-laplacian(nuEff, U)``.

    OpenFOAM negates the laplacian *matrix*; pybFoam binds no unary ``-`` on a
    matrix, so the coefficient is negated instead — kept under the name ``nuEff``
    so the case's ``laplacian(nuEff,U)`` scheme still resolves (the operator is
    linear in the coefficient, so this is identical). A ``dimensionedScalar`` is
    rebuilt; a field is negated then materialised under the ``nuEff`` name.
    """
    import pybFoam

    try:
        negated = -nuEff
    except TypeError:
        return pybFoam.dimensionedScalar(
            pybFoam.Word("nuEff"), nuEff.dimensions(), -nuEff.value()
        )
    return pybFoam.volScalarField(pybFoam.Word("nuEff"), negated)


class LinearViscousStress:
    """Boussinesq (linear eddy-viscosity) stress: ``nuEff = nu + nut``.

    ``divDevReff(U) = -laplacian(nuEff, U) - div(nuEff*dev2(T(grad U)))``.
    """

    stress_kind = "linear"

    def __init__(self) -> None:
        self._nuEff: Any = None

    def update(self, ctx: Any) -> None:
        """Refresh ``nuEff`` from the Context's ``nu`` and ``nut`` (read-only).

        Called by the momentum predictor right before ``divDevReff`` consumes it.
        ``nut`` is optional: a laminar momentum-transport model registers none, so an
        absent ``nut`` means no eddy viscosity (``nuEff = nu``); a RAS/LES model
        registers ``nut`` and ``nuEff = nu + nut``.
        """
        nut = ctx.fields.get("nut")
        if nut is None:
            self._nuEff = _materialize(ctx.fields["nu"])
        else:
            self._nuEff = _add_viscosity(ctx.fields["nu"], nut)

    def divDevReff(self, U: Any) -> Any:
        """Momentum stress term assembled from the refreshed ``nuEff``.

        The tensor handed to ``div`` is stamped with OpenFOAM's canonical key
        ``(nuEff*dev2(T(grad(U))))`` so the case's ``divScheme`` resolves it.
        """
        import pybFoam
        from pybFoam import fvc, fvm

        nuEff = self._nuEff
        scheme_name = f"(nuEff*dev2(T(grad({U.name()}))))"
        stress = pybFoam.volTensorField(
            pybFoam.Word(scheme_name), pybFoam.dev2(pybFoam.T(fvc.grad(U))) * nuEff
        )
        return fvm.laplacian(_negate(nuEff), U) - fvc.div(stress)


class OpenFOAMStress:
    """Stress delegated to the OpenFOAM fallback model (it assembles its own).

    Holds the momentum-transport adapter (which owns the pybFoam model); does not
    capture it in an operation closure, so it stays out of the execution-graph
    reference cycle (see [[project_pybfoam_op_closure_cycle]]).
    """

    stress_kind = "openfoam"

    def __init__(self, turbulence: Any) -> None:
        self._turbulence = turbulence

    def update(self, ctx: Any) -> None:
        """No-op — the OpenFOAM model owns its eddy viscosity and stress."""
        return None

    def divDevReff(self, U: Any) -> Any:
        return self._turbulence.divDevReff(U)
