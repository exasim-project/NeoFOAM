# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Shared build + types for the alpha-advection family.

Every advection scheme (MULES, isoAdvector, …) owns the *same* set of fields —
``phi``, ``mixture``, ``alpha1``, ``alpha2``, ``rho``, ``rhoPhi`` — and only its
per-step ``alpha_advection`` operation differs. That common initialisation lives
here so each family member's ``@build`` is a one-liner returning these steps
(isoAdvector appends its own ``advector`` model on top).
"""

from typing import Any, Protocol

import pybFoam as pyf
import pybFoam.multiphase as multiphase
from pybFoam import fvc, surfaceScalarField, volScalarField

from neofoam.framework.initialization import field, model


class MixtureProtocol(Protocol):
    """Static-analysis view of ``immiscibleIncompressibleTwoPhaseMixture``.

    The single protocol for every consumer of the mixture model: the
    alpha-advection schemes use ``cAlpha``/``nHatf``, the PIMPLE momentum and
    continuity operations use ``surfaceTensionForce``.
    """

    def alpha1(self) -> volScalarField: ...
    def alpha2(self) -> volScalarField: ...
    def rho1(self) -> Any: ...
    def rho2(self) -> Any: ...
    def cAlpha(self) -> float: ...
    def nHatf(self) -> surfaceScalarField: ...
    def surfaceTensionForce(self) -> surfaceScalarField: ...
    def correct(self) -> None: ...


def shared_field_build_steps() -> list[Any]:
    """InitSteps for the fields every advection scheme owns.

    phi (face flux from U) → mixture → alpha1/alpha2 → rho → rhoPhi.
    """

    def create_phi(context: dict[str, Any]) -> surfaceScalarField:
        """Create face flux phi from U."""
        return pyf.createPhi(context["fields.U"])

    def create_mixture(context: dict[str, Any]) -> Any:
        """Create immiscibleIncompressibleTwoPhaseMixture and register alpha flux."""
        U = context["fields.U"]
        phi = context["fields.phi"]
        mesh = context["mesh"]
        mixture = multiphase.immiscibleIncompressibleTwoPhaseMixture(U, phi)
        # alpha.water face flux required for the alpha solve.
        mesh.setFluxRequired(mixture.alpha1().name())
        return mixture

    def create_alpha1(context: dict[str, Any]) -> volScalarField:
        return context["models.mixture"].alpha1()  # type: ignore[no-any-return]

    def create_alpha2(context: dict[str, Any]) -> volScalarField:
        return context["models.mixture"].alpha2()  # type: ignore[no-any-return]

    def create_rho(context: dict[str, Any]) -> volScalarField:
        mixture = context["models.mixture"]
        alpha1 = context["fields.alpha1"]
        alpha2 = context["fields.alpha2"]
        rho1 = mixture.rho1()
        rho2 = mixture.rho2()
        rho = volScalarField(pyf.Word("rho"), alpha1 * rho1 + alpha2 * rho2)
        # Store old-time so that fvm.ddt(rho, U) in momentum has a valid old value.
        rho.oldTime()
        return rho

    def create_rho_phi(context: dict[str, Any]) -> surfaceScalarField:
        rho = context["fields.rho"]
        phi = context["fields.phi"]
        return surfaceScalarField(pyf.Word("rhoPhi"), fvc.interpolate(rho) * phi)

    return [
        field("phi", create_phi, depends_on=["fields.U"]),
        model("mixture", create_mixture, depends_on=["fields.U", "fields.phi", "mesh"]),
        field("alpha1", create_alpha1, depends_on=["models.mixture"]),
        field("alpha2", create_alpha2, depends_on=["models.mixture"]),
        field(
            "rho",
            create_rho,
            depends_on=["models.mixture", "fields.alpha1", "fields.alpha2"],
        ),
        field("rhoPhi", create_rho_phi, depends_on=["fields.rho", "fields.phi"]),
    ]
