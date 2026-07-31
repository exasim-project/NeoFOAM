# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Shared build + types for the alpha-advection family.

Every advection scheme (MULES, isoAdvector, …) owns the *same* set of fields —
``phi``, ``mixture``, ``alpha1``, ``alpha2``, ``rho``, ``rhoPhi`` — and only its
per-step ``alpha_advection`` operation differs. That common initialisation lives
here so each family member's ``@build`` is a one-liner returning these steps
(isoAdvector appends its own ``advector`` model on top). The ``nAlphaSubCycles``
machinery of ``alphaEqnSubCycle.H`` is shared for the same reason: interFoam and
interIsoFoam sub-cycle the alpha equation identically.
"""

from collections.abc import Iterator
from contextlib import contextmanager
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


@contextmanager
def alpha_sub_cycle(alpha1: volScalarField, n_alpha_sub_cycles: int) -> Iterator[Any]:
    """``subCycle<volScalarField>`` for the alpha equation; yields the sub-cycled ``Time``.

    Wrap the ``nAlphaSubCycles > 1`` loop of ``alphaEqnSubCycle.H`` in this —
    MULES and isoAdvector sub-cycle identically, only the pass inside differs.
    Call ``increment()`` on the yielded ``Foam::Time`` once per sub-step and run
    one alpha pass after each; on exit the real time state and ``alpha1``'s real
    old-time value are back, so the momentum/pressure steps that follow see an
    untouched time step. Read anything that must refer to the *whole* step (the
    total ``deltaT``) before entering.

    Reproduces both halves of ``Foam::subCycle``:

    * ``subCycleTime``: ``Time.subCycle(n)`` rewinds one ``deltaT``, scales the
      time index by ``n`` and divides ``deltaT`` by ``n``; ``endSubCycle()``
      restores the step's real time state.
    * ``subCycleField``: ``alpha1``'s old-time value belongs to the *real* time
      step, so it is copied out first and put back at the end, and both fields'
      time indices are moved onto the sub-cycle clock so ``oldTime()`` rolls
      exactly once per sub-step (and, on the way out, once at the next real step).

    Advancing ``Time`` — rather than the equivalent trick of scaling ``phi`` by
    ``1/n``, which the arithmetic of one alpha pass alone cannot tell apart — is
    what makes time-dependent boundary conditions correct: a ``waveAlpha`` /
    ``waveVelocity`` inlet re-evaluates once per sub-step, at the sub-step's own
    time, because its ``updateCoeffs`` is keyed on the time index.

    Example::

        with alpha_sub_cycle(alpha1, 3) as runtime:
            for _ in range(3):
                runtime.increment()
                ...  # one alpha pass over deltaT/3
    """
    runtime = alpha1.mesh().time()

    # First access rolls alpha1 into its old-time slot for this time step; take
    # the copy subCycleField keeps so the real old time can be restored after.
    alpha1_0 = volScalarField(pyf.Word("alpha1_0_"), 1.0 * alpha1.oldTime())

    runtime.subCycle(n_alpha_sub_cycles)
    # subCycleField::updateTimeIndex(): one ahead of the sub-cycle clock, so the
    # first sub-step keeps the real old-time value and every later one rolls.
    alpha1.setTimeIndex(runtime.timeIndex() + 1)
    alpha1.oldTime().setTimeIndex(runtime.timeIndex() + 1)
    try:
        yield runtime
    finally:
        runtime.endSubCycle()
        # subCycleField's destructor: real old time back, time index back on the
        # global clock so the next real step rolls alpha1 exactly once.
        alpha1.oldTime().assign(alpha1_0)
        alpha1.setTimeIndex(runtime.timeIndex())
        alpha1.oldTime().setTimeIndex(runtime.timeIndex())


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

    def create_alpha_phi_un(context: dict[str, Any]) -> surfaceScalarField:
        """Zero-init the MULES compressed flux (``createAlphaFluxes.H``).

        Registered under its final name up front — not just computed later —
        so it stays in the objectRegistry (and lookup-able by e.g. the
        ``scalarTransport`` functionObject's ``phase`` option) for the whole
        run; ``alpha_eqn`` (MULES) assigns into it in place every alpha solve.
        """
        phi = context["fields.phi"]
        return surfaceScalarField(pyf.Word("alphaPhiUn"), 0.0 * phi)

    def create_alpha_phi10(context: dict[str, Any]) -> surfaceScalarField:
        """Seed the MULES phase flux (``createAlphaFluxes.H``).

        Created once for the whole run rather than per alpha solve, because the
        Crank-Nicolson tail of ``alphaEqn.H`` converts the off-centred flux back
        to an end-of-time-step one using ``alphaPhi10.oldTime()`` — a per-call
        field would have no old time to roll. Native additionally reads and
        writes it as ``alphaPhi0.<phase>`` so a restart can pick the previous
        step's flux back up; that restart path is not implemented here.
        """
        phi = context["fields.phi"]
        alpha1 = context["fields.alpha1"]
        return surfaceScalarField(pyf.Word("alphaPhi10"), phi * fvc.interpolate(alpha1))

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
        field("alphaPhiUn", create_alpha_phi_un, depends_on=["fields.phi"]),
        field("alphaPhi10", create_alpha_phi10, depends_on=["fields.phi", "fields.alpha1"]),
    ]
