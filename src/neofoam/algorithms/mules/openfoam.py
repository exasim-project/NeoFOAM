# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""MULES on pybFoam fields: ``alphaEqn.H`` primitives, controls and :class:`OpenFOAMMulesOps`."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Protocol

import pybFoam as pyf
import pybFoam.fvm as fvm
from pybFoam import Info, fvc, surfaceScalarField, volScalarField
from pybFoam import mules as mules_lib

from .equation import PhaseFractions

__all__ = [
    "InterfaceProtocol",
    "OpenFOAMMulesOps",
    "alpha_ddt_off_centring",
    "alpha_phase_flux",
    "alpha_sub_cycle",
    "crank_nicolson_flux",
    "interface_compression_velocity",
    "mules_implicit_predictor",
    "read_alpha_controls",
    "read_alpha_ddt_off_centring",
]

_ALPHA_SCHEME = "div(phi,alpha)"
_ALPHAR_SCHEME = "div(phirb,alpha)"
_ALPHA_DDT = "ddt(alpha)"
_EULER_DDT_SCHEMES = ("Euler", "localEuler")


class InterfaceProtocol(Protocol):
    """The slice of the mixture the MULES alpha equation reads.

    ``immiscibleIncompressibleTwoPhaseMixture`` satisfies it; advection-only VoF
    (:mod:`neofoam.algorithms.vof`) passes a stand-in with no mixture behind it.

    Example::

        ops = OpenFOAMMulesOps(alpha1, interface)
    """

    def cAlpha(self) -> float: ...
    def nHatf(self) -> surfaceScalarField: ...
    def correct(self) -> None: ...


def read_alpha_controls(alpha_name: str) -> tuple[int, int, bool, bool]:
    """Read the alpha solver controls of ``alpha_name`` from ``system/fvSolution``.

    Returns ``(nAlphaCorr, nAlphaSubCycles, MULESCorr, alphaApplyPrevCorr)``, the
    field order of :class:`~neofoam.algorithms.mules.AlphaControls`.
    Re-read on every alpha solve — faithful to alphaEqn.H, which pulls the
    controls from ``mesh.solverDict`` each time step (runTimeModifiable).
    Absent dicts/keys fall back to the interFoam defaults with a logged notice;
    a malformed value is a fatal OpenFOAM IO error (not catchable from Python).
    ``found``/``subDict`` regex-match the OpenFOAM solver keys (damBreak:
    ``"alpha.water.*"``).

    Example::

        controls = AlphaControls(*read_alpha_controls("alpha.water"))
    """
    defaults = (1, 1, False, False)
    try:
        fv_solution = pyf.dictionary.read("system/fvSolution")
    except RuntimeError as err:
        Info(f"read_alpha_controls: cannot read system/fvSolution ({err}); using defaults.")
        return defaults
    if not fv_solution.found("solvers"):
        Info("read_alpha_controls: no 'solvers' dict in fvSolution; using defaults.")
        return defaults
    solvers = fv_solution.subDict("solvers")
    if not solvers.found(alpha_name):
        Info(f"read_alpha_controls: no '{alpha_name}' solver dict; using defaults.")
        return defaults
    alpha_dict = solvers.subDict(alpha_name)
    return (
        alpha_dict.getOrDefault[int]("nAlphaCorr", 1),
        alpha_dict.getOrDefault[int]("nAlphaSubCycles", 1),
        alpha_dict.getOrDefault[bool]("MULESCorr", False),
        alpha_dict.getOrDefault[bool]("alphaApplyPrevCorr", False),
    )


def alpha_ddt_off_centring(
    scheme_name: str,
    scheme_coefficient: float,
    n_alpha_sub_cycles: int,
    after_first_time_step: bool,
) -> float:
    """``ocCoeff`` of ``alphaEqn.H`` for an already-read ``ddt(alpha)`` scheme.

    The decision table on its own, so it can be exercised without a mesh; use
    ``read_alpha_ddt_off_centring`` to get the inputs from a live case. 0 means
    "integrate exactly as Euler". Raises for the configurations native rejects.

    >>> alpha_ddt_off_centring("CrankNicolson", 0.5, 1, True)
    0.5
    """
    if scheme_name in _EULER_DDT_SCHEMES:
        return 0.0
    if scheme_name != "CrankNicolson":
        raise ValueError(
            f"ddtSchemes/{_ALPHA_DDT} is '{scheme_name}': only Euler and "
            "CrankNicolson ddt schemes are supported for the alpha equation."
        )
    if n_alpha_sub_cycles > 1:
        raise ValueError(
            "Sub-cycling is not supported with the CrankNicolson ddt scheme "
            f"(nAlphaSubCycles = {n_alpha_sub_cycles})."
        )
    # The scheme needs an old-time alpha flux it does not have yet, so the first
    # step of a fresh run is integrated with Euler.
    return scheme_coefficient if after_first_time_step else 0.0


def read_alpha_ddt_off_centring(mesh: pyf.fvMesh, n_alpha_sub_cycles: int) -> float:
    """``ocCoeff`` for the alpha equation, read from the case's ``system/fvSchemes``.

    Re-read on every alpha solve, as alphaEqn.H does: the ``CrankNicolson``
    coefficient may be a ``Function1`` of time. A restart is treated as a fresh
    start — ``alphaPhi0.<phase>`` is neither written nor read here. Use
    :func:`alpha_ddt_off_centring` when the scheme is already known.

    Example::

        oc_coeff = read_alpha_ddt_off_centring(mesh, controls.n_alpha_sub_cycles)
    """
    scheme_name, scheme_coefficient = mesh.ddtSchemeInfo(pyf.Word(_ALPHA_DDT))
    runtime = mesh.time()
    return alpha_ddt_off_centring(
        scheme_name,
        scheme_coefficient,
        n_alpha_sub_cycles,
        runtime.timeIndex() > runtime.startTimeIndex() + 1,
    )


def crank_nicolson_flux(phi: surfaceScalarField, oc_coeff: float) -> surfaceScalarField:
    """The off-centred volumetric flux ``phiCN`` the alpha equation advects with.

    At ``ocCoeff == 0`` ``phi`` is returned unchanged, so the Euler path never
    reads ``phi``'s old-time slot — doing so would roll it where native does not.

    Example::

        phi_cn = crank_nicolson_flux(phi, oc_coeff)
    """
    if oc_coeff <= 0:
        return phi
    cn_coeff = 1.0 / (1.0 + oc_coeff)
    return surfaceScalarField(pyf.Word("phiCN"), cn_coeff * phi + (1.0 - cn_coeff) * phi.oldTime())


@contextmanager
def alpha_sub_cycle(alpha1: volScalarField, n_alpha_sub_cycles: int) -> Iterator[Any]:
    """``subCycle<volScalarField>`` for the alpha equation; yields the sub-cycled ``Time``.

    Wrap the ``nAlphaSubCycles > 1`` loop of ``alphaEqnSubCycle.H`` in this —
    MULES and isoAdvector sub-cycle identically, only the pass inside differs.
    Call ``increment()`` on the yielded ``Foam::Time`` once per sub-step; on exit
    the real time state and ``alpha1``'s real old-time value are back, so read
    anything that refers to the *whole* step (the total ``deltaT``) before
    entering. ``Time`` is advanced rather than ``phi`` scaled by ``1/n`` so that
    time-dependent BCs (``waveAlpha``/``waveVelocity``) re-evaluate once per
    sub-step at the sub-step's own time.

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


def interface_compression_velocity(
    interface: InterfaceProtocol, phi: surfaceScalarField
) -> surfaceScalarField:
    """phic = cAlpha * |phi / magSf|, with non-coupled boundary faces zeroed.

    Mirrors the ``phic`` block at the top of ``alphaEqn.H``.  The boundary
    zeroing is done in Python via ``fvPatch.coupled()``.

    Example::

        phic = interface_compression_velocity(mixture, phi)
    """
    mesh = phi.mesh()
    phic = surfaceScalarField(pyf.Word("phic"), interface.cAlpha() * pyf.mag(phi / mesh.magSf()))
    # Index access (fvPatch by reference) — the fvBoundaryMesh iterator returns
    # fvPatch by value, which is invalid for the abstract fvPatch base.
    boundary = mesh.boundary()
    for i in range(len(boundary)):
        patch = boundary[i]
        if not patch.coupled():
            name = str(patch.name())
            phic[name] = pyf.scalarField([0.0] * len(phic[name]))
    return phic


def alpha_phase_flux(
    phi: surfaceScalarField,
    phases: PhaseFractions,
    phic: surfaceScalarField,
    interface: InterfaceProtocol,
    alphaPhiUn: surfaceScalarField,
) -> surfaceScalarField:
    """Scheme-based phase flux with interface compression (``alphaPhiUn``).

    alphaPhiUn = fvc::flux(phi, alpha1, alphaScheme)
               + fvc::flux(-fvc::flux(-phir, alpha2, alpharScheme),
                            alpha1, alpharScheme)
    with phir = phic * interface.nHatf().

    Composed from ``fvc.flux(phi, field, key=...)`` (by-name fvSchemes lookup).
    The negated intermediate fluxes are materialised into ``surfaceScalarField``
    (bitwise-identical) so they satisfy the ``fvc.flux`` surface-field
    first-argument overload.

    Writes into the persistent ``alphaPhiUn`` (registered once by
    createAlphaFluxes.H) via ``.assign``: a same-named local would deregister it
    on return.

    Example::

        alpha_phase_flux(phi, PhaseFractions(alpha1, alpha2), phic, mixture, alpha_phi_un)
    """
    phir = surfaceScalarField(pyf.Word("phir"), phic * interface.nHatf())
    neg_phir = surfaceScalarField(pyf.Word("negPhir"), -phir)
    phir_flux = surfaceScalarField(
        pyf.Word("phirFlux"), fvc.flux(neg_phir, phases.alpha2, key=_ALPHAR_SCHEME)
    )
    neg_phir_flux = surfaceScalarField(pyf.Word("negPhirFlux"), -phir_flux)
    alphaPhiUn.assign(
        fvc.flux(phi, phases.alpha1, key=_ALPHA_SCHEME)
        + fvc.flux(neg_phir_flux, phases.alpha1, key=_ALPHAR_SCHEME)
    )
    return alphaPhiUn


def mules_implicit_predictor(
    alpha1: volScalarField, phi_cn: surfaceScalarField
) -> surfaceScalarField:
    """Implicit upwind predictor for the MULESCorr branch.

    Modifies ``alpha1`` in-place and returns the upwind face flux
    (``talphaPhi1UD``). Both schemes are inline specs rather than fvSchemes
    lookups because alphaEqn.H hardcodes them: the predictor stays Euler even
    under a CrankNicolson ``ddt(alpha)`` (``phiCN`` carries the off-centring).

    Example::

        alpha_phi_upwind = mules_implicit_predictor(alpha1, phi_cn)
    """
    alpha1_eqn = pyf.fvScalarMatrix(
        fvm.ddt(alpha1, scheme="Euler") + fvm.div(phi_cn, alpha1, scheme="Gauss upwind")
    )
    alpha1_eqn.solve()
    return surfaceScalarField(pyf.Word("alphaPhi1UD"), alpha1_eqn.flux())


class OpenFOAMMulesOps:
    """:class:`~neofoam.algorithms.mules.MulesOps` on pybFoam fields, every ``alphaEqn.H`` branch.

    Holds the phase fraction it sub-cycles and the interface (mixture or a
    stand-in) that gives ``cAlpha``/``nHatf``. Sub-steps run on the case's
    ``Foam::Time`` (:func:`alpha_sub_cycle`); one sub-cycle uses the time step
    as it is, with OpenFOAM's own old-time handling.

    Example::

        ops = OpenFOAMMulesOps(alpha1, mixture)
        alpha_eqn(ops, PhaseFractions(alpha1, alpha2), VolumetricFlux(phi, phi), controls, fluxes)
    """

    def __init__(self, alpha1: volScalarField, interface: InterfaceProtocol) -> None:
        self._alpha1 = alpha1
        self._interface = interface

    def compression_velocity(self, phi: surfaceScalarField) -> surfaceScalarField:
        return interface_compression_velocity(self._interface, phi)

    def phase_flux(
        self,
        phases: PhaseFractions,
        phi: surfaceScalarField,
        phic: surfaceScalarField,
        out: surfaceScalarField,
    ) -> surfaceScalarField:
        return alpha_phase_flux(phi, phases, phic, self._interface, out)

    def implicit_upwind_predictor(
        self, alpha1: volScalarField, phi_cn: surfaceScalarField
    ) -> surfaceScalarField:
        return mules_implicit_predictor(alpha1, phi_cn)

    def explicit_solve(
        self, alpha1: volScalarField, phi_cn: surfaceScalarField, alpha_phi: surfaceScalarField
    ) -> None:
        mules_lib.explicit_solve(alpha1, phi_cn, alpha_phi)

    def correct(
        self,
        alpha1: volScalarField,
        alpha_phi: surfaceScalarField,
        alpha_phi_corr: surfaceScalarField,
    ) -> None:
        mules_lib.correct(alpha1, alpha_phi, alpha_phi_corr)

    def copy(self, field: Any) -> Any:
        return type(field)(pyf.Word(f"{field.name()}Copy"), 1.0 * field)

    def linear_combination(self, a: float, x: Any, b: float, y: Any, out: Any) -> None:
        # The loop's coefficients are 1, -1 and 0.5, so a*x + b*y rounds exactly as
        # alphaEqn.H's x + y, x - y and 0.5*x + 0.5*y do.
        out.assign(a * x + b * y)

    def correct_interface(self) -> None:
        self._interface.correct()

    def info(self, message: str) -> None:
        Info(message)

    def sub_steps(self, n_sub_cycles: int) -> Iterator[float]:
        if n_sub_cycles == 1:
            yield 1.0
            return
        total_delta_t = self._alpha1.mesh().time().deltaTValue()
        with alpha_sub_cycle(self._alpha1, n_sub_cycles) as runtime:
            for _ in range(n_sub_cycles):
                runtime.increment()
                yield runtime.deltaTValue() / total_delta_t
