# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""interFoam's ``alphaEqn.H`` corrector loop and ``alphaEqnSubCycle.H`` driver, for any backend."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Protocol

__all__ = [
    "AlphaControls",
    "AlphaFluxes",
    "MulesOps",
    "PhaseFractions",
    "VolumetricFlux",
    "alpha_eqn",
    "sub_cycled_alpha_eqn",
]


@dataclass(frozen=True)
class AlphaControls:
    """The ``alpha.<phase>`` solver controls ``alphaEqn.H`` branches on.

    Build it from ``read_alpha_controls``, whose tuple has the same order:

    >>> AlphaControls(*(2, 1, True, False)).mules_corr
    True
    """

    n_alpha_corr: int
    n_alpha_sub_cycles: int
    mules_corr: bool
    alpha_apply_prev_corr: bool


@dataclass(frozen=True)
class PhaseFractions:
    """``alpha1`` and ``alpha2 = 1 - alpha1``; both are updated in place.

    ``alpha2`` is ``None`` for a backend whose kernels derive ``1 - alpha1``
    themselves (NeoN): keeping the field up to date would cost a field per corrector.

    >>> PhaseFractions(alpha1, alpha2)  # doctest: +SKIP
    """

    alpha1: Any
    alpha2: Any = None


@dataclass(frozen=True)
class VolumetricFlux:
    """The flux ``phi`` that compresses and the off-centred ``phi_cn`` that transports.

    ``phi_cn`` is ``phi`` itself under Euler and in every sub-cycle; the two must
    not be swapped (``alphaEqn.H``).

    >>> VolumetricFlux(phi, phi)  # doctest: +SKIP
    """

    phi: Any
    phi_cn: Any


@dataclass
class AlphaFluxes:
    """The phase fluxes of ``createAlphaFluxes.H`` that live across alpha passes.

    ``un`` (``alphaPhiUn``) and ``phi10`` (``alphaPhi10``) are persistent fields,
    overwritten in place every corrector; ``corr0`` is the ``talphaPhi1Corr0``
    slot (``None`` when empty), which ``alpha_eqn`` rebinds.

    >>> AlphaFluxes(alpha_phi_un, alpha_phi10)  # doctest: +SKIP
    """

    un: Any
    phi10: Any
    corr0: Any = None


class MulesOps(Protocol):
    """The backend operations of one ``alphaEqn.H`` pass.

    Implement it to run :func:`alpha_eqn` on another field backend.
    ``OpenFOAMMulesOps`` implements every branch on pybFoam fields, ``NeoNMulesOps``
    on NeoN fields. Everything backend-specific (fluxes, MULES, the implicit
    predictor, sub-stepping, the field updates of the MULESCorr correctors) goes
    through here: the loop itself only uses ``assign`` and field arithmetic with
    scalars and fields (``+``, ``-``, ``*``, ``-x + 1.0``). Methods that take an
    ``out`` or a field to advance update it in place.

    Example::

        ops: MulesOps = OpenFOAMMulesOps(alpha1, mixture)
        alpha_phi10 = alpha_eqn(ops, phases, flux, controls, fluxes)
    """

    def compression_velocity(self, phi: Any) -> Any:
        """``phic = cAlpha*|phi/magSf|``, zero on non-coupled boundary faces."""

    def phase_flux(self, phases: PhaseFractions, phi: Any, phic: Any, out: Any) -> Any:
        """``alphaPhiUn`` (scheme flux plus compression) assigned into ``out``; returns ``out``."""

    def implicit_upwind_predictor(self, alpha1: Any, phi_cn: Any) -> Any:
        """Solve the Euler implicit-upwind alpha equation in place; return its face flux."""

    def explicit_solve(self, alpha1: Any, phi_cn: Any, alpha_phi: Any) -> None:
        """``MULES::explicitSolve``: limit ``alpha_phi`` and advance ``alpha1``."""

    def correct(self, alpha1: Any, alpha_phi: Any, alpha_phi_corr: Any) -> None:
        """``MULES::correct``: limit ``alpha_phi_corr`` and apply it to ``alpha1``."""

    def copy(self, field: Any) -> Any:
        """A copy of ``field``'s values, valid until the next ``copy`` of the same field."""

    def linear_combination(self, a: float, x: Any, b: float, y: Any, out: Any) -> None:
        """``out = a*x + b*y``; ``out`` may be ``x`` or ``y``."""

    def correct_interface(self) -> None:
        """Refresh the interface state after ``alpha1`` changed (``mixture.correct()``)."""

    def info(self, message: str) -> None:
        """Log ``message`` as the backend's solver output."""

    def sub_steps(self, n_sub_cycles: int) -> Iterator[float]:
        """Set up each of the ``n_sub_cycles`` sub-steps; yield its fraction of the step."""


def alpha_eqn(
    ops: MulesOps,
    phases: PhaseFractions,
    flux: VolumetricFlux,
    controls: AlphaControls,
    fluxes: AlphaFluxes,
) -> Any:
    """One pass of ``alphaEqn.H``: advance ``phases`` in place and return ``fluxes.phi10``.

    Use :func:`sub_cycled_alpha_eqn` to run a whole time step (it sub-steps);
    call this directly only inside a (sub-)step the caller has set up.
    ``controls.n_alpha_sub_cycles`` is not read here. ``phic`` is computed once
    per pass, from ``flux.phi``, as ``alphaEqn.H`` does.

    Example::

        alpha_phi10 = alpha_eqn(ops, PhaseFractions(a1, a2), VolumetricFlux(phi, phi),
                                AlphaControls(1, 1, False, False), AlphaFluxes(un, phi10))
    """
    phic = ops.compression_velocity(flux.phi)
    if controls.mules_corr:
        _implicit_predictor(ops, phases, flux.phi_cn, controls, fluxes)

    for a_corr in range(controls.n_alpha_corr):
        alpha_phi_un = ops.phase_flux(phases, flux.phi, phic, fluxes.un)
        if controls.mules_corr:
            _limited_correction(ops, phases.alpha1, alpha_phi_un, fluxes, a_corr == 0)
        else:
            fluxes.phi10.assign(alpha_phi_un)
            ops.explicit_solve(phases.alpha1, flux.phi_cn, fluxes.phi10)
        _update_alpha2(ops, phases)

    _cache_compression_correction(ops, controls, fluxes)
    return fluxes.phi10


def sub_cycled_alpha_eqn(
    ops: MulesOps,
    phases: PhaseFractions,
    flux: VolumetricFlux,
    controls: AlphaControls,
    fluxes: AlphaFluxes,
) -> Iterator[tuple[float, Any]]:
    """``alphaEqnSubCycle.H``: ``nAlphaSubCycles`` alpha passes over one time step.

    Yields ``(sub-step fraction of deltaT, alphaPhi10)`` after every pass, so the
    caller can accumulate ``rhoPhi``; iterate it to the end, since the last
    ``next`` closes the sub-cycle (``ops.sub_steps``). With one sub-cycle this
    is a single :func:`alpha_eqn` pass over the whole step. Sub-cycling only runs
    under Euler, so for more than one sub-cycle ``flux.phi_cn`` must be ``phi``.

    Example::

        for fraction, alpha_phi10 in sub_cycled_alpha_eqn(ops, phases, flux, controls, fluxes):
            rho_phi_sum.assign(rho_phi_sum + fraction * rho_phi_of(alpha_phi10))
    """
    for fraction in ops.sub_steps(controls.n_alpha_sub_cycles):
        yield fraction, alpha_eqn(ops, phases, flux, controls, fluxes)


def _update_alpha2(ops: MulesOps, phases: PhaseFractions) -> None:
    """alpha2 = 1 - alpha1 (when kept), then refresh the interface."""
    if phases.alpha2 is not None:
        phases.alpha2.assign(-phases.alpha1 + 1.0)
    ops.correct_interface()


def _implicit_predictor(
    ops: MulesOps,
    phases: PhaseFractions,
    phi_cn: Any,
    controls: AlphaControls,
    fluxes: AlphaFluxes,
) -> None:
    """The MULESCorr implicit-upwind predictor, plus the previous pass's correction."""
    # The pure upwind flux stays alive alongside alphaPhi10, which is about to grow
    # by the previous pass's correction; the cache at the end measures against it.
    alpha_phi1_ud = ops.implicit_upwind_predictor(phases.alpha1, phi_cn)
    fluxes.phi10.assign(alpha_phi1_ud)

    previous_corr = fluxes.corr0
    if controls.alpha_apply_prev_corr and previous_corr is not None:
        ops.info("Applying the previous iteration compression flux")
        ops.correct(phases.alpha1, fluxes.phi10, previous_corr)
        ops.linear_combination(1.0, fluxes.phi10, 1.0, previous_corr, fluxes.phi10)

    fluxes.corr0 = alpha_phi1_ud
    _update_alpha2(ops, phases)


def _limited_correction(
    ops: MulesOps, alpha1: Any, alpha_phi_un: Any, fluxes: AlphaFluxes, first: bool
) -> None:
    """One MULESCorr corrector: limit and apply the correction to the upwind base."""
    # alpha1 before this corrector, so later correctors can under-relax (alpha10).
    alpha10 = ops.copy(alpha1)
    alpha_phi_corr = ops.copy(alpha_phi_un)
    ops.linear_combination(1.0, alpha_phi_un, -1.0, fluxes.phi10, alpha_phi_corr)
    ops.correct(alpha1, alpha_phi_un, alpha_phi_corr)

    if first:
        ops.linear_combination(1.0, fluxes.phi10, 1.0, alpha_phi_corr, fluxes.phi10)
        return
    ops.linear_combination(0.5, alpha1, 0.5, alpha10, alpha1)
    ops.linear_combination(1.0, fluxes.phi10, 0.5, alpha_phi_corr, fluxes.phi10)


def _cache_compression_correction(
    ops: MulesOps, controls: AlphaControls, fluxes: AlphaFluxes
) -> None:
    """Turn the cached upwind flux into this pass's correction, or empty the slot."""
    # Any other configuration empties the slot, so switching alphaApplyPrevCorr off
    # mid-run cannot leave a stale correction in use.
    cached_upwind = fluxes.corr0
    if controls.mules_corr and controls.alpha_apply_prev_corr and cached_upwind is not None:
        ops.linear_combination(1.0, fluxes.phi10, -1.0, cached_upwind, cached_upwind)
    else:
        fluxes.corr0 = None
