# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The ``pyb`` / ``nb`` two-phase mixture and buoyancy operators of the parity tests."""

from __future__ import annotations

from typing import Callable

import numpy as np

from ._parity_backends import _Field, _neon_worker, _of_worker, _Worker


def _flux(velocity: _Field) -> _Field:
    """Rebuild ``phi = fvc::flux(U)`` on both backends and return its handle."""
    velocity.pool.of_worker(velocity.case).command("flux.update", (velocity.name,))
    velocity.pool.neon_worker(velocity.case, velocity.executor).command(
        "flux.update", (velocity.name,)
    )
    return _Field("phi", velocity.pool, velocity.case, velocity.executor)


def _transport_properties(field: _Field) -> np.ndarray:
    """``[rho1, nu1, rho2, nu2, sigma]`` of ``read_two_phase_transport_properties``."""
    return _neon_worker(field).evaluate("twoPhase.properties", ())


def _phases(field: _Field) -> np.ndarray:
    """``[phase1, phase2]`` of ``read_two_phase_transport_properties``."""
    return _neon_worker(field).evaluate("twoPhase.phases", ())


class _TwoPhaseOps:
    """interFoam's two-phase mixture and buoyancy terms on either backend's worker."""

    def __init__(self, worker: Callable[[_Field], _Worker]) -> None:
        self._worker = worker

    def _face_op(self, func: str, field: _Field, into: bool) -> np.ndarray:
        # into: neon writes through out= into a stale buffer; pybFoam ignores it
        return self._worker(field).evaluate(func, (field.name,), data=np.array([into]))

    def density(self, alpha: _Field) -> np.ndarray:
        return self._worker(alpha).evaluate("twoPhase.density", (alpha.name,))

    def viscosity(self, alpha: _Field) -> np.ndarray:
        return self._worker(alpha).evaluate("twoPhase.viscosity", (alpha.name,))

    def rho_phi(self, alpha: _Field) -> np.ndarray:
        """``interpolate(rho)*phi``, createFields.H's initial mass flux."""
        return self._worker(alpha).evaluate("twoPhase.rhoPhi", (alpha.name,))

    def interpolate(self, rau: _Field, into: bool = False) -> np.ndarray:
        return self._face_op("twoPhase.interpolate", rau, into)

    def multiply(self, t: _Field, rau: _Field) -> np.ndarray:
        return self._worker(t).evaluate("twoPhase.multiply", (t.name, rau.name))

    def gh(self, alpha: _Field) -> np.ndarray:
        return self._worker(alpha).evaluate("twoPhase.gh", ())

    def ghf(self, alpha: _Field) -> np.ndarray:
        return self._worker(alpha).evaluate("twoPhase.ghf", ())

    def static_pressure(self, alpha: _Field) -> np.ndarray:
        return self._worker(alpha).evaluate("twoPhase.staticPressure", (alpha.name,))

    def gravity_face_force(self, alpha: _Field, into: bool = False) -> np.ndarray:
        return self._face_op("buoyancy.gravityFaceForce", alpha, into)

    def buoyancy_flux(self, alpha: _Field, into: bool = False) -> np.ndarray:
        """``phig = gravity_face_force*interpolate(rAU)*magSf``."""
        return self._face_op("buoyancy.flux", alpha, into)

    def constrain_pressure(self, alpha: _Field) -> np.ndarray:
        """p_rgh's boundary values after constraining it with ``phiHbyA = phi + phig``."""
        return self._worker(alpha).evaluate("buoyancy.constrainPressure", (alpha.name,))

    def momentum_flux(self, alpha: _Field, into: bool = False) -> np.ndarray:
        """``(gravity_face_force - snGrad(p_rgh))*magSf`` after ``constrain_pressure``."""
        return self._face_op("buoyancy.momentumFlux", alpha, into)

    def reconstruct(self, phi: _Field) -> np.ndarray:
        return self._worker(phi).evaluate("buoyancy.reconstruct", (phi.name,))

    def buoyant_velocity(self, alpha: _Field) -> np.ndarray:
        """``U + rAU*reconstruct((phig - phi)/rAUf)``: HbyA = U, phi as the pressure flux."""
        return self._worker(alpha).evaluate("buoyancy.velocity", (alpha.name,))

    def correct_with_flux(self, field: _Field) -> np.ndarray:
        """The non-empty patches' values, in NeoN's face order, corrected with the case's phi."""
        return self._worker(field).evaluate("bc.correctWithFlux", (field.name,))


pyb = _TwoPhaseOps(_of_worker)
nb = _TwoPhaseOps(_neon_worker)
