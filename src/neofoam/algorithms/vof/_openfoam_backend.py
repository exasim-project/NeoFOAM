# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The openfoam backend: interFoam's alpha equation on pybFoam fields, without the mixture."""

from __future__ import annotations

from collections.abc import Callable
from typing import Optional

import numpy as np
import pybFoam as pyf
from pybFoam import Word, dimensionedScalar, fvc, surfaceScalarField, volScalarField

from neofoam.algorithms.mules import (
    AlphaControls,
    AlphaFluxes,
    OpenFOAMMulesOps,
    PhaseFractions,
    VolumetricFlux,
    crank_nicolson_flux,
    read_alpha_controls,
    read_alpha_ddt_off_centring,
    sub_cycled_alpha_eqn,
)

_VelocityFunction = Callable[[np.ndarray, float], np.ndarray]

_DIM_INVERSE_LENGTH = pyf.dimensionSet(0, -1, 0, 0, 0, 0, 0)

# alphaCourantNo.H of advectorVoF: pos(alpha - 0.01)*pos(0.99 - alpha), strict.
_INTERFACE_LOW = 0.01
_INTERFACE_HIGH = 0.99


class _InterfaceStandIn:
    """``cAlpha``/``nHatf``/``correct`` of interfaceProperties, for a run without a mixture."""

    def __init__(self, alpha1: volScalarField, c_alpha: float) -> None:
        mesh = alpha1.mesh()
        self._alpha1 = alpha1
        self._c_alpha = c_alpha
        # interfaceProperties' deltaN: 1e-8/cbrt(average(V)).
        mean_volume = float(pyf.sum(mesh.V())) / mesh.nCells()
        self._delta_n = dimensionedScalar(
            Word("deltaN"), _DIM_INVERSE_LENGTH, 1e-8 / np.cbrt(mean_volume)
        )

    def cAlpha(self) -> float:
        return self._c_alpha

    def nHatf(self) -> surfaceScalarField:
        """``interfaceProperties::calculateK``'s nHatf for the current alpha (no contact angle)."""
        mesh = self._alpha1.mesh()
        grad_alpha_f = pyf.surfaceVectorField(
            Word("gradAlphaf"), fvc.interpolate(fvc.grad(self._alpha1, key="nHat"))
        )
        denominator = surfaceScalarField(
            Word("magGradAlphaf"), pyf.mag(grad_alpha_f) + self._delta_n
        )
        n_hat_fv = pyf.surfaceVectorField(Word("nHatfv"), grad_alpha_f / denominator)
        return surfaceScalarField(Word("nHatf"), n_hat_fv & mesh.Sf())

    def correct(self) -> None:
        # nHatf is computed from the current alpha on every call, so there is no
        # cached interface state to refresh.
        return None


class OpenFOAMBackend:
    """alpha, the fixed flux and the MULES fluxes as pybFoam fields."""

    def __init__(
        self, runtime: pyf.Time, alpha_name: str, velocity: Optional[_VelocityFunction]
    ) -> None:
        self._runtime = runtime
        self._mesh = pyf.fvMesh(runtime)
        self._volumes = np.asarray(self._mesh.V()).copy()
        self._velocity = velocity
        self._U = pyf.volVectorField.read_field(self._mesh, "U")
        if velocity is not None:
            self._set_velocity(velocity, runtime.value())
        self.phi = pyf.createPhi(self._U)
        # advectorVoF's phi0: the start-time flux that scale_flux modulates.
        self._phi0 = surfaceScalarField(Word("phi0"), self.phi * 1.0)
        self._alpha1 = volScalarField.read_field(self._mesh, alpha_name)
        self._alpha2 = volScalarField(Word("alpha2"), -self._alpha1 + 1.0)
        self._mesh.setFluxRequired(Word(alpha_name))
        self._phases = PhaseFractions(self._alpha1, self._alpha2)
        self._ops = OpenFOAMMulesOps(
            self._alpha1, _InterfaceStandIn(self._alpha1, _read_c_alpha(alpha_name))
        )
        # createAlphaFluxes.H: the persistent phase fluxes and the alphaApplyPrevCorr slot.
        self._fluxes = AlphaFluxes(
            surfaceScalarField(Word("alphaPhiUn"), 0.0 * self.phi),
            surfaceScalarField(Word("alphaPhi10"), self.phi * fvc.interpolate(self._alpha1)),
        )

    def alpha(self) -> np.ndarray:
        return np.asarray(self._alpha1.internalField()).copy()

    def cell_volumes(self) -> np.ndarray:
        return self._volumes.copy()

    def courant_numbers(self, delta_t: float) -> tuple[float, float]:
        return _courant_numbers(self.phi, self.alpha(), self._volumes, delta_t)

    def write(self) -> None:
        pyf.write(self._alpha1)

    def scale_flux(self, factor: float) -> None:
        """``phi = phi0*factor``, advectorVoF's time-dependent flux."""
        self.phi.assign(self._phi0 * factor)

    def advance(self) -> None:
        """``alphaEqnSubCycle.H`` minus rho/rhoPhi, at the runtime's current deltaT."""
        if self._velocity is not None:
            mid_step = self._runtime.value() - 0.5 * self._runtime.deltaTValue()
            self._set_velocity(self._velocity, mid_step)
            self.phi.assign(fvc.flux(self._U))
        controls = AlphaControls(*read_alpha_controls(self._alpha1.name()))
        oc_coeff = read_alpha_ddt_off_centring(self._mesh, controls.n_alpha_sub_cycles)
        flux = VolumetricFlux(self.phi, crank_nicolson_flux(self.phi, oc_coeff))
        for _ in sub_cycled_alpha_eqn(self._ops, self._phases, flux, controls, self._fluxes):
            pass

    def _set_velocity(self, velocity: _VelocityFunction, time: float) -> None:
        cell_centres = np.asarray(self._mesh.C().internalField()).copy()
        np.asarray(self._U.internalField())[:] = velocity(cell_centres, time)
        self._U.correctBoundaryConditions()


def _read_c_alpha(alpha_name: str) -> float:
    """``cAlpha`` of the alpha solver dict; required, as interfaceProperties requires it."""
    solvers = pyf.dictionary.read("system/fvSolution").subDict("solvers")
    return float(solvers.subDict(alpha_name).get[float]("cAlpha"))


def _courant_numbers(
    phi: surfaceScalarField, alpha: np.ndarray, volumes: np.ndarray, delta_t: float
) -> tuple[float, float]:
    """Flow and interface Courant numbers (``CourantNo.H``, advectorVoF's ``alphaCourantNo.H``)."""
    # pybFoam's __array__ hands out a view whatever numpy asks for, so copy
    # while the temporary field is still alive.
    sum_phi = np.asarray(
        pyf.volScalarField(pyf.Word("sumPhi"), fvc.surfaceSum(pyf.mag(phi))).internalField()
    ).copy()
    interface = (alpha > _INTERFACE_LOW) & (alpha < _INTERFACE_HIGH)
    co_num = 0.5 * float(np.max(sum_phi / volumes)) * delta_t
    alpha_co_num = 0.5 * float(np.max(interface * sum_phi / volumes)) * delta_t
    return co_num, alpha_co_num
