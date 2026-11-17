# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The face forces of interFoam's momentum and pressure equations: gravity and surface tension."""

from dataclasses import dataclass
from typing import Any, Protocol

import pybFoam as pyf

from neofoam import neofoam_bindings as nfb

from ..interface import InterfaceModel

_DEFAULT_SURFACE_TENSION_MODEL = "Brackbill"


def _surface_like(phi: Any, name: str) -> Any:
    """A face field on ``phi``'s mesh, to be overwritten."""
    face_field = phi * 0.0
    face_field.name = name
    return face_field


def _add_into(out: Any, contribution: Any) -> None:
    """out += contribution, on internal and boundary faces."""
    nfb.linear_combination(1.0, out, 1.0, contribution, out)


@dataclass(frozen=True)
class ForceState:
    """What the face forces of one time step read: alpha, rho, the face head and the interface.

    Build one per step after the alpha advection; a force only reads it, so it is frozen.
    Tests of a single force can pass it directly instead of running ``surfaceForces``.

    Example::

        state = ForceState(alpha1, rho, ghf, interface)
    """

    alpha1: Any
    rho: Any
    ghf: Any
    interface: InterfaceModel


class FaceForce(Protocol):
    """A face-normal force contribution; ``uses_interface`` asks for ``interface.update``.

    Implement it to add a force to the sum in :class:`surfaceForces`; ``add_to`` adds in
    N/m^3 to internal and boundary faces and must leave ``state`` unchanged.

    Example::

        force: FaceForce = gravityForce(phi)
        force.add_to(out, state, runtime)
    """

    uses_interface: bool

    def add_to(self, out: Any, state: ForceState, runtime: Any) -> None:
        """Add the contribution to the face field ``out`` (internal and boundary faces)."""
        ...


class SigmaModel(Protocol):
    """The surface tension coefficient, as the cell field ``sigma*K``.

    Implement it for a coefficient other than :class:`constantSigma`; ``sigma_k`` overwrites
    ``out`` and leaves ``curvature`` unchanged.

    Example::

        sigma: SigmaModel = constantSigma(0.07)
        sigma.sigma_k(curvature, sigma_k)
    """

    def sigma_k(self, curvature: Any, out: Any) -> None:
        """Write ``sigma*curvature`` into the cell field ``out`` (internal and boundary)."""
        ...


class constantSigma:
    """A constant ``sigma`` (OpenFOAM's ``surfaceTensionModels::constant``).

    The only :class:`SigmaModel` so far, used for every case with ``sigma`` as a number;
    the coefficient never changes after construction.

    Example::

        constantSigma(0.07).sigma_k(curvature, sigma_k)
    """

    def __init__(self, sigma: float) -> None:
        self.sigma = sigma

    def sigma_k(self, curvature: Any, out: Any) -> None:
        """``sigma*K``; the zero-weighted second term leaves the product exact."""
        nfb.linear_combination(self.sigma, curvature, 0.0, curvature, out)


class gravityForce:
    """The buoyancy force ``-ghf*snGrad(rho)``.

    Always part of the sum; :class:`brackbillForce` is the surface tension contribution.
    Owns one face buffer, so an instance serves one ``surfaceForces``.

    Example::

        gravityForce(phi).add_to(out, state, runtime)
    """

    uses_interface = False

    def __init__(self, phi: Any) -> None:
        self._force = _surface_like(phi, "gravityFaceForce")

    def add_to(self, out: Any, state: ForceState, runtime: Any) -> None:
        """out += -ghf*snGrad(rho)."""
        nfb.gravity_face_force(runtime, state.ghf, state.rho, out=self._force)
        _add_into(out, self._force)


class brackbillForce:
    """Brackbill's continuum surface force ``interpolate(sigma*K)*snGrad(alpha1)``.

    interfaceProperties::surfaceTensionForce, with ``K`` the interface model's curvature.
    Selected by ``surfaceTensionModel`` (the default and only member) when ``sigma`` is not
    zero. Owns its buffers, and needs ``interface.update`` before ``add_to``.

    Example::

        brackbillForce(constantSigma(0.07), phi, runtime).add_to(out, state, runtime)
    """

    uses_interface = True

    def __init__(self, sigma: SigmaModel, phi: Any, runtime: Any) -> None:
        self._sigma = sigma
        self._sigma_k = nfb.create_uniform_volume_field(runtime, "sigmaK", 0.0)
        self._force = _surface_like(phi, "surfaceTensionForce")

    def add_to(self, out: Any, state: ForceState, runtime: Any) -> None:
        """out += interpolate(sigma*K)*snGrad(alpha1)."""
        self._sigma.sigma_k(state.interface.curvature, self._sigma_k)
        nfb.surface_tension_face_force(runtime, self._sigma_k, state.alpha1, out=self._force)
        _add_into(out, self._force)


_SURFACE_TENSION_MODELS: dict[str, Any] = {"Brackbill": brackbillForce}


def surface_tension_model(name: str) -> Any:
    """The surface tension force class for ``surfaceTensionModel`` ``name``.

    Use it to resolve a case's model; :func:`create_surface_forces` builds the whole sum.
    Unknown names raise ValueError.

    Example::

        surface_tension_model("Brackbill")
    """
    member = _SURFACE_TENSION_MODELS.get(name)
    if member is None:
        raise ValueError(
            f"Unknown surfaceTensionModel '{name}'; registered: {list(_SURFACE_TENSION_MODELS)}"
        )
    return member


def surface_tension_model_name() -> str:
    """The optional ``surfaceTensionModel`` of ``constant/transportProperties`` (Brackbill).

    Reads the case in the working directory; pass the result to :func:`surface_tension_model`.

    Example::

        surface_tension_model(surface_tension_model_name())
    """
    transport = pyf.dictionary.read("constant/transportProperties")
    return str(transport.getOrDefault[str]("surfaceTensionModel", _DEFAULT_SURFACE_TENSION_MODEL))


class surfaceForces:
    """The sum of the face forces of a run, recomputed once per time step.

    ``face_force`` holds the sum after :meth:`update`; it feeds ``phig`` and the
    fixedFluxPressure constraint of every pressure corrector. ``uses_interface`` says
    whether any contribution reads the interface, so the caller can skip its update.
    Use :func:`create_surface_forces` to build it from a case. Not thread-safe: ``update``
    overwrites ``face_force``.

    Example::

        forces = surfaceForces([gravityForce(phi)], phi)
        forces.update(ForceState(alpha1, rho, ghf, interface), runtime)
        forces.face_force
    """

    def __init__(self, forces: list[FaceForce], phi: Any) -> None:
        self.forces = forces
        self.face_force = _surface_like(phi, "faceForce")
        self.uses_interface = any(force.uses_interface for force in forces)

    def update(self, state: ForceState, runtime: Any) -> None:
        """face_force = the sum of every contribution, in list order."""
        nfb.linear_combination(0.0, self.face_force, 0.0, self.face_force, self.face_force)
        for force in self.forces:
            force.add_to(self.face_force, state, runtime)


def create_surface_forces(
    phase_properties: dict[str, Any], phi: Any, runtime: Any
) -> surfaceForces:
    """Gravity, plus surface tension when ``sigma`` is not zero.

    interFoam requires ``sigma`` in transportProperties; ``sigma 0`` leaves the face force
    gravity alone, the same sum without the zero term. Use it once per run; build
    :class:`surfaceForces` directly for a hand-picked list of forces.

    Example::

        forces = create_surface_forces(phase_properties, phi, runtime)
    """
    forces: list[FaceForce] = [gravityForce(phi)]
    sigma = float(phase_properties["sigma"])
    if sigma != 0.0:
        model = surface_tension_model(surface_tension_model_name())
        forces.append(model(constantSigma(sigma), phi, runtime))
    return surfaceForces(forces, phi)
