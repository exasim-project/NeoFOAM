# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""MRF (multiple reference frame) rotating zones for incompressibleFluidNeoN.

Use it when a case drives its flow through a rotating cell zone declared in
``constant/MRFProperties`` (``simpleFoam/mixerVessel2D``, …). The model is
*detected*: without that file it is never instantiated and the pressure-velocity
algorithms, which inject it optionally and branch on ``None``, assemble exactly the
equations they did before this model existed.

The spec owns one runtime object — the NeoN frame handle
:class:`neofoam_bindings.MRFNeoN` — whose frame terms the contributions apply
where native's ``UEqn.H``/``pEqn.H`` apply them. ``@build`` stashes it on the
model runtime, so the contributions read it off ``self``, and its InitStep also
publishes it on the Context as ``models.mrf_neon`` for the consumers that look it
up by name. This spec covers a *subset* of the cases native MRF supports, so
``@build`` rejects the rest up front (see the guards below) rather than producing
quietly wrong fields. The pybFoam families carry their own spec in
:mod:`neofoam.solver.incompressibleFluid.models.mrf`.

Example::

    from neofoam.solver.incompressibleFluidNeoN.models.mrf import mrfNeoN
    mrfNeoN.register_with(incompressibleFluidNeoNModel)
"""

from pathlib import Path
from typing import Any

import neon._neon as nn  # NeoN Python bindings
from pydantic import ConfigDict

from neofoam import neofoam_bindings as nfb  # NeoFOAM Python bindings
from neofoam.framework.initialization import model
from neofoam.framework.model import Model
from neofoam.io import OF, BaseConfig, IOStrategy, read_section, read_toplevel

from .pressure_velocity.base import PressureVelocityAlgorithmNeoN
from .pressure_velocity.extension import momentum_extension, pressure_extension
from .pressure_velocity.simpleAlgorithm import simpleNeoN

__all__ = ["MRFPropertiesConfig", "mrfNeoN"]

# Case-relative: the solver runs with the case directory as its working directory.
_MRF_PROPERTIES = "constant/MRFProperties"
_CONTROL_DICT = Path("system/controlDict")
# The marker createDynamicFvMesh.H keys off, as neofoam.foam.initialization.new_mesh
# does: a case carrying it selects a moving mesh, a case without it a static one.
_DYNAMIC_MESH_DICT = Path("constant/dynamicMeshDict")


@IOStrategy(OF(_MRF_PROPERTIES))
class MRFPropertiesConfig(BaseConfig):
    """``constant/MRFProperties`` — one sub-dict per rotating zone.

    Declared so the file is part of the solver's config schema. The zone entries
    stay free-form: their values are consumed by ``IOMRFZoneList`` straight off
    disk, and ``omega`` alone is a Function1 with several spellings.

    Example::

        MRFPropertiesConfig.load(case_dir=case).model_extra["MRF1"]["cellZone"]
    """

    model_config = ConfigDict(extra="allow")


mrfNeoN = Model("mrf").labeled("Rotating zones (MRF)")
mrfNeoN.config(MRFPropertiesConfig)


@mrfNeoN.detect
def detect_model() -> bool:
    """MRF is active exactly when the case carries ``constant/MRFProperties``."""
    return Path(_MRF_PROPERTIES).is_file()


def _reject_time_varying_omega(config: MRFPropertiesConfig) -> None:
    """Raise unless every zone's ``omega`` is constant in time."""
    # The frame fields are probed once at build time, so a Function1 omega would
    # be silently frozen at t=0.
    for zone, entries in (config.model_extra or {}).items():
        if not isinstance(entries, dict) or "omega" not in entries:
            continue
        omega = str(entries["omega"]).strip()
        # Both Function1 spellings of a constant: the bare scalar shorthand
        # (``omega 104.72;``) and the explicit ``constant`` type.
        if omega.split()[:1] == ["constant"]:
            continue
        try:
            float(omega)
        except ValueError as exc:
            raise ValueError(
                f"MRF zone '{zone}': omega '{omega}' is not constant in time, which "
                "incompressibleFluidNeoN does not support — its frame fields are built "
                "once from the zone list. Use a constant omega."
            ) from exc


def _start_time_pressure() -> Path:
    """The ``p`` file this run reads, per ``system/controlDict``'s ``startFrom``."""
    # Not hardcoded ``0/p``: a restart reads a later directory, and a caller that
    # found no file would wave through the configuration it exists to block.
    selector = getattr(read_toplevel(_CONTROL_DICT, "startFrom"), "text", None)
    times = sorted(
        (d for d in Path().iterdir() if d.is_dir() and d.name.replace(".", "", 1).isdigit()),
        key=lambda d: float(d.name),
    )
    start: Path | None = None
    if times and selector == "latestTime":
        start = times[-1]
    elif times and selector == "firstTime":
        start = times[0]
    elif selector == "startTime":
        value = getattr(read_toplevel(_CONTROL_DICT, "startTime"), "text", None)
        start = next(
            (d for d in times if value is not None and float(d.name) == float(value)), None
        )
    if start is None or not (start / "p").is_file():
        raise ValueError(
            "MRF: cannot locate the 'p' field this case starts from (system/controlDict "
            f"startFrom {selector!r}), so the boundary-condition check cannot run. Write "
            "the pressure field into the time directory the run starts from."
        )
    return start / "p"


def _reject_fixed_flux_pressure() -> None:
    """Raise when the start-time ``p`` carries a ``fixedFluxPressure`` patch."""
    # Native closes the corrector with constrainPressure(p, U, phiHbyA, rAU, MRF),
    # which writes the frame-aware wall gradient onto exactly those patches. NeoN
    # has the boundary condition but nothing to update its refGrad, so under MRF
    # the patch would silently behave as zeroGradient — a separate feature.
    pressure = _start_time_pressure()
    offending = [
        patch
        for patch, leaves in read_section(pressure, "boundaryField").items()
        if getattr(leaves.get("type"), "text", None) == "fixedFluxPressure"
    ]
    if offending:
        raise ValueError(
            f"MRF: {pressure} patch(es) {', '.join(offending)} use fixedFluxPressure, which "
            "incompressibleFluidNeoN cannot constrain — NeoN has the boundary condition "
            "but no constrainPressure to set its refGrad from the rotating-frame flux."
        )


def _reject_transient_algorithm() -> None:
    """Raise unless the case runs the steady SIMPLE algorithm."""
    if PressureVelocityAlgorithmNeoN.detect_and_create() is simpleNeoN:
        return
    raise ValueError(
        "MRF: incompressibleFluidNeoN supports rotating zones only for the steady "
        "SIMPLE algorithm, and system/fvSolution selects a transient one. A transient "
        "run adds the absolute-frame ddt flux correction to phiHbyA, which native "
        "filters with MRF.zeroFilter() and the NeoN pressure extension has no hook "
        "for. Use a SIMPLE block, or add that hook and contribute zero_filter to it."
    )


def _reject_dynamic_mesh() -> None:
    """Raise when the case selects a moving mesh."""
    # File presence is how this codebase selects a moving mesh (foam.initialization
    # .new_mesh). Native rebuilds the zone faces from MRFZone::update(), which the
    # pybFoam spec reaches through on_mesh_change; NeoN has no such hook.
    if not _DYNAMIC_MESH_DICT.is_file():
        return
    raise ValueError(
        "MRF: incompressibleFluidNeoN does not support rotating zones on a moving "
        "mesh, and constant/dynamicMeshDict selects one. The frame fields are probed "
        "once when the handle is built and there is no mesh-change hook to rebuild "
        "them, so they would keep describing the initial mesh. Use a static mesh, or "
        "add that hook and contribute the rebuild to it."
    )


@mrfNeoN.build
def build_neon(self: Any, config: MRFPropertiesConfig) -> list[Any]:
    """Build the NeoN rotating-frame handle, also published as ``models.mrf_neon``.

    It owns its own ``Foam::IOMRFZoneList`` (built on the NeoN runtime's
    OpenFOAM mesh) plus the frame constants probed out of it — ``frame_flux``
    and ``relative_keep``, which the contributions below compose in Python — so
    no pybFoam mesh is needed on the Context. The InitStep is what orders the
    handle behind ``_neon_runtime``; the contributions take it off ``self``.
    """
    _reject_time_varying_omega(config)
    _reject_fixed_flux_pressure()
    _reject_transient_algorithm()
    _reject_dynamic_mesh()

    def create_mrf_neon(context: dict[str, Any]) -> Any:
        self.frame = nfb.MRFNeoN(context["_neon_runtime"])
        return self.frame

    return [model("mrf_neon", create_mrf_neon, depends_on=["_neon_runtime"])]


# ---------------------------------------------------------------------------
# Contributions — the rotating-frame hooks of UEqn.H / pEqn.H
# ---------------------------------------------------------------------------


@mrfNeoN.contributes(momentum_extension.constrain)
def mrf_neon_correct_boundary_velocity(self: Any, U: nn.VectorVolumeField) -> None:
    # UEqn.H's ``MRF.correctBoundaryVelocity(U)``: Omega x r on the rotating
    # wall faces, which the momentum boundary coefficients are built from.
    self.frame.correct_boundary_velocity(U)


@mrfNeoN.contributes(momentum_extension.terms)
def mrf_neon_frame_acceleration(self: Any, U: nn.VectorVolumeField) -> nn.SpatialOperatorVector:
    # UEqn.H's ``+ MRF.DDt(U)``: the frame acceleration Omega x U. MRF solves for
    # the *absolute* velocity, so this is the Coriolis term alone — no centrifugal
    # Omega x (Omega x r), which belongs to the relative-velocity (SRF)
    # formulation. The binding refreshes one acceleration buffer it owns and
    # sources from it, so no field is allocated per outer iteration.
    return self.frame.DDt(U)


@mrfNeoN.contributes(pressure_extension.predicted_flux)
def mrf_neon_predicted_flux(self: Any, phiHbyA: nn.ScalarSurfaceField) -> nn.ScalarSurfaceField:
    # pEqn.H's ``MRF.makeRelative(phiHbyA)``, in place as native does it. One
    # kernel pass subtracts the frame flux and zeroes the rotating patch faces,
    # so no temporary flux field is allocated per outer iteration.
    self.frame.make_relative(phiHbyA)
    return phiHbyA


@mrfNeoN.contributes(pressure_extension.constrain_corrected_velocity)
def mrf_neon_restore_boundary_velocity(self: Any, U: nn.VectorVolumeField) -> None:
    # See the hook: OpenFOAM's rotating wall keeps Omega x r once assigned, so
    # the turbulence correction that follows sees a moving wall. NeoN's
    # fixed-value boundary would have reset it to the case's noSlip value.
    self.frame.correct_boundary_velocity(U)
