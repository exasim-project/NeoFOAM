# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""MRF (multiple reference frame) rotating zones, as an optional model.

Use it when a case drives its flow through a rotating cell zone declared in
``constant/MRFProperties`` (``simpleFoam/mixerVessel2D``, …). The model is
*detected*: without that file it is never instantiated and the pressure-velocity
algorithms, which inject it optionally and branch on ``None``, assemble exactly the
equations they did before this model existed.

Each spec owns one runtime object — ``ctx.models["mrf_zones"]`` for the pybFoam
families and ``ctx.models["mrf_neon"]`` for the NeoN one, both built on OpenFOAM's
:class:`Foam::IOMRFZoneList` — whose frame terms the algorithms apply where native
applies them. The contributions at the bottom of this module are one set per
solver, because the momentum term differs (``DDt(U)`` vs ``DDt(rho, U)`` for
VoF). Shared by ``incompressibleFluid`` and ``incompressibleVoF``, which
each register this one spec with their own plugin family.

Example::

    from neofoam.mrf import mrf
    mrf.register_with(incompressibleFluidModel)
"""

from pathlib import Path
from typing import Annotated, Any

import neon._neon as nn  # NeoN Python bindings
import pybFoam as pyf
from pybFoam import surfaceScalarField, volScalarField, volVectorField
from pydantic import ConfigDict

from neofoam import neofoam_bindings as nfb  # NeoFOAM Python bindings
from neofoam.framework.initialization import model
from neofoam.framework.model import Model
from neofoam.io import OF, BaseConfig, IOStrategy, read_section

__all__ = ["MRFPropertiesConfig", "mrf", "mrfNeoN"]

# Case-relative: the solver runs with the case directory as its working directory.
_MRF_PROPERTIES = "constant/MRFProperties"


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


mrf = Model("mrf").labeled("Rotating zones (MRF)")
mrf.config(MRFPropertiesConfig)


@mrf.detect
def detect_model() -> bool:
    """MRF is active exactly when the case carries ``constant/MRFProperties``."""
    return Path(_MRF_PROPERTIES).is_file()


@mrf.build
def build(_config: MRFPropertiesConfig) -> list[Any]:
    """Publish the zone list on the Context as ``models.mrf_zones``.

    Native constructs the list *after* ``createPhi.H``, so the initial flux is
    deliberately left as the absolute flux of ``0/U``. Named ``mrf_zones`` because
    the model runtime itself already occupies ``models.mrf``.
    """

    def create_mrf_zones(context: dict[str, Any]) -> pyf.IOMRFZoneList:
        return pyf.IOMRFZoneList(context["mesh"])

    return [model("mrf_zones", create_mrf_zones, depends_on=["mesh"])]


# ---------------------------------------------------------------------------
# Contributions — the rotating-frame hooks of UEqn.H / pEqn.H
# ---------------------------------------------------------------------------
# Imported below the spec: the solver package imports this module back, so
# ``mrf`` must already exist when that re-enters here mid-initialization.
from neofoam.solver.incompressibleFluid.models.pressure_velocity.extension import (  # noqa: E402
    mesh_update_extension,
    momentum_extension,
    pressure_extension,
)


@mrf.contributes(momentum_extension.correct_boundary_velocity)
def mrf_correct_boundary_velocity(U: volVectorField, mrf_zones: Annotated[Any, "models"]) -> None:
    mrf_zones.correctBoundaryVelocity(U)


@mrf.contributes(momentum_extension.terms)
def mrf_frame_acceleration(U: volVectorField, mrf_zones: Annotated[Any, "models"]) -> Any:
    return mrf_zones.DDt(U)


@mrf.contributes(pressure_extension.filter_ddt_corr)
def mrf_filter_ddt_corr(corr: Any, mrf_zones: Annotated[Any, "models"]) -> Any:
    # The ddt correction is absolute-frame, so zero it inside the MRF cells.
    return mrf_zones.zeroFilter(corr)


@mrf.contributes(pressure_extension.make_relative)
def mrf_make_relative(phiHbyA: surfaceScalarField, mrf_zones: Annotated[Any, "models"]) -> None:
    mrf_zones.makeRelative(phiHbyA)


@mrf.contributes(pressure_extension.constrain_pressure)
def mrf_constrain_pressure(
    p: volScalarField,
    U: volVectorField,
    phiHbyA: surfaceScalarField,
    rAU: volScalarField,
    mrf_zones: Annotated[Any, "models"],
) -> bool:
    pyf.constrainPressure(p, U, phiHbyA, rAU, mrf_zones)
    return True


@mrf.contributes(mesh_update_extension.on_mesh_change)
def mrf_on_mesh_change(mrf_zones: Annotated[Any, "models"]) -> None:
    mrf_zones.update()


# The VoF twins: same zone list, mass-weighted frame acceleration, ``p_rgh``
# constrained on the face mobility ``rAUf``; import placed as above.
from neofoam.solver.incompressibleVoF.models.pressure_velocity.extension import (  # noqa: E402
    mesh_update_extension as vof_mesh_update_extension,
)
from neofoam.solver.incompressibleVoF.models.pressure_velocity.extension import (  # noqa: E402
    momentum_extension as vof_momentum_extension,
)
from neofoam.solver.incompressibleVoF.models.pressure_velocity.extension import (  # noqa: E402
    pressure_extension as vof_pressure_extension,
)


@mrf.contributes(vof_momentum_extension.correct_boundary_velocity)
def mrf_vof_correct_boundary_velocity(
    U: volVectorField, mrf_zones: Annotated[Any, "models"]
) -> None:
    mrf_zones.correctBoundaryVelocity(U)


@mrf.contributes(vof_momentum_extension.terms)
def mrf_vof_frame_acceleration(
    rho: volScalarField, U: volVectorField, mrf_zones: Annotated[Any, "models"]
) -> Any:
    return mrf_zones.DDt(rho, U)


@mrf.contributes(vof_pressure_extension.filter_ddt_corr)
def mrf_vof_filter_ddt_corr(corr: Any, mrf_zones: Annotated[Any, "models"]) -> Any:
    # The ddt correction is absolute-frame, so zero it inside the MRF cells.
    return mrf_zones.zeroFilter(corr)


@mrf.contributes(vof_pressure_extension.make_relative)
def mrf_vof_make_relative(phiHbyA: surfaceScalarField, mrf_zones: Annotated[Any, "models"]) -> None:
    mrf_zones.makeRelative(phiHbyA)


@mrf.contributes(vof_pressure_extension.constrain_pressure)
def mrf_vof_constrain_pressure(
    p_rgh: volScalarField,
    U: volVectorField,
    phiHbyA: surfaceScalarField,
    rAUf: surfaceScalarField,
    mrf_zones: Annotated[Any, "models"],
) -> bool:
    pyf.constrainPressure(p_rgh, U, phiHbyA, rAUf, mrf_zones)
    return True


@mrf.contributes(vof_mesh_update_extension.on_mesh_change)
def mrf_vof_on_mesh_change(mrf_zones: Annotated[Any, "models"]) -> None:
    mrf_zones.update()


mrfNeoN = Model("mrf").labeled("Rotating zones (MRF)")
mrfNeoN.config(MRFPropertiesConfig)
mrfNeoN.detect(detect_model)


def _reject_time_varying_omega(config: MRFPropertiesConfig) -> None:
    """Raise unless every zone's ``omega`` is constant in time."""
    # omega is a Function1, but the NeoN frame fields are probed out of the zone
    # list once at build time, so a time-varying one would be silently frozen at
    # t=0 — the defect this repo is removing at readers.hpp:406,460, where a case
    # then differs for a reason nobody can see.
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
                "incompressibleFluidNeoN does not support — the rotating-frame fields are "
                "built once from the zone list. Use a constant omega, or extend the "
                "NeoN rotating-frame model to rebuild them per time step."
            ) from exc


def _reject_fixed_flux_pressure() -> None:
    """Raise when ``0/p`` carries a ``fixedFluxPressure`` patch."""
    # Native closes the corrector with constrainPressure(p, U, phiHbyA, rAU, MRF),
    # which writes the frame-aware wall gradient onto exactly those patches. NeoN
    # has the boundary condition but nothing to update its refGrad, so under MRF
    # the patch would silently behave as zeroGradient — a separate feature.
    offending = [
        patch
        for patch, leaves in read_section(Path("0/p"), "boundaryField").items()
        if getattr(leaves.get("type"), "text", None) == "fixedFluxPressure"
    ]
    if offending:
        raise ValueError(
            f"MRF: 0/p patch(es) {', '.join(offending)} use fixedFluxPressure, which "
            "incompressibleFluidNeoN cannot constrain — NeoN has the boundary condition "
            "but no constrainPressure to set its refGrad from the rotating-frame flux."
        )


@mrfNeoN.build
def build_neon(config: MRFPropertiesConfig) -> list[Any]:
    """Publish the NeoN rotating-frame handle as ``models.mrf_neon``.

    It owns its own ``Foam::IOMRFZoneList`` (built on the NeoN runtime's
    OpenFOAM mesh) plus the frame constants probed out of it — ``frame_flux``
    and ``relative_keep``, which the contributions below compose in Python — so
    no pybFoam mesh is needed on the Context.
    """
    _reject_time_varying_omega(config)
    _reject_fixed_flux_pressure()

    def create_mrf_neon(context: dict[str, Any]) -> Any:
        return nfb.MRFNeoN(context["_neon_runtime"])

    return [model("mrf_neon", create_mrf_neon, depends_on=["_neon_runtime"])]


# Import placed as above: the NeoN solver package imports this module back.
from neofoam.solver.incompressibleFluidNeoN.models.pressure_velocity.extension import (  # noqa: E402
    momentum_extension as neon_momentum_extension,
)
from neofoam.solver.incompressibleFluidNeoN.models.pressure_velocity.extension import (  # noqa: E402
    pressure_extension as neon_pressure_extension,
)


@mrfNeoN.contributes(neon_momentum_extension.constrain)
def mrf_neon_correct_boundary_velocity(U: Any, mrf_neon: Annotated[Any, "models"]) -> None:
    # UEqn.H's ``MRF.correctBoundaryVelocity(U)``: Omega x r on the rotating
    # wall faces, which the momentum boundary coefficients are built from.
    mrf_neon.correct_boundary_velocity(U)


@mrfNeoN.contributes(neon_momentum_extension.terms)
def mrf_neon_frame_acceleration(U: Any, mrf_neon: Annotated[Any, "models"]) -> Any:
    # UEqn.H's ``+ MRF.DDt(U)``, written out: the frame acceleration Omega x U.
    # MRF solves for the *absolute* velocity, so this is the Coriolis term alone
    # — no centrifugal Omega x (Omega x r), which belongs to the relative-velocity
    # (SRF) formulation. ``omega`` is zero outside the zones, so no mask is
    # needed. The cross product is a temporary; the DSL bindings keep it alive.
    return nn.exp.source(nn.cross(mrf_neon.omega, U))


@mrfNeoN.contributes(neon_pressure_extension.predicted_flux)
def mrf_neon_predicted_flux(phiHbyA: Any, mrf_neon: Annotated[Any, "models"]) -> Any:
    # pEqn.H's ``MRF.makeRelative(phiHbyA)``, written out. Native does two
    # different things — subtracts the frame flux on internal and non-rotating
    # faces, assigns zero on the rotating patch faces — and one expression covers
    # both because ``relative_keep`` is 0 exactly on the faces native assigns.
    return mrf_neon.relative_keep * (phiHbyA - mrf_neon.frame_flux)


@mrfNeoN.contributes(neon_pressure_extension.constrain)
def mrf_neon_restore_boundary_velocity(U: Any, mrf_neon: Annotated[Any, "models"]) -> None:
    # See the hook: OpenFOAM's rotating wall keeps Omega x r once assigned, so
    # the turbulence correction that follows sees a moving wall. NeoN's
    # fixed-value boundary would have reset it to the case's noSlip value.
    mrf_neon.correct_boundary_velocity(U)
