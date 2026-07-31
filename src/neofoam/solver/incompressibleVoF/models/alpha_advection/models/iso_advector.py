# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""isoAdvector alpha-advection scheme (interIsoFoam-based).

Geometric VoF advection, a member of the ``advectionModel`` family. The
interface is reconstructed (``reconstructionScheme``, e.g. isoAlpha) and
``alpha1`` is advected geometrically by the C++ ``Foam::isoAdvection`` object
(bound as ``pybFoam.multiphase.isoAdvection``), which reads its controls from the case's
``system/fvSolution`` ``solvers."alpha.water"`` sub-dict at construction.

Unlike MULES there is no ``nAlphaCorr``/``MULESCorr``/interface-compression loop:
the advector is constructed **once** (mirroring interIsoFoam's createFields.H) and
its ``advect`` is called each outer corrector. This model owns the shared VoF
fields (via ``shared_field_build_steps``) plus an ``advector`` model, and exposes
the same ``alpha_advection`` operation contract (updates ``alpha1``/``alpha2`` and
``rho``/``rhoPhi``).

Per-pass sequence (transcription of interIsoFoam ``alphaEqn.H``): make ``U``
relative to the mesh motion → ``advect`` → make ``U`` absolute again →
``rhoPhi = getRhoPhi(rho1, rho2)`` → ``alpha2 = 1 - alpha1`` →
``mixture.correct()``. ``alphaEqnSubCycle.H`` then runs that pass
``nAlphaSubCycles`` times over a sub-cycled ``Foam::Time``, sums ``rhoPhi`` over
the sub-steps, and closes the step with ``rho = alpha1*rho1 + alpha2*rho2``.
"""

from pathlib import Path
from typing import Annotated, Any

import pybFoam as pyf
import pybFoam.multiphase as multiphase
from pybFoam import (
    fvc,
    surfaceScalarField,
    volScalarField,
    volVectorField,
)

from neofoam.foam.initialization import read_vol_field
from neofoam.framework.context import FieldUpdates
from neofoam.framework.initialization import model
from neofoam.framework.operations import (
    Operation,
    Operations,
    SequentialOp,
)
from neofoam.framework.types import OperationMetadata

from ..advectionModel import Model, advectionModel
from ..shared import MixtureProtocol, alpha_sub_cycle, shared_field_build_steps

__all__ = ["iso_advector"]

iso_advector = Model("isoAdvector").register_with(advectionModel).labeled("isoAdvector")


# ---------------------------------------------------------------------------
# Build: the shared VoF fields + the isoAdvection advector (constructed once)
# ---------------------------------------------------------------------------


def _porosity_enabled() -> bool:
    """Whether ``constant/porosityProperties`` switches porosity on."""
    path = Path("constant/porosityProperties")
    if not path.is_file():
        return False
    # Bound to a name on purpose: ``getOrDefault`` is a view into the dictionary,
    # so reading it off a temporary silently yields the default.
    properties = pyf.dictionary.read(str(path))
    return bool(properties.getOrDefault[bool]("porosityEnabled", False))


@iso_advector.build
def build(self: object) -> list[object]:
    """Register the shared VoF fields plus the persistent isoAdvection advector.

    The advector holds references to alpha1/phi/U and is reused each step, so it
    is created as a model in the init graph (interIsoFoam createFields.H).
    """

    def create_advector(context: dict[str, Any]) -> Any:
        return multiphase.isoAdvection(
            context["fields.alpha1"],
            context["fields.phi"],
            context["fields.U"],
        )

    steps = shared_field_build_steps()
    advector_depends_on = ["fields.alpha1", "fields.phi", "fields.U"]
    if _porosity_enabled():
        # createPorosity.H: isoAdvection's constructor looks "porosity" up in the
        # object registry and aborts when the switch is on but the field is absent,
        # so the read has to be ordered ahead of the advector.
        steps.append(read_vol_field(volScalarField, "porosity"))
        advector_depends_on.append("fields.porosity")
    steps.append(model("advector", create_advector, depends_on=advector_depends_on))
    return steps


# ---------------------------------------------------------------------------
# Operation
# ---------------------------------------------------------------------------


def read_n_alpha_sub_cycles(alpha_name: str) -> int:
    """``nAlphaSubCycles`` for *alpha_name*, read from ``system/fvSolution``.

    interIsoFoam's ``alphaControls.H``, which reads the one control it has out
    of ``mesh.solverDict(alpha1.name())`` on every alpha solve — so this is
    re-read every pass too, and a ``runTimeModifiable`` case can change it
    mid-run. The advector's own controls live in the same sub-dict, so the dict
    is always there by the time this runs; only the key itself falls back (to
    the no-sub-cycling 1).
    """
    solvers = pyf.dictionary.read("system/fvSolution").subDict("solvers")
    return int(solvers.subDict(alpha_name).getOrDefault[int]("nAlphaSubCycles", 1))


def advect_alpha(
    alpha1: volScalarField,
    alpha2: volScalarField,
    rhoPhi: surfaceScalarField,
    U: volVectorField,
    mixture: MixtureProtocol,
    advector: Any,
) -> None:
    """One pass of interIsoFoam's ``alphaEqn.H``: advect ``alpha1``, update ``rhoPhi``.

    ``advect`` mutates ``alpha1`` in place and ``rhoPhi`` comes from the
    advector's density-weighted flux. On a **moving** mesh the pass is bracketed
    by native's ``U -= fvc::reconstruct(mesh.phi())`` / ``U += …``: isoAdvection
    interpolates ``U`` onto the iso-face centres to get the interface normal
    velocity ``Un0``, so — unlike the flux ``phi``, which is already relative —
    it has to be handed the velocity *relative to the mesh motion*. The
    subtraction and its undo are not an exact floating-point round trip, in this
    transcription no more than in native.
    """
    mesh = alpha1.mesh()
    # Held across the advect call rather than rebuilt after it, as native does:
    # advect() changes neither the mesh geometry nor its motion flux, so native's
    # second fvc::reconstruct recomputes exactly this field. Materialised into a
    # field because fvc.reconstruct hands back a single-use ``tmp``.
    mesh_velocity = (
        volVectorField(pyf.Word("meshU"), fvc.reconstruct(mesh.phi())) if mesh.moving() else None
    )

    if mesh_velocity is not None:
        U.assign(U - mesh_velocity)

    advector.advect()

    if mesh_velocity is not None:
        U.assign(U + mesh_velocity)

    rhoPhi.assign(advector.get_rho_phi(mixture.rho1(), mixture.rho2()))
    alpha2.assign(-alpha1 + 1.0)
    mixture.correct()


@iso_advector.operation(operation_number="2.0")
def alpha_advection(
    alpha1: volScalarField,
    alpha2: volScalarField,
    phi: surfaceScalarField,
    rhoPhi: surfaceScalarField,
    rho: volScalarField,
    U: volVectorField,
    mixture: Annotated[MixtureProtocol, "models"],
    advector: Annotated[Any, "models"],
) -> FieldUpdates:
    """Advect alpha geometrically (isoAdvector) and update rho / rhoPhi.

    Transcribes interIsoFoam's ``alphaEqnSubCycle.H``: ``nAlphaSubCycles``
    passes of :func:`advect_alpha` over a sub-cycled ``Foam::Time``, each
    advancing ``alpha1`` by ``deltaT/n`` with the *full* flux, with ``rhoPhi``
    accumulated as the sub-step-length-weighted mean. ``rho`` is rebuilt once,
    after the sub-cycle: writing it rolls its old-time value, which the momentum
    ``fvm::ddt(rho, U)`` needs from the start of the real time step.

    The trailing ``mixture.correct()`` is ``interIsoFoam.C:166``, the call that
    follows the whole alpha block on top of the one inside ``alphaEqn.H`` — a
    no-op unless the case has an ``alphaContactAngle``-family patch, whose
    ``correctContactAngle`` rewrites alpha1's patch gradient on every call and so
    shifts the curvature ``K``.
    """
    n_alpha_sub_cycles = read_n_alpha_sub_cycles(alpha1.name())

    if n_alpha_sub_cycles > 1:
        total_delta_t = alpha1.mesh().time().deltaTValue()
        rho_phi_sum = surfaceScalarField(pyf.Word("rhoPhiSum"), 0.0 * rhoPhi)
        with alpha_sub_cycle(alpha1, n_alpha_sub_cycles) as runtime:
            for _ in range(n_alpha_sub_cycles):
                runtime.increment()
                advect_alpha(alpha1, alpha2, rhoPhi, U, mixture, advector)
                rho_phi_sum.assign(rho_phi_sum + (runtime.deltaTValue() / total_delta_t) * rhoPhi)
        rhoPhi.assign(rho_phi_sum)
    else:
        advect_alpha(alpha1, alpha2, rhoPhi, U, mixture, advector)

    rho.assign(alpha1 * mixture.rho1() + alpha2 * mixture.rho2())
    mixture.correct()
    return FieldUpdates({"alpha1": alpha1, "alpha2": alpha2, "rho": rho, "rhoPhi": rhoPhi, "U": U})


# ---------------------------------------------------------------------------
# Operation collection: expose alpha_advection
# ---------------------------------------------------------------------------


@iso_advector.operation_collection
def collected_operations(self: object) -> Operations:
    # The collection path bypasses the spec's default operation wrapping, so
    # wrap alpha_advection with dependency resolution here (``self`` is the
    # bound runtime).
    wrapped_alpha_advection = iso_advector.wrap_operation(alpha_advection, self)
    model_ops = Operations()
    model_ops.add(
        Operation(
            func=SequentialOp(wrapped_alpha_advection),
            metadata=OperationMetadata(
                op_name="alpha_advection",
                depends_on=[],
                before=[],
                shape="box",
                color="lightgreen",
            ),
        )
    )
    return model_ops
