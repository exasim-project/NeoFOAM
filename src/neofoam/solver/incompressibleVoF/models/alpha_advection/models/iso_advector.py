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

Per-pass sequence (transcription of interIsoFoam ``alphaEqn.H``, nAlphaSubCycles=1):
``advect`` → ``rhoPhi = getRhoPhi(rho1, rho2)`` → ``alpha2 = 1 - alpha1`` →
``mixture.correct()`` → ``rho = alpha1*rho1 + alpha2*rho2``.
"""

from typing import Annotated, Any

import pybFoam.multiphase as multiphase
from pybFoam import (
    surfaceScalarField,
    volScalarField,
)

from neofoam.framework.context import FieldUpdates
from neofoam.framework.initialization import model
from neofoam.framework.operations import (
    Operation,
    Operations,
    SequentialOp,
)
from neofoam.framework.types import OperationMetadata

from ..advectionModel import Model, advectionModel
from ..shared import MixtureProtocol, shared_field_build_steps

__all__ = ["iso_advector"]

iso_advector = Model("isoAdvector").register_with(advectionModel).labeled("isoAdvector")


# ---------------------------------------------------------------------------
# Build: the shared VoF fields + the isoAdvection advector (constructed once)
# ---------------------------------------------------------------------------


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
    steps.append(
        model(
            "advector",
            create_advector,
            depends_on=["fields.alpha1", "fields.phi", "fields.U"],
        )
    )
    return steps


# ---------------------------------------------------------------------------
# Operation
# ---------------------------------------------------------------------------


@iso_advector.operation(operation_number="2.0")
def alpha_advection(
    alpha1: volScalarField,
    alpha2: volScalarField,
    phi: surfaceScalarField,
    rhoPhi: surfaceScalarField,
    rho: volScalarField,
    mixture: Annotated[MixtureProtocol, "models"],
    advector: Annotated[Any, "models"],
) -> FieldUpdates:
    """Advect alpha geometrically (isoAdvector) and update rho / rhoPhi.

    Transcribes interIsoFoam's alphaEqn.H: ``advect`` mutates ``alpha1`` in
    place; ``rhoPhi`` comes from the advector's density-weighted flux; ``rho`` is
    rebuilt from the updated phase fractions.
    """
    advector.advect()
    rhoPhi.assign(advector.get_rho_phi(mixture.rho1(), mixture.rho2()))
    alpha2.assign(-alpha1 + 1.0)
    mixture.correct()
    rho.assign(alpha1 * mixture.rho1() + alpha2 * mixture.rho2())
    return FieldUpdates(
        {"alpha1": alpha1, "alpha2": alpha2, "rho": rho, "rhoPhi": rhoPhi}
    )


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
