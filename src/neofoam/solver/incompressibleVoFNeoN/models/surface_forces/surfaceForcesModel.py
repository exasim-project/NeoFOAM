# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

# NB: no ``from __future__ import annotations`` — the dependency_resolver
# matches ``Annotated[..., "models"]`` live.

"""interFoam's ``mixture.correct()`` interface update and the face forces of the step."""

from typing import Annotated, Any

from neofoam.framework.context import FieldUpdates
from neofoam.framework.initialization import model
from neofoam.framework.model import Model
from neofoam.framework.operations import Operation, Operations, SequentialOp
from neofoam.framework.types import OperationMetadata

from ..interface import InterfaceModel, interfaceModelNeoN
from .forces import ForceState, create_surface_forces, surfaceForces

__all__ = ["SurfaceForcesVoFNeoN", "surfaceForcesNeoN"]

surfaceForcesNeoN = Model("SurfaceForcesNeoN")


@surfaceForcesNeoN.build
def build(self: Any) -> list[Any]:
    """The interface model and the face forces of the case."""
    return [
        model(
            "interface",
            lambda ctx: interfaceModelNeoN.create(ctx["fields.alpha1"], ctx["_neon_runtime"]),
            depends_on=["_neon_runtime", "fields.alpha1"],
        ),
        model(
            "surface_forces",
            lambda ctx: create_surface_forces(
                ctx["models.phase_properties"], ctx["fields.phi"], ctx["_neon_runtime"]
            ),
            depends_on=["_neon_runtime", "models.phase_properties", "fields.phi"],
        ),
    ]


@surfaceForcesNeoN.operation(operation_number="2.05", name="surface_forces")
def surface_forces_update(
    alpha1: Any,
    rho: Any,
    ghf: Any,
    interface: Annotated[InterfaceModel, "models"],
    surface_forces: Annotated[surfaceForces, "models"],
    neon_runtime: Annotated[Any, "models"],
) -> FieldUpdates:
    """mixture.correct()'s calculateK on the new alpha1, then the step's face force."""
    # rho and alpha1 stay fixed through the PIMPLE correctors, so interFoam evaluates this once.
    if surface_forces.uses_interface:
        interface.update(alpha1, neon_runtime)
    surface_forces.update(ForceState(alpha1, rho, ghf, interface), neon_runtime)
    return FieldUpdates({})


@surfaceForcesNeoN.operation_collection
def collected_operations(self: Any) -> Operations:
    """Expose ``surface_forces``."""
    model_ops = Operations()
    model_ops.add(
        Operation(
            func=SequentialOp(surfaceForcesNeoN.wrap_operation(surface_forces_update, self)),
            metadata=OperationMetadata(
                op_name="surface_forces", depends_on=[], shape="box", color="lightyellow"
            ),
        )
    )
    return model_ops


class SurfaceForcesVoFNeoN:
    """Required model family whose single member updates the interface and the face forces.

    The solver registers it; use :func:`create_surface_forces` for the forces alone. The
    family holds no state.

    Example::

        member = SurfaceForcesVoFNeoN.detect_and_create()
    """

    @classmethod
    def all_specs(cls) -> list[Any]:
        """Every member spec of the family, case-free."""
        return [surfaceForcesNeoN]

    @classmethod
    def detect_and_create(cls) -> Any:
        """The active member for the case in the working directory."""
        return surfaceForcesNeoN
