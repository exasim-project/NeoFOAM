# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""OpenFOAM viscosity (transport) fallback adapter.

When no native viscosity model matches the configured ``transportModel``, the
selector returns an :class:`OpenFOAMViscosityModel`. Like a native model it
**owns the molecular ``nu`` field**: its ``operations`` publish ``fields.nu`` from
the pybFoam ``singlePhaseTransportModel`` (driving ``correct()`` for
rate-dependent transport), so the fallback is unified onto the same Context
fields the native path uses.

pybFoam is imported lazily, so importing this module needs no OpenFOAM build.
"""

from typing import Any, Callable, Optional

from neofoam.framework.context import Context, FieldUpdates
from neofoam.framework.dependency_resolver import (
    DependencyResolver,
    wrap_with_dependency_resolution,
)
from neofoam.framework.operations import Operation, SequentialOp
from neofoam.framework.types import OperationMetadata

__all__ = ["OpenFOAMViscosityModel", "TransportFactory"]

#: A pybFoam-style transport factory: ``(U, phi) -> model``.
TransportFactory = Callable[[Any, Any], Any]


def _default_factory() -> "TransportFactory":
    """Return pybFoam's single-phase transport factory (lazy import)."""
    from pybFoam.turbulence import singlePhaseTransportModel

    return singlePhaseTransportModel


class OpenFOAMViscosityModel:
    """Adapter publishing a pybFoam transport's ``nu`` as the Context field.

    Construction is side-effect free; the underlying model is created on
    :meth:`build` (or an existing ``transport`` is adopted).
    """

    def __init__(
        self,
        U: Any = None,
        phi: Any = None,
        factory: Optional["TransportFactory"] = None,
    ) -> None:
        self._U = U
        self._phi = phi
        self._factory = factory
        self._impl: Any = None

    def build(self, transport: Any = None) -> "OpenFOAMViscosityModel":
        """Adopt an existing ``transport`` or instantiate the pybFoam model."""
        if transport is not None:
            self._impl = transport
        else:
            factory = self._factory or _default_factory()
            self._impl = factory(self._U, self._phi)
        return self

    def _require_impl(self) -> Any:
        if self._impl is None:
            raise RuntimeError(
                "OpenFOAMViscosityModel.build() must be called before use"
            )
        return self._impl

    def nu(self) -> Any:
        return self._require_impl().nu()

    def nu_field(self) -> Any:
        """Materialise the transport's ``nu`` as the Context field the model owns.

        The pybFoam ``nu()`` is a ``tmp``; it is copied into a concrete
        ``volScalarField`` named ``nu`` so the held Context field outlives the
        ``tmp`` (a retained ``tmp`` would be use-after-free).
        """
        import pybFoam

        return pybFoam.volScalarField(pybFoam.Word("nu"), self.nu())

    def correct(self) -> None:
        self._require_impl().correct()

    @property
    def operations(self) -> list[Operation]:
        """One operation advancing a rate-dependent transport model after coupling.

        ``correct()`` runs after the pressure-velocity coupling (after
        ``continuity``). It resolves the model from the :class:`Context` at run
        time rather than closing over the pybFoam transport, so the mesh-bound
        object is not captured into the execution-graph reference cycle (which
        would leave it as cyclic garbage freed at an unsafe time during a later
        in-process run).
        """

        def correct(ctx: Context) -> FieldUpdates:
            ctx.models["viscosity"].correct()
            return FieldUpdates({})

        return [
            Operation(
                func=SequentialOp(
                    wrap_with_dependency_resolution(correct, None, DependencyResolver())
                ),
                metadata=OperationMetadata(op_name="of_correct_viscosity"),
            ),
        ]
