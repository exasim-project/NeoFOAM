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

from neofoam.framework.context import FieldUpdates
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

    def correct(self) -> None:
        self._require_impl().correct()

    @property
    def operations(self) -> list[Operation]:
        """One operation advancing a rate-dependent transport model after coupling.

        The OpenFOAM transport owns its viscosity; ``correct()`` runs after the
        pressure-velocity coupling (after ``continuity``).
        """
        impl = self._require_impl()

        def correct() -> FieldUpdates:
            impl.correct()
            return FieldUpdates({})

        return [
            Operation(
                func=SequentialOp(
                    wrap_with_dependency_resolution(correct, None, DependencyResolver())
                ),
                metadata=OperationMetadata(
                    op_name="of_correct_viscosity", depends_on=["continuity"]
                ),
            ),
        ]
