# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""OpenFOAM turbulence (momentum-transport) fallback adapter.

When no native model matches the configured turbulence model, the selector
returns an :class:`OpenFOAMTurbulenceModel`. The pybFoam
``incompressibleTurbulenceModel`` owns its eddy viscosity and assembles its own
``divDevReff`` internally, so the adapter delegates the momentum stress to it
(ignoring the native Context ``nu``/``nut``, which a native closure would use)
and advances the model via a ``correct()`` operation after the pressure-velocity
coupling — the proven OpenFOAM lifecycle. Materialising the OF eddy viscosity as
a Context field is unsafe (it duplicates the model's registered ``nut``), which is
why the fallback keeps its own stress rather than the shared assembly.

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

__all__ = ["OpenFOAMTurbulenceModel", "TurbulenceFactory"]

#: A pybFoam-style turbulence factory: ``(U, phi, transport) -> model``.
TurbulenceFactory = Callable[[Any, Any, Any], Any]


def _default_factory() -> "TurbulenceFactory":
    """Return pybFoam's incompressible turbulence factory (lazy import)."""
    from pybFoam.turbulence import incompressibleTurbulenceModel

    return incompressibleTurbulenceModel.New


class OpenFOAMTurbulenceModel:
    """Adapter publishing a pybFoam turbulence model's ``nut`` as a field.

    Construction is side-effect free; the underlying model is created on
    :meth:`build`. ``divDevReff`` reuses the shared linear viscous stress, so the
    fallback assembles the momentum term the same way the native path does.
    """

    def __init__(
        self,
        U: Any,
        phi: Any,
        transport: Any,
        factory: Optional["TurbulenceFactory"] = None,
    ) -> None:
        self._U = U
        self._phi = phi
        self._transport = transport
        self._factory = factory
        self._impl: Any = None

    def build(self) -> "OpenFOAMTurbulenceModel":
        """Instantiate the underlying pybFoam turbulence model."""
        factory = self._factory or _default_factory()
        self._impl = factory(self._U, self._phi, self._transport)
        return self

    def _require_impl(self) -> Any:
        if self._impl is None:
            raise RuntimeError(
                "OpenFOAMTurbulenceModel.build() must be called before use"
            )
        return self._impl

    def nut(self) -> Any:
        return self._require_impl().nut()

    def nu(self) -> Any:
        return self._require_impl().nu()

    def divDevReff(self, U: Any, nu: Any = None, nut: Any = None) -> Any:
        """Momentum stress term — the pybFoam model assembles it internally.

        The native Context ``nu``/``nut`` are ignored: the OpenFOAM model owns its
        own eddy viscosity and stress assembly.
        """
        return self._require_impl().divDevReff(U)

    def correct(self) -> None:
        self._require_impl().correct()

    @property
    def operations(self) -> list[Operation]:
        """One operation advancing the model after the pressure-velocity coupling."""
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
                    op_name="of_correct_turbulence", depends_on=["continuity"]
                ),
            ),
        ]
