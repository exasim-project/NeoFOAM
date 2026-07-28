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
"""

from typing import Any, Callable, Optional

from neofoam.framework.context import Context, FieldUpdates
from neofoam.framework.dependency_resolver import (
    DependencyResolver,
    wrap_with_dependency_resolution,
)
from neofoam.framework.operations import Operation, SequentialOp
from neofoam.framework.types import OperationMetadata

from .stress import OpenFOAMStress

__all__ = ["OpenFOAMTurbulenceModel", "FallbackHandle", "TurbulenceFactory"]

#: A pybFoam-style turbulence factory: ``(U, phi, transport) -> model``.
TurbulenceFactory = Callable[[Any, Any, Any], Any]


def _default_factory() -> "TurbulenceFactory":
    """Return pybFoam's incompressible turbulence factory (lazy import)."""
    # Lazy so constructing the adapter stays side-effect-free and unit-testable
    # with an injected factory (test_construction_does_not_import_pybfoam).
    from pybFoam.turbulence import incompressibleTurbulenceModel  # noqa: PLC0415

    return incompressibleTurbulenceModel.New


class OpenFOAMTurbulenceModel:
    """The OpenFOAM fallback momentum-transport model — a peer of the native models.

    Selected on the fallback path, it sits at ``models.turbulence`` (wrapped by
    :class:`FallbackHandle`) as the nut/stress provider. Construction is
    side-effect free; the underlying pybFoam model is created on
    :meth:`build`. It **defines its own stress** via :meth:`viscous_stress`,
    returning an :class:`OpenFOAMStress` that delegates to the pybFoam model's own
    ``divDevReff``.
    """

    #: Descriptive tag for the stress family this model uses.
    stress_kind = "openfoam"

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
        """Instantiate the underlying pybFoam turbulence model.

        ``validate()`` recomputes the eddy viscosity from the initial ``k``/
        ``epsilon`` (``correctNut``) exactly as ``pimpleFoam`` does before the
        first solve — the ``0/nut`` shipped with a case is usually a placeholder
        (e.g. ``uniform 0``), so skipping this leaves the first momentum equation
        with ``nuEff = nu`` and diverges from native OpenFOAM.
        """
        factory = self._factory or _default_factory()
        self._impl = factory(self._U, self._phi, self._transport)
        self._impl.validate()
        return self

    def _require_impl(self) -> Any:
        if self._impl is None:
            raise RuntimeError("OpenFOAMTurbulenceModel.build() must be called before use")
        return self._impl

    def has_nut(self) -> bool:
        """The OpenFOAM model always exposes an eddy viscosity (zero when laminar)."""
        return True

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

    def viscous_stress(self) -> OpenFOAMStress:
        """The stress this model uses — delegates ``divDevReff`` to the pybFoam model."""
        return OpenFOAMStress(self)

    def correct(self) -> None:
        self._require_impl().correct()

    @property
    def operations(self) -> list[Operation]:
        """One operation advancing the model after the pressure-velocity coupling.

        The operation resolves the model from the :class:`Context` at run time
        (``ctx.models["turbulence"]``) rather than closing over ``self`` / the
        pybFoam model. Capturing the mesh-bound model in an operation closure
        would tie it into the execution-graph reference cycle, leaving it as
        cyclic garbage that the collector frees at an unsafe time during a later
        in-process run; resolving via the (acyclic) context lets it free by
        refcount when the run ends.

        .. note::
            For a *registered* model the scheduled correct comes from the model
            file's ``fallback=True`` op (via :class:`FallbackHandle`). This
            property is the correct op for a model with no registered spec,
            which the selector builds straight from OpenFOAM's own table.
        """

        def correct(ctx: Context) -> FieldUpdates:
            ctx.models["turbulence"].correct()
            return FieldUpdates({})

        return [
            Operation(
                func=SequentialOp(
                    wrap_with_dependency_resolution(correct, None, DependencyResolver())
                ),
                metadata=OperationMetadata(op_name="of_correct_turbulence"),
            ),
        ]


class FallbackHandle:
    """The momentum-transport handle for the pybFoam-OpenFOAM fallback path.

    Returned by :func:`~neofoam.turbulence.selection.select_turbulence_model`
    when a solver selects ``fallback=True`` (today: ``incompressibleFluid``). It
    pairs the :class:`OpenFOAMTurbulenceModel` — which owns the eddy viscosity
    ``nut`` and assembles its own momentum stress — with the model's co-located
    ``fallback=True`` operations (a single ``correct`` that advances the pybFoam
    model after the pressure-velocity loop).

    Satisfies :class:`~neofoam.turbulence.protocol.MomentumTransport`: it forwards
    ``nut`` / ``has_nut`` / ``nu`` / ``viscous_stress`` / ``divDevReff`` to the OF
    model and returns the spec's fallback ops as :attr:`operations`.
    """

    #: Descriptive tag for the stress family this handle uses.
    stress_kind = "openfoam"

    def __init__(self, of_model: OpenFOAMTurbulenceModel, operations: list[Operation]) -> None:
        self._of = of_model
        self._operations = list(operations)

    def build(self) -> "FallbackHandle":
        """Instantiate the underlying pybFoam turbulence model (delegates)."""
        self._of.build()
        return self

    def has_nut(self) -> bool:
        return self._of.has_nut()

    def nut(self) -> Any:
        return self._of.nut()

    def nu(self) -> Any:
        return self._of.nu()

    def divDevReff(self, U: Any, nu: Any = None, nut: Any = None) -> Any:
        return self._of.divDevReff(U, nu, nut)

    def viscous_stress(self) -> OpenFOAMStress:
        return self._of.viscous_stress()

    def correct(self) -> None:
        """Advance the wrapped pybFoam model one step (no-arg, OpenFOAM lifecycle)."""
        self._of.correct()

    @property
    def operations(self) -> list[Operation]:
        """The model's ``fallback=True`` operations, stepped after the loop."""
        return list(self._operations)
