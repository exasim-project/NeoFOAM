# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""OpenFOAM turbulence fallback adapter.

When no native NeoFOAM turbulence model is registered for the configured model
name, the selector returns an :class:`OpenFOAMTurbulenceModel`. It delegates to
pybFoam's ``incompressibleTurbulenceModel.New(U, phi, transport)`` factory — the
same call the incompressibleFluid solver uses today in ``create_fields.py``.

The pybFoam factory is *injectable* via the ``factory`` argument and otherwise
imported lazily inside :func:`_default_factory`. Importing this module therefore
never imports pybFoam, so the fallback wiring is unit-testable in a plain
(OpenFOAM-free) environment.
"""

from typing import Any, Callable, Optional

__all__ = ["OpenFOAMTurbulenceModel", "TurbulenceFactory"]

#: A pybFoam-style turbulence factory: ``(U, phi, transport) -> model``.
TurbulenceFactory = Callable[[Any, Any, Any], Any]


def _default_factory() -> "TurbulenceFactory":
    """Return pybFoam's incompressible turbulence factory (lazy import)."""
    from pybFoam.turbulence import incompressibleTurbulenceModel

    return incompressibleTurbulenceModel.New


class OpenFOAMTurbulenceModel:
    """Adapter wrapping a pybFoam turbulence model as a ``TurbulenceModel``.

    Construction is side-effect free; the underlying pybFoam model is created
    only when :meth:`build` is called. ``build`` uses the injected ``factory``
    if given, else lazily imports pybFoam via :func:`_default_factory`.
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

    def divDevReff(self, U: Any) -> Any:
        return self._require_impl().divDevReff(U)

    def correct(self) -> None:
        self._require_impl().correct()
