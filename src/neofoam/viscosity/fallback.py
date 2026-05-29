# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""OpenFOAM viscosity (transport) fallback adapter.

When no native NeoFOAM viscosity model is registered for the configured
``transportModel``, the selector returns an :class:`OpenFOAMViscosityModel`. It
delegates to pybFoam's ``singlePhaseTransportModel(U, phi)`` factory — the same
call the incompressibleFluid solver uses today in ``create_fields.py`` (the
``laminarTransport`` model).

The pybFoam factory is *injectable* via the ``factory`` argument and otherwise
imported lazily inside :func:`_default_factory`. Importing this module therefore
never imports pybFoam, so the fallback wiring is unit-testable in a plain
(OpenFOAM-free) environment.
"""

from typing import Any, Callable, Optional

__all__ = ["OpenFOAMViscosityModel", "TransportFactory"]

#: A pybFoam-style transport factory: ``(U, phi) -> model``.
TransportFactory = Callable[[Any, Any], Any]


def _default_factory() -> "TransportFactory":
    """Return pybFoam's single-phase transport factory (lazy import)."""
    from pybFoam.turbulence import singlePhaseTransportModel

    return singlePhaseTransportModel


class OpenFOAMViscosityModel:
    """Adapter wrapping a pybFoam transport model as a ``ViscosityModel``.

    Construction is side-effect free; the underlying pybFoam model is created
    only when :meth:`build` is called. ``build`` uses the injected ``factory``
    if given, else lazily imports pybFoam via :func:`_default_factory`.
    """

    def __init__(
        self,
        U: Any,
        phi: Any,
        factory: Optional["TransportFactory"] = None,
    ) -> None:
        self._U = U
        self._phi = phi
        self._factory = factory
        self._impl: Any = None

    def build(self) -> "OpenFOAMViscosityModel":
        """Instantiate the underlying pybFoam transport model."""
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
