# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The unified momentum-transport read surface both solvers consume.

After the family merge one :func:`~neofoam.turbulence.selection.select_turbulence_model`
returns either a :class:`~neofoam.turbulence.native.NeoNHandle` (native NeoN
ModelSpec, ``fallback=False``) or a
:class:`~neofoam.turbulence.fallback.FallbackHandle` (pybFoam-OpenFOAM,
``fallback=True``). :class:`MomentumTransport` pins only the members **both**
handles share, so the shared call sites (the solver execution graphs, the field
writer) type-check against one Protocol.

The two handles additionally expose backend-specific surfaces that do NOT unify —
documented here but intentionally left off the Protocol so mypy only checks the
common members:

* **native** (``incompressibleFluidNeoN``): ``correct(U, phi, runtime)`` /
  ``nu_eff()`` / ``rotate_old_times()`` / ``write(mesh)`` / ``validate(U)`` /
  ``field(name)``.
* **fallback** (``incompressibleFluid``): ``correct()`` / ``nu()`` /
  ``viscous_stress()`` / ``divDevReff(U)``.

The two ``correct`` conventions differ by design — the NeoN algorithms call
``turbulence.correct(U, phi, neon_runtime)`` explicitly, whereas the pybFoam
fallback's scheduled op resolves the handle from the Context and calls no-arg
``correct()``. The merge is therefore at the family/registry level, not a single
``correct`` signature.
"""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class MomentumTransport(Protocol):
    """Common read surface of every momentum-transport handle."""

    def nut(self) -> Any:
        """The eddy viscosity ``nut`` the model maintains (zero for laminar)."""
        ...

    def has_nut(self) -> bool:
        """Whether the model exposes an eddy viscosity."""
        ...

    @property
    def operations(self) -> list[Any]:
        """The operations the solver steps after the pressure-velocity loop."""
        ...
