# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The alpha-advection model family of incompressibleVoFNeoN, selected by ``advectionScheme``."""

from typing import Any

from neofoam.solver.incompressibleVoF.models.alpha_advection.selection import (
    scheme_name_from_case,
)

from .mules import mulesNeoN

# The ``advectionScheme`` names of incompressibleVoF, so a case selects the same scheme.
_SCHEMES: dict[str, Any] = {"MULES": mulesNeoN}


class advectionModelNeoN:
    """Required model family of the NeoN advection schemes.

    Use it as the solver's advection family; ``incompressibleVoF``'s ``advectionModel`` is the
    pybFoam alternative. The member is the ``advectionScheme`` key of ``system/fvSolution``
    (default MULES), see :func:`scheme_name_from_case`. A further scheme adds itself with
    :meth:`register`; the registry is process-wide and not thread-safe.

    Example::

        spec = advectionModelNeoN.detect_and_create()
    """

    @classmethod
    def register(cls, name: str, spec: Any) -> None:
        """Add ``spec`` as the member for ``advectionScheme`` ``name``."""
        _SCHEMES[name] = spec

    @classmethod
    def all_specs(cls) -> list[Any]:
        """Every member spec of the family, case-free."""
        return list(_SCHEMES.values())

    @classmethod
    def registered_names(cls) -> list[str]:
        """The ``advectionScheme`` names the family accepts."""
        return list(_SCHEMES)

    @classmethod
    def select(cls, name: str) -> Any:
        """The member spec for ``advectionScheme`` ``name``; unknown names raise ValueError."""
        spec = _SCHEMES.get(name)
        if spec is None:
            raise ValueError(
                f"Unknown advectionScheme '{name}'; registered: {cls.registered_names()}"
            )
        return spec

    @classmethod
    def detect_and_create(cls) -> Any:
        """The active member for the case in the working directory."""
        return cls.select(scheme_name_from_case())
