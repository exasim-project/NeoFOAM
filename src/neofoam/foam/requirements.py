# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""
fvSchemes / fvSolution requirement decorators.

Usage:
    from neofoam.foam import fvSchemes, fvSolution

    @fvSchemes.add(ddt="default", div="div(phi,U)", grad="default")
    @fvSolution.add("U")
    def momentum(self, U, phi, p) -> FieldUpdates:
        ...

Short names map to full OpenFOAM section names:
    ddt → ddtSchemes, div → divSchemes, grad → gradSchemes,
    laplacian → laplacianSchemes, snGrad → snGradSchemes,
    interpolation → interpolationSchemes.
Unknown names pass through (e.g., wallDist → wallDist).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

_SECTION_MAP = {
    "ddt": "ddtSchemes",
    "div": "divSchemes",
    "grad": "gradSchemes",
    "laplacian": "laplacianSchemes",
    "snGrad": "snGradSchemes",
    "interpolation": "interpolationSchemes",
}


@dataclass(frozen=True)
class SchemeRequirement:
    """A single required fvSchemes entry (section + key)."""

    section: str  # e.g. "ddtSchemes", "divSchemes", "wallDist"
    key: str  # e.g. "default", "div(phi,U)", "method"


@dataclass(frozen=True)
class SolverRequirement:
    """A single required fvSolution solver entry."""

    field: str  # e.g. "p", "U", "nuTilda"


class _FvSchemes:
    """Decorator factory for declaring fvSchemes requirements on operations.

    Usage::

        @fvSchemes.add(ddt="default", div="div(phi,U)")
        def momentum(...):
            ...
    """

    def add(self, **kwargs: str | list[str]) -> Callable[..., Any]:
        requirements: list[SchemeRequirement] = []
        for k, v in kwargs.items():
            section = _SECTION_MAP.get(k, k)
            if isinstance(v, list):
                for entry in v:
                    requirements.append(SchemeRequirement(section=section, key=entry))
            else:
                requirements.append(SchemeRequirement(section=section, key=v))

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            existing: list[SchemeRequirement] = getattr(
                func, "_scheme_requirements", []
            )
            func._scheme_requirements = existing + requirements  # type: ignore[attr-defined]
            return func

        return decorator


class _FvSolution:
    """Decorator factory for declaring fvSolution requirements on operations.

    Usage::

        @fvSolution.add("p", "U")
        def continuity(...):
            ...
    """

    def add(self, *fields: str) -> Callable[..., Any]:
        requirements = [SolverRequirement(field=f) for f in fields]

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            existing: list[SolverRequirement] = getattr(
                func, "_solver_requirements", []
            )
            func._solver_requirements = existing + requirements  # type: ignore[attr-defined]
            return func

        return decorator


fvSchemes = _FvSchemes()
fvSolution = _FvSolution()
