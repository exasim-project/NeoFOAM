# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Where a port sits in its region's mesh: :class:`Patch`, :class:`Cells`, :class:`Global`.

A location is its geometry: it resolves it from the region's :class:`MeshGeometry`,
which each solver backend provides. The mappings use it; the coupled instances
never see it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Optional, Protocol, Union

import numpy as np


class MeshGeometry(Protocol):
    """The geometry of a region's mesh, as a backend provides it, in mesh order.

    ``face_centres`` / ``cell_centres`` are ``(n, dim)``; ``face_areas`` (``|Sf|``)
    and ``cell_volumes`` are ``(n,)``. A backend that lacks one returns ``None``;
    a mapping that needs it then says so.
    """

    def face_centres(self, patch: str) -> Any: ...

    def face_areas(self, patch: str) -> Any: ...

    def cell_centres(self) -> Any: ...

    def cell_volumes(self) -> Any: ...


@dataclass(frozen=True, eq=False)
class Geometry:
    """A location's points (one row each) and their measures (area or volume); ``None``: unknown."""

    points: Optional[Any] = None
    measures: Optional[Any] = None
    size: Optional[int] = None

    @staticmethod
    def of(points: Any, measures: Any) -> Geometry:
        """The geometry of ``points`` and ``measures`` as float arrays, sized by either."""
        points = None if points is None else np.asarray(points, dtype=float)
        measures = None if measures is None else np.asarray(measures, dtype=float)
        sized = points if points is not None else measures
        return Geometry(points, measures, None if sized is None else len(sized))


@dataclass(frozen=True)
class Patch:
    """The faces of the boundary patch ``name``; holds ``FaceData``."""

    name: str
    entities: ClassVar[tuple[str, ...]] = ("face",)

    def geometry(self, mesh: MeshGeometry) -> Geometry:
        return Geometry.of(mesh.face_centres(self.name), mesh.face_areas(self.name))


@dataclass(frozen=True)
class Cells:
    """Every cell of the region; holds ``CellData``. ``name`` names its port."""

    name: str = "cells"
    entities: ClassVar[tuple[str, ...]] = ("cell",)

    def geometry(self, mesh: MeshGeometry) -> Geometry:
        return Geometry.of(mesh.cell_centres(), mesh.cell_volumes())


@dataclass(frozen=True)
class Global:
    """One value per field, no geometry: holds ``ValueData``, or ``ProbeData`` of a probe."""

    name: str
    entities: ClassVar[tuple[str, ...]] = ("value", "probe")

    def geometry(self, mesh: MeshGeometry) -> Geometry:
        return Geometry(size=1)


Location = Union[Patch, Cells, Global]
