# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Field functions — transforms that replace a dataset's values."""

from __future__ import annotations

from typing import Any, Literal, Optional

import numpy as np
from pydantic import Field

from neofoam.postprocess.node import DataSet, Node


def _vector_values(dataset: DataSet, function: str) -> "np.ndarray[Any, Any]":
    """The dataset's values as ``(n, 3)``, or a TypeError naming what is scalar."""
    values = np.asarray(dataset.values, dtype=float)
    if values.ndim != 2:
        raise TypeError(
            f"{function} needs a vector field; {dataset.name!r} has scalar values "
            f"(shape {values.shape})"
        )
    return values


@Node.register
class Mag(Node):
    """The magnitude of a vector field, ``(n, 3)`` values to ``(n,)``.

    The way to aggregate a vector as one number (``mean |U|`` rather than the
    mean of each component); use :class:`Component` to pick one component
    instead. The dataset is renamed ``mag(<name>)``::

        field("U") | Mag() | Mean()
    """

    type: Literal["mag"] = "mag"

    def compute(self, dataset: DataSet) -> DataSet:
        values = _vector_values(dataset, "mag")
        return dataset.model_copy(
            update={
                "name": f"mag({dataset.name})",
                "values": np.linalg.norm(values, axis=1),
            }
        )


@Node.register
class Component(Node):
    """One component of a vector field, ``(n, 3)`` values to ``(n,)``.

    Use it for a directional quantity (the wall-normal velocity, say); for the
    length of the vector use :class:`Mag`. The dataset is renamed
    ``<name>_<index>``::

        field("U") | Component(index=0) | Mean()
    """

    type: Literal["component"] = "component"
    index: int = Field(ge=0, le=2)

    def compute(self, dataset: DataSet) -> DataSet:
        values = _vector_values(dataset, "component")
        return dataset.model_copy(
            update={
                "name": f"{dataset.name}_{self.index}",
                "values": values[:, self.index],
            }
        )


@Node.register
class Area(Node):
    """Replace the values by the per-element measure — the face areas.

    The way to get a surface's area out of a table: the values of the sampled
    field are dropped and summing what is left is the area. Needs a geometry
    that has a measure (a patch or a sampled surface, not a point cloud)::

        patch("U", "movingWall") | Area() | Sum(name="wall_area")
    """

    type: Literal["area"] = "area"

    def compute(self, dataset: DataSet) -> DataSet:
        measure: Optional[np.ndarray[Any, Any]] = dataset.geometry.measure
        if measure is None:
            raise TypeError(
                f"area needs face areas; the source of {dataset.name!r} "
                f"({type(dataset.geometry).__name__}) provides none"
            )
        return dataset.model_copy(
            update={"name": "area", "values": np.asarray(measure, dtype=float)}
        )
