# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Field functions — transforms that replace a dataset's values."""

from __future__ import annotations

from typing import Any, Literal, Union

from pydantic import Field

from neofoam import neofoam_bindings as nfb  # NeoFOAM Python bindings
from neofoam.postprocess.node import (
    FieldDataSets,
    InternalDataSet,
    Node,
    PatchDataSet,
    SurfaceDataSet,
)
from neofoam.postprocess.nodes._arrays import is_vector

#: The post-processing kernels: element-wise math and reductions that run on the
#: executor the values live on, host numpy included.
pp = nfb.postprocess


def _vector_field(dataset: FieldDataSets, function: str) -> Any:
    """The dataset's values, or a TypeError naming what is scalar."""
    if not is_vector(dataset.field):
        raise TypeError(f"{function} needs a vector field; {dataset.name!r} has scalar values")
    return dataset.field


@Node.register
class Mag(Node):
    """The magnitude of a vector field, ``(n, 3)`` values to ``(n,)``.

    The way to aggregate a vector as one number (``mean |U|`` rather than the
    mean of each component); use :class:`Component` to pick one component
    instead. The dataset is renamed ``mag(<name>)``::

        field("U") | Mag() | Mean()
    """

    type: Literal["mag"] = "mag"

    def compute(self, dataset: FieldDataSets) -> FieldDataSets:
        return dataset.model_copy(
            update={
                "name": f"mag({dataset.name})",
                "field": pp.mag(_vector_field(dataset, "mag")),
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

    def compute(self, dataset: FieldDataSets) -> FieldDataSets:
        return dataset.model_copy(
            update={
                "name": f"{dataset.name}_{self.index}",
                "field": pp.component(_vector_field(dataset, "component"), self.index),
            }
        )


@Node.register
class Area(Node):
    """Replace the values by the per-element measure — face areas, or cell volumes.

    The way to get a surface's area out of a table: the values of the sampled
    field are dropped and summing what is left is the area. Which measure it is
    belongs to the geometry — the face area on a patch or a sampled surface, the
    cell volume on the cells, so summing it there is the mesh volume — and a
    point cloud, which has neither, is refused by name::

        patch("U", "movingWall") | Area() | Sum(name="wall_area")
        field("p") | Area() | Sum(name="mesh_volume")
    """

    type: Literal["area"] = "area"

    def compute(self, dataset: Union[InternalDataSet, PatchDataSet, SurfaceDataSet]) -> Any:
        measure = next(
            (
                accessor
                for name in ("face_area_magnitudes", "volumes")
                if (accessor := getattr(dataset.geometry, name, None)) is not None
            ),
            None,
        )
        if measure is None:
            raise TypeError(
                f"area needs a per-element measure; the source of {dataset.name!r} "
                f"({type(dataset.geometry).__name__}) provides none"
            )
        return dataset.model_copy(update={"name": "area", "field": measure()})
