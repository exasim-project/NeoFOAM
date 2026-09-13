# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

# NB: no ``from __future__ import annotations`` — a Source is annotated with
# ``Context``, which the framework matches literally (see node.py).

"""Sampled surfaces: a cutting plane or an iso-surface as a source, and the Sample node.

A surface source cuts the mesh and hands the nodes a
:class:`~neofoam.postprocess.sources.geometry.SurfaceGeometry` —
face centres as positions, face areas as measure — so ``| Sum()`` is the cut
area and ``| Sample(field=...) | Mean()`` the average of a field over it::

    plane(point=(0.05, 0, 0), normal=(1, 0, 0)) | Sum(name="cross_section")
    iso_surface("alpha.water", 0.5, field="p") | Mean(name="p_interface")
"""

from typing import Any, Literal, Optional

from pybFoam import sampling

from neofoam.framework.context import Context
from neofoam.postprocess.node import DataSet, Node, Pipeline, SamplingGeometry, Source
from neofoam.postprocess.sources.geometry import SurfaceGeometry


class _SampledSurface(Source):
    """Shared resolve for the surface sources: cut the mesh, then read or sample it."""

    field: Optional[str] = None
    scheme: str = "cellPoint"

    def _foam_config(self) -> Any:
        """The ``pybFoam.sampling`` config describing the surface."""
        raise NotImplementedError

    def resolve(self, ctx: Context) -> DataSet:
        config = self._foam_config()
        # Rebuilt every time step: an iso-surface follows its field, and a plane
        # follows a moving mesh.
        surface = sampling.sampledSurface.New(config.type, ctx.mesh, config.to_foam_dict())
        surface.update()
        geometry = SurfaceGeometry(surface, ctx.fields, self.scheme)
        if self.field is None:
            return DataSet(name="area", values=geometry.measure, geometry=geometry)
        return DataSet(name=self.field, values=geometry.sample(self.field), geometry=geometry)


@Source.register
class PlaneSurface(_SampledSurface):
    """A cutting plane through the mesh, sampled on its face centres.

    Without ``field`` the values *are* the face areas, so a bare plane measures
    its own cut; name a ``field`` (or pipe into :class:`Sample`) to put a field
    on it instead. Build it through :func:`plane`::

        plane(point=(0.05, 0, 0), normal=(1, 0, 0)) | Sum(name="cross_section")
    """

    type: Literal["plane"] = "plane"
    point: tuple[float, float, float]
    normal: tuple[float, float, float]

    def _foam_config(self) -> Any:
        return sampling.SampledPlaneConfig(point=list(self.point), normal=list(self.normal))


@Source.register
class IsoSurface(_SampledSurface):
    """The surface where a registered scalar field takes ``iso_value``.

    The interface of a VOF or level-set field, or any level set of a scalar;
    like :class:`PlaneSurface` it yields face areas unless a ``field`` is named.
    Build it through :func:`iso_surface`::

        iso_surface("alpha.water", 0.5) | Sum(name="interface_area")
    """

    type: Literal["isoSurface"] = "isoSurface"
    iso_field: str
    iso_value: float

    def _foam_config(self) -> Any:
        return sampling.SampledIsoSurfaceConfig(isoField=self.iso_field, isoValue=self.iso_value)


@Node.register
class Sample(Node):
    """Interpolate a registered field onto the surface the dataset already carries.

    The way to put a second field on one surface, or to keep the geometry and
    the field apart in a case file; naming ``field`` on the source does the same
    for the first one. The interpolation scheme belongs to the surface (the
    source's ``scheme``), so every sample on one surface shares it::

        plane(point=(0.05, 0, 0), normal=(1, 0, 0)) | Sample(field="U") | Mag() | Mean()
    """

    type: Literal["sample"] = "sample"
    field: str

    def compute(self, dataset: DataSet) -> DataSet:
        geometry = dataset.geometry
        if not isinstance(geometry, SamplingGeometry):
            raise TypeError(
                f"sample needs a sampled surface; the source of {dataset.name!r} "
                f"({type(geometry).__name__}) provides none"
            )
        return dataset.model_copy(
            update={"name": self.field, "values": geometry.sample(self.field)}
        )


def plane(
    point: tuple[float, float, float],
    normal: tuple[float, float, float],
    field: Optional[str] = None,
    scheme: str = "cellPoint",
) -> Pipeline:
    """Start a pipeline from a cutting plane through *point* with *normal*.

    With no *field* the values are the face areas::

        plane((0.05, 0, 0), (1, 0, 0)) | Sum(name="cross_section")
        plane((0.05, 0, 0), (1, 0, 0), field="U") | Mag() | Mean(name="mean_speed")
    """
    return Pipeline(source=PlaneSurface(point=point, normal=normal, field=field, scheme=scheme))


def iso_surface(
    iso_field: str,
    iso_value: float,
    field: Optional[str] = None,
    scheme: str = "cellPoint",
) -> Pipeline:
    """Start a pipeline from the ``iso_field == iso_value`` surface.

    With no *field* the values are the face areas::

        iso_surface("alpha.water", 0.5) | Sum(name="interface_area")
        iso_surface("alpha.water", 0.5, field="p") | Mean(name="p_interface")
    """
    return Pipeline(
        source=IsoSurface(iso_field=iso_field, iso_value=iso_value, field=field, scheme=scheme)
    )
