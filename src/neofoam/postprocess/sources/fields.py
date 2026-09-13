# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

# NB: no ``from __future__ import annotations`` — a source's ``resolve`` takes the
# live Context, and the framework matches ``param.annotation is Context``
# literally (see node.py).

"""Sources reading a cell field, a boundary patch and a set of sample points.

:class:`InternalField` is the default source — the internal values of a
registered volume field. Its counterparts for data that is not a cell field are
:class:`PatchField` for a wall or an inlet and :class:`LineField` for a probe
along a line, a cloud of points, a polyline or a circle. All are declarative —
the pybFoam objects live in :mod:`neofoam.postprocess.sources.geometry` and are
built per time step::

    field("p") | VolIntegrate(name="volume_p")
    patch("p", "movingWall") | SurfIntegrate(name="wall_force")
    line("U", start=(0, 0, 0.005), end=(0, 0.1, 0.005), n_points=20) | Mag() | Rows()
"""

from collections.abc import Mapping
from typing import Any, Literal, Optional

import numpy as np
from pybFoam import sampling
from pydantic import model_validator

from neofoam.framework.context import Context
from neofoam.postprocess.node import DataSet, Pipeline, Source
from neofoam.postprocess.sources.geometry import CellGeometry, PatchGeometry, PointGeometry

#: A point in space, as a spec writes it.
Point = tuple[float, float, float]

#: The value ``sampleSet*`` returns where it could not evaluate the field,
#: i.e. ``pTraits<scalar>::max``.
OUT_OF_MESH: float = sampling.OUT_OF_MESH

#: Per ``set_type``: the pybFoam config class, its required fields, its optional ones.
_SET_TYPES: dict[str, tuple[str, tuple[str, ...], tuple[str, ...]]] = {
    "uniform": ("UniformSetConfig", ("start", "end", "n_points"), ()),
    "cloud": ("CloudSetConfig", ("points",), ()),
    "polyLine": ("PolyLineSetConfig", ("points",), ("n_points",)),
    "circle": ("CircleSetConfig", ("origin", "circle_axis", "start_point", "d_theta"), ()),
}

#: The spec's field names where pybFoam spells them differently.
_FOAM_NAMES = {
    "n_points": "nPoints",
    "circle_axis": "circleAxis",
    "start_point": "startPoint",
    "d_theta": "dTheta",
}


def _registered_field(fields: Mapping[str, Any], name: str) -> Any:
    """The field registered under *name*, or the one that answers to it.

    A solver may register a field under a key of its own: incompressibleVoF
    registers the phase fraction ``alpha.water`` as ``alpha1``, and a table
    written against either spelling has to find it.
    """
    if name in fields:
        return fields[name]
    for field in fields.values():
        own_name = getattr(field, "name", None)
        if callable(own_name) and str(own_name()) == name:
            return field
    raise KeyError(f"postProcess: field {name!r} is not registered; available: {sorted(fields)}")


def _internal_values(field: Any) -> "np.ndarray[Any, Any]":
    """The internal values of a volume field as host numpy, pybFoam or NeoN.

    A pybFoam volume field hands its cell values over as ``internalField()``; a
    NeoN one keeps them in a ``Vector`` on its executor, which may be a device,
    so ``copy_to_host`` is the only way in — NeoN's ``__array__`` refuses a
    device vector outright. Duck-typed on the accessor rather than on the class
    so this module keeps importing neither NeoN nor its bindings.
    """
    internal_field = getattr(field, "internalField", None)
    if internal_field is not None:
        return np.asarray(internal_field())
    return np.asarray(field.internal_vector().copy_to_host())


def _in_mesh(values: "np.ndarray[Any, Any]") -> "np.ndarray[Any, Any]":
    """Which sampled values are real — the rest carry the ``OUT_OF_MESH`` sentinel.

    OpenFOAM already drops the spec's points that fall outside the mesh, so this
    catches only the points it kept but could not evaluate. The sentinel is the
    largest representable double, so any threshold below it separates the two —
    and the largest component is the test, not the norm, which would overflow on
    the sentinel itself (R9).
    """
    magnitude = np.abs(values) if values.ndim == 1 else np.abs(values).max(axis=1)
    return magnitude < 0.1 * OUT_OF_MESH


@Source.register
class InternalField(Source):
    """The internal field of a registered volume field, on the cell geometry.

    The default source: use it whenever a table works on cell values. The
    ``field`` is the key the solver registered, or the name the field itself
    carries (incompressibleVoF registers ``alpha.water`` as ``alpha1``, and
    either spelling finds it). A pybFoam field and a NeoN one are both read —
    the NeoN one through a host copy of its internal vector. Build it through
    :func:`field`::

        field("p") | VolIntegrate()
    """

    type: Literal["internal"] = "internal"
    field: str

    def resolve(self, ctx: Context) -> DataSet:
        return DataSet(
            name=self.field,
            values=_internal_values(_registered_field(ctx.fields, self.field)),
            geometry=CellGeometry(ctx.mesh),
        )


@Source.register
class PatchField(Source):
    """A registered volume field's values on one boundary patch, on the face geometry.

    The source for anything measured on a wall, an inlet or an outlet; the
    geometry's measure is the face area, so
    :class:`~neofoam.postprocess.nodes.aggregators.SurfIntegrate` integrates over the
    patch and :class:`~neofoam.postprocess.nodes.field_functions.Area` yields its area.
    Build it through :func:`patch`::

        patch("p", "movingWall") | SurfIntegrate(name="wall_p")
    """

    type: Literal["patch"] = "patch"
    field: str
    patch: str

    def resolve(self, ctx: Context) -> DataSet:
        geometry = PatchGeometry(ctx.mesh, self.patch, ctx.fields)
        return DataSet(name=self.field, values=geometry.sample(self.field), geometry=geometry)


@Source.register
class LineField(Source):
    """A registered volume field sampled at a set of points, on the point geometry.

    The source for a probe: ``set_type`` picks the point layout and the rest of
    the spec is that layout's data — ``start``/``end``/``n_points`` for
    ``uniform``, ``points`` for ``cloud`` and ``polyLine``,
    ``origin``/``circle_axis``/``start_point``/``d_theta`` for ``circle``; a
    layout's missing data is rejected when the source is built, not at the first
    time step. There is no measure, so pair it with
    :class:`~neofoam.postprocess.nodes.rows.Rows` (one row per point) or with
    :class:`~neofoam.postprocess.nodes.aggregators.Mean`. Points outside the mesh are
    masked out. Build it through :func:`line`::

        line("U", start=(0.05, 0, 0.005), end=(0.05, 0.1, 0.005), n_points=20) | Mag() | Rows()
    """

    type: Literal["line"] = "line"
    field: str
    set_type: Literal["uniform", "cloud", "polyLine", "circle"] = "uniform"
    start: Optional[Point] = None
    end: Optional[Point] = None
    n_points: Optional[int] = None
    points: Optional[list[Point]] = None
    origin: Optional[Point] = None
    circle_axis: Optional[Point] = None
    start_point: Optional[Point] = None
    d_theta: Optional[float] = None

    @model_validator(mode="after")
    def _the_set_type_has_its_data(self) -> "LineField":
        _, required, _ = _SET_TYPES[self.set_type]
        missing = [name for name in required if getattr(self, name) is None]
        if missing:
            raise ValueError(
                f"line source of {self.field!r}: set_type {self.set_type!r} needs {missing}"
            )
        return self

    def resolve(self, ctx: Context) -> DataSet:
        geometry = PointGeometry(
            ctx.mesh, self._set_config().to_foam_dict(), self.field, ctx.fields
        )
        values = geometry.sample(self.field)
        return DataSet(name=self.field, values=values, geometry=geometry, mask=_in_mesh(values))

    def _set_config(self) -> Any:
        """The spec as the ``pybFoam.sampling`` config that ``sampledSet::New`` reads."""
        config_name, required, optional = _SET_TYPES[self.set_type]
        data = {
            _FOAM_NAMES.get(name, name): getattr(self, name)
            for name in (*required, *optional)
            if getattr(self, name) is not None
        }
        # "distance" only selects how the set reports a point's coordinate, which
        # nothing downstream reads — the positions come from the geometry.
        return getattr(sampling, config_name)(axis="distance", **data)


def field(name: str) -> Pipeline:
    """Start a pipeline from a registered volume field's internal values.

    The entry point of the user-facing API::

        field("alpha.water") | VolIntegrate(name="water_volume")
    """
    return Pipeline(source=InternalField(field=name))


def patch(field: str, patch_name: str) -> Pipeline:
    """Start a pipeline from a field's values on one boundary patch.

    The patch counterpart of :func:`field`::

        patch("p", "movingWall") | SurfIntegrate(name="wall_p")
    """
    return Pipeline(source=PatchField(field=field, patch=patch_name))


def line(
    field: str,
    start: Point,
    end: Point,
    n_points: int,
) -> Pipeline:
    """Start a pipeline from a field sampled at ``n_points`` points between two points.

    The sugar for the common ``uniform`` probe; use :class:`LineField` directly
    for a cloud, a polyline or a circle::

        line("U", (0.05, 0.0, 0.005), (0.05, 0.1, 0.005), 20) | Mag() | Rows()
    """
    return Pipeline(
        source=LineField(field=field, set_type="uniform", start=start, end=end, n_points=n_points)
    )
