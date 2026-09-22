# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

# NB: no ``from __future__ import annotations`` — Context injection matches
# ``param.annotation is Context`` literally (see field_writer.py).

"""The post-processing data contract: the datasets, the Source/Node families and Pipeline.

One dataset class per geometry kind (cells, a patch, a sampled surface, a set of
points), each holding the field, the active-element mask and the bins, the way
pyOFTools shapes them. A node does its arithmetic with the NeoN kernels in
``neofoam.neofoam_bindings.postprocess``, so ``field``, ``mask``, ``groups`` and
the geometry's arrays are either host numpy (a pybFoam field, every sampling
source) or NeoN vectors on the executor a NeoN field lives on — the kernels take
both, as long as one call sees one executor. The backend libraries themselves
stay in the sources, the writers and the reductions.

Both extension points are :class:`~neofoam.core.plugin_system.PluginSystem`
families discriminated by ``type``: a new source or node is a new
``@Source.register`` / ``@Node.register`` class, never an edit here.

Compose a table with ``|``::

    field("p") | VolIntegrate(name="volume_p")
"""

from typing import Any, ClassVar, Optional, Protocol, TypeVar, Union, runtime_checkable

import numpy as np
from pydantic import BaseModel, ConfigDict, SerializeAsAny, model_validator

from neofoam.core.plugin_system import PluginSystem
from neofoam.framework.context import Context
from neofoam.io import BaseConfig


@runtime_checkable
class InternalMesh(Protocol):
    """The cells a volume field lives on — what an internal dataset is measured against."""

    def positions(self) -> Any:
        """Cell centres, ``(n, 3)``."""

    def volumes(self) -> Any:
        """Cell volumes, ``(n,)``."""


@runtime_checkable
class BoundaryMesh(Protocol):
    """One boundary patch: its face centres and the area of each face."""

    def positions(self) -> Any:
        """Face centres, ``(n, 3)``."""

    def face_area_magnitudes(self) -> Any:
        """Face areas, ``(n,)``."""


@runtime_checkable
class SurfaceMesh(Protocol):
    """A sampled surface: face centres, face area vectors and their magnitudes."""

    def positions(self) -> Any:
        """Face centres, ``(n, 3)``."""

    def face_areas(self) -> Any:
        """Face area vectors, ``(n, 3)``."""

    def face_area_magnitudes(self) -> Any:
        """Face areas, ``(n,)``."""

    def total_area(self) -> float:
        """The area of the whole surface."""


@runtime_checkable
class SetGeometry(Protocol):
    """A set of sample points: where they are, and how far along the set each one is.

    A probe has no area or volume, which is why the measure-weighted
    aggregators have nothing to ask it for.
    """

    def positions(self) -> Any:
        """Sample points, ``(n, 3)``."""

    def distance(self) -> Any:
        """Distance along the set, ``(n,)``."""


@runtime_checkable
class SamplingGeometry(Protocol):
    """A geometry that can also interpolate a registered field onto its elements.

    What a sampled surface, a boundary patch and a point set have over the
    cells, and what :class:`~neofoam.postprocess.sources.sampling.Sample` needs: the
    node holds a field *name*, never a live mesh, so the geometry keeps the
    field registry and the interpolation scheme. Implement it for a new
    samplable geometry; leave it off (as
    :class:`~neofoam.postprocess.sources.geometry.CellGeometry` does) and ``Sample``
    refuses the pipeline by name.
    """

    def sample(self, name: str) -> Any:
        """The registered field *name*, interpolated onto the elements."""


_DataSetT = TypeVar("_DataSetT", bound="_FieldDataSet")


class _FieldDataSet(BaseModel):
    """What every field dataset carries, whatever geometry it sits on."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    name: str
    field: Any  # host numpy or a NeoN vector, (n,) or (n, 3)
    mask: Optional[Any] = None  # 0/1 labels (n,); None = every element active
    groups: Optional[Any] = None  # labels (n,), the bin index per element; None = one group
    n_groups: int = 1  # from the binner's spec, so every rank agrees on the row count

    def with_field(self: _DataSetT, field: Any) -> _DataSetT:
        """The same dataset with new values."""
        return self.model_copy(update={"field": field})

    def with_mask(self: _DataSetT, mask: Any) -> _DataSetT:
        """The same dataset with a new active-element mask (already ANDead by the caller)."""
        return self.model_copy(update={"mask": mask})

    def with_groups(self: _DataSetT, groups: Any, n_groups: int) -> _DataSetT:
        """The same dataset binned into ``n_groups`` groups."""
        return self.model_copy(update={"groups": groups, "n_groups": n_groups})


class InternalDataSet(_FieldDataSet):
    """A volume field on the cells — what the ``internal`` source hands its nodes.

    A node returns a *new* dataset (use :meth:`with_field`, :meth:`with_mask`,
    :meth:`with_groups`) and never mutates its input, so the same pipeline can
    be evaluated every time step. Every array of one dataset is of the same kind
    and on the same executor, because that is what a kernel call takes::

        dataset.with_field(pp.scale(dataset.field, 2.0))
    """

    geometry: InternalMesh

    @classmethod
    def from_field(cls, name: str, field: Any, mesh: Any) -> "InternalDataSet":
        """The internal values of a volume field, on the cells they sit on.

        The one place that branches on the field library, so nothing downstream
        has to: a pybFoam field hands its cell values over as host numpy, a NeoN
        one keeps them in a ``Vector`` on its executor and stays there — the
        geometry is then the same cells mirrored onto that executor, which is
        what lets a GPU field be reduced where it lives::

            InternalDataSet.from_field("p", ctx.fields["p"], ctx.mesh)
        """
        from neofoam.postprocess.sources.geometry import (  # noqa: PLC0415  # cycle: sources->node
            CellGeometry,
            neon_cell_geometry,
        )

        internal_field = getattr(field, "internalField", None)
        if internal_field is not None:
            return cls(name=name, field=np.asarray(internal_field()), geometry=CellGeometry(mesh))
        values = field.internal_vector()
        return cls(name=name, field=values, geometry=neon_cell_geometry(mesh, values.exec()))


class PatchDataSet(_FieldDataSet):
    """A field on the faces of one boundary patch — the ``patch`` source's dataset."""

    geometry: BoundaryMesh


class SurfaceDataSet(_FieldDataSet):
    """A field on the faces of a sampled surface — a cutting plane or an iso-surface."""

    geometry: SurfaceMesh


class PointDataSet(_FieldDataSet):
    """A field at a set of sample points — a probe, which carries no measure."""

    geometry: SetGeometry


#: The datasets that carry a field plus its mask and bins: what a node transforms.
FieldDataSets = Union[InternalDataSet, PatchDataSet, SurfaceDataSet, PointDataSet]


def _flatten(value: Any) -> list[float]:
    """One aggregated value as the columns it fills — one per vector component."""
    if hasattr(value, "__len__"):
        return [float(component) for component in value]
    return [float(value)]


class AggregatedData(BaseModel):
    """One row of an aggregation: the value, and the labels that say which one it is.

    ``group`` holds the labels themselves (the bin index, or a residual's field
    and solver) and ``group_name`` their column names; both are ``None`` for a
    table that reduces to a single row.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    value: Any
    group: Optional[list[Any]] = None
    group_name: Optional[list[str]] = None


class AggregatedDataSet(BaseModel):
    """An aggregation result: the rows a table appends at this write step.

    The terminal value of a pipeline — what a
    :class:`~neofoam.postprocess.writers.writer.TableWriter` turns into
    time-prefixed lines. :attr:`headers` and :attr:`grouped_values` put the
    value columns first and the group columns after them; a writer that wants
    the group columns first (NeoFOAM's CSV layout does) reorders with
    :func:`~neofoam.postprocess.writers.writer.table_headers` and
    :func:`~neofoam.postprocess.writers.writer.table_rows`. An entry is a number
    for every aggregator; a source that aggregates itself (the solver
    residuals) also puts labels in its groups, which is why a group entry is not
    typed as a float.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    values: list[AggregatedData]

    @property
    def headers(self) -> list[str]:
        """One column name per value component, then one per group label."""
        if not self.values:
            return []
        first = self.values[0]
        if hasattr(first.value, "__len__"):
            columns = [f"{self.name}_{index}" for index in range(len(first.value))]
        else:
            columns = [self.name]
        return [*columns, *(first.group_name or [])]

    @property
    def grouped_values(self) -> list[list[Any]]:
        """One row per aggregated value: its components, then its group labels."""
        return [[*_flatten(value.value), *(value.group or [])] for value in self.values]


#: Everything that may flow between the nodes of a pipeline.
DataSets = Union[FieldDataSets, AggregatedDataSet]


@PluginSystem.register(discriminator_variable="source", discriminator="type")
class Source(BaseConfig):
    """Plugin interface: turn the live Context into a dataset.

    A source is where a table reaches into the simulation backend. Add one with
    a ``@Source.register`` subclass carrying a ``type: Literal[...]``
    discriminator; it is then selectable by that string from a case file. A
    source with no geometry behind it (the solver residuals) may return the
    terminal :class:`AggregatedDataSet` instead, and its pipeline then carries
    no nodes. Unknown keys are refused, so a case file's typo is an error and
    not a default.
    """

    model_config = ConfigDict(extra="forbid")

    #: Whether :meth:`resolve` already returns the terminal
    #: :class:`AggregatedDataSet`; such a source's table carries no nodes, and
    #: :meth:`Pipeline.compute` refuses one that does.
    self_aggregating: ClassVar[bool] = False

    def resolve(self, ctx: Context) -> Union[FieldDataSets, AggregatedDataSet]:
        raise NotImplementedError


@PluginSystem.register(discriminator_variable="node", discriminator="type")
class Node(BaseConfig):
    """Plugin interface: one pipeline step, ``dataset -> dataset``.

    Add a selector, a transform or an aggregator with a ``@Node.register``
    subclass carrying a ``type: Literal[...]`` discriminator. A transform returns
    one of the :data:`FieldDataSets`, an aggregator the terminal
    :class:`AggregatedDataSet`. Unknown keys are refused, so a case file's typo
    is an error and not a default.
    """

    model_config = ConfigDict(extra="forbid")

    #: Whether this node also handles the terminal :class:`AggregatedDataSet`.
    #: Only a pass-through node (a printer) may sit behind an aggregator; for
    #: every other node :meth:`Pipeline.compute` refuses the pipeline.
    accepts_aggregated: ClassVar[bool] = False

    def compute(self, dataset: Any) -> Any:
        raise NotImplementedError


class Pipeline(BaseModel):
    """A source plus ordered nodes — one table's recipe, evaluated per time step.

    Compose with ``|`` (each ``|`` returns a *new* pipeline, so a partial
    pipeline can be branched) and evaluate with :meth:`compute`::

        field("p") | Scale(factor=2.0) | VolIntegrate(name="volume_p")
    """

    # SerializeAsAny: the annotation is the family base, and without it a dump
    # keeps only the base's (empty) fields; the validator is the way back — a
    # dumped ``{"type": ...}`` mapping picks its class from the family's union.
    source: SerializeAsAny[Source]
    steps: list[SerializeAsAny[Node]] = []

    @model_validator(mode="before")
    @classmethod
    def _resolve_the_plugins(cls, data: Any) -> Any:
        """Turn a dumped ``{"type": ...}`` mapping back into the plugin it names."""
        if not isinstance(data, dict):
            return data
        resolved = dict(data)
        if isinstance(data.get("source"), dict):
            resolved["source"] = _created(Source, "source", data["source"])
        if isinstance(data.get("steps"), list):
            resolved["steps"] = [
                _created(Node, "node", step) if isinstance(step, dict) else step
                for step in data["steps"]
            ]
        return resolved

    def __or__(self, node: Node) -> "Pipeline":
        return Pipeline(source=self.source, steps=[*self.steps, node])

    def compute(self, ctx: Context) -> Any:
        """Resolve the source against ``ctx`` and run every step in order."""
        dataset: Any = self.source.resolve(ctx)
        if self.steps and isinstance(dataset, AggregatedDataSet):
            raise TypeError(
                f"postProcess: source {_type_name(self.source)!r} aggregates by itself, "
                f"so nothing may be piped onto it; got node {_type_name(self.steps[0])!r}"
            )
        for index, step in enumerate(self.steps):
            if isinstance(dataset, AggregatedDataSet) and not step.accepts_aggregated:
                raise TypeError(
                    f"postProcess: node {_type_name(self.steps[index - 1])!r} aggregates, so "
                    f"only a pass-through node may follow it; got {_type_name(step)!r}"
                )
            dataset = step.compute(dataset)
        return dataset


def _created(family: Any, variable: str, payload: dict[str, Any]) -> Any:
    """The plugin a ``{"type": ...}`` mapping names, from its family's union."""
    return getattr(family.create(**{variable: payload}), variable)


def _type_name(plugin: Any) -> str:
    """A source's or node's ``type`` discriminator, for an error message."""
    return str(getattr(plugin, "type", type(plugin).__name__))
