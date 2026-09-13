# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

# NB: no ``from __future__ import annotations`` — Context injection matches
# ``param.annotation is Context`` literally (see field_writer.py).

"""The post-processing data contract: DataSet, the Source/Node plugin families and Pipeline.

A node sees only numpy through the :class:`Geometry` protocol, so this module —
and every node built on it — is pure numpy; the simulation backend (pybFoam
today, NeoN later) stays in the sources, the writers and the reductions.

Both extension points are :class:`~neofoam.core.plugin_system.PluginSystem`
families discriminated by ``type``: a new source or node is a new
``@Source.register`` / ``@Node.register`` class, never an edit here.

Compose a table with ``|``::

    field("p") | VolIntegrate(name="volume_p")
"""

from typing import Any, ClassVar, Optional, Protocol, Union, runtime_checkable

import numpy as np
from pydantic import BaseModel, ConfigDict, SerializeAsAny, model_validator

from neofoam.core.plugin_system import PluginSystem
from neofoam.framework.context import Context
from neofoam.io import BaseConfig


@runtime_checkable
class Geometry(Protocol):
    """Positions and an optional per-element measure — all a node sees of the mesh.

    Implement it for a new geometry kind (cells, patch faces, sampled points);
    ``measure`` is the cell volume or face area, and ``None`` where no measure
    exists (a point cloud), which the measure-weighted aggregators reject.
    """

    @property
    def positions(self) -> "np.ndarray[Any, Any]":
        """Element centres, shape ``(n, 3)``."""

    @property
    def measure(self) -> "Optional[np.ndarray[Any, Any]]":
        """Per-element volume/area, shape ``(n,)``, or ``None``."""


@runtime_checkable
class SamplingGeometry(Protocol):
    """A :class:`Geometry` that can also interpolate a registered field onto its elements.

    What a sampled surface, a boundary patch and a point set have over the
    cells, and what :class:`~neofoam.postprocess.sources.sampling.Sample` needs: the
    node holds a field *name*, never a live mesh, so the geometry keeps the
    field registry and the interpolation scheme. Implement it for a new
    samplable geometry; leave it off (as
    :class:`~neofoam.postprocess.sources.geometry.CellGeometry` does) and ``Sample``
    refuses the pipeline by name. It is deliberately *not* a subclass of
    :class:`Geometry`: a protocol whose members are all methods is the only kind
    ``issubclass`` accepts, which is how the adapters pin their conformance.
    """

    def sample(self, name: str) -> "np.ndarray[Any, Any]":
        """The registered field *name*, interpolated onto the elements."""


class DataSet(BaseModel):
    """One field on one geometry — what flows between the nodes of a pipeline.

    A node returns a *new* DataSet (use :meth:`with_values`, :meth:`with_mask`,
    :meth:`with_groups`) and never mutates its input, so the same pipeline can
    be evaluated every time step. A selector narrows ``mask``, a binner sets
    ``groups``/``n_groups``, and an aggregator honours both (one row per group)::

        dataset.with_values(np.asarray(dataset.values) * 2.0)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)
    name: str
    values: Any  # numpy (n,) or (n, 3)
    geometry: Any  # Geometry
    mask: Optional[Any] = None  # numpy bool (n,); None = every element active
    groups: Optional[Any] = None  # numpy int64 (n,) bin index per element; None = one group
    n_groups: int = 1  # from the binner's spec, so every rank agrees on the row count

    def with_values(self, values: Any) -> "DataSet":
        """The same dataset with new values."""
        return self.model_copy(update={"values": values})

    def with_mask(self, mask: Any) -> "DataSet":
        """The same dataset with a new active-element mask (already ANDead by the caller)."""
        return self.model_copy(update={"mask": mask})

    def with_groups(self, groups: Any, n_groups: int) -> "DataSet":
        """The same dataset binned into ``n_groups`` groups."""
        return self.model_copy(update={"groups": groups, "n_groups": n_groups})


class AggregatedDataSet(BaseModel):
    """An aggregation result: the CSV column names and the rows to append.

    The terminal value of a pipeline — what a
    :class:`~neofoam.postprocess.writers.writer.TableWriter` turns into time-prefixed
    lines. An ungrouped table has one row and one column per value component
    (``["p_sum"]``, or ``["U_sum_0", "U_sum_1", "U_sum_2"]``); a table behind a
    binner has one row per bin and a leading ``bin`` column holding the bin
    index, so the file reads ``time,bin,p_sum``. Every row has as many entries
    as there are headers. An entry is a number for every aggregator; a source
    that aggregates itself (the solver residuals) also puts labels in its rows,
    which is why a row is not typed as floats.
    """

    name: str
    headers: list[str]
    rows: list[list[Any]]


@PluginSystem.register(discriminator_variable="source", discriminator="type")
class Source(BaseConfig):
    """Plugin interface: turn the live Context into a DataSet.

    A source is where a table reaches into the simulation backend. Add one with
    a ``@Source.register`` subclass carrying a ``type: Literal[...]``
    discriminator; it is then selectable by that string from a case file. A
    source with no geometry behind it (the solver residuals) may return the
    terminal :class:`AggregatedDataSet` instead, and its pipeline then carries
    no nodes. Unknown keys are refused, so a case file's typo is an error and
    not a default.
    """

    model_config = ConfigDict(extra="forbid")

    def resolve(self, ctx: Context) -> Union[DataSet, AggregatedDataSet]:
        raise NotImplementedError


@PluginSystem.register(discriminator_variable="node", discriminator="type")
class Node(BaseConfig):
    """Plugin interface: one pipeline step, ``dataset -> dataset``.

    Add a selector, a transform or an aggregator with a ``@Node.register``
    subclass carrying a ``type: Literal[...]`` discriminator. A transform returns
    a :class:`DataSet`, an aggregator the terminal :class:`AggregatedDataSet`.
    Unknown keys are refused, so a case file's typo is an error and not a
    default.
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
