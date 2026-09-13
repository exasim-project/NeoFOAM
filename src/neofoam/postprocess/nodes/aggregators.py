# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Aggregators — the terminal nodes that reduce a DataSet to CSV rows."""

from __future__ import annotations

from typing import Any, Literal, Optional

import numpy as np

from neofoam.postprocess._reduce import reduce_max, reduce_min, reduce_sum
from neofoam.postprocess.node import AggregatedDataSet, DataSet, Node

GREAT = 1.0e15
"""The sentinel an aggregator writes for a group with no active element (``Foam::GREAT``)."""

_SMALL = 1.0e-15


def group_sum(
    dataset: DataSet, weights: Optional["np.ndarray[Any, Any]"]
) -> "np.ndarray[Any, Any]":
    """Per-group, mask- and weight-scaled sum of a DataSet, reduced over all ranks.

    The one kernel every additive aggregator shares (``Sum``, ``Mean``,
    ``VolIntegrate``, ``SurfIntegrate``); pass ``weights=None`` for a plain sum
    or the geometry's measure for an integral. A masked-out element contributes
    zero rather than being dropped, so the result is ``(n_groups,)`` for a scalar
    field and ``(n_groups, 3)`` for a vector one whatever the mask says::

        group_sum(dataset, weights=dataset.geometry.measure)
    """
    values = np.asarray(dataset.values, dtype=float)
    groups = _group_index(dataset, len(values))
    scale = _scale(dataset, weights, len(values))
    if values.ndim == 1:
        local = np.bincount(groups, weights=values * scale, minlength=dataset.n_groups)
    else:
        local = np.stack(
            [
                np.bincount(groups, weights=values[:, c] * scale, minlength=dataset.n_groups)
                for c in range(values.shape[1])
            ],
            axis=1,
        )
    return reduce_sum(local)


def _group_index(dataset: DataSet, n_elements: int) -> "np.ndarray[Any, Any]":
    """The bin index of every element; all zeros when the pipeline holds no binner."""
    if dataset.groups is None:
        return np.zeros(n_elements, dtype=np.int64)
    groups = np.asarray(dataset.groups, dtype=np.int64)
    # A row count that follows the data instead of the binner's spec would differ
    # between ranks (R6), so an out-of-range index is an error, not a wider table.
    if groups.size and (groups.min() < 0 or groups.max() >= dataset.n_groups):
        raise ValueError(
            f"postProcess: {dataset.name!r} has group indices "
            f"[{groups.min()}, {groups.max()}] outside the {dataset.n_groups} groups "
            "the binner declared"
        )
    return groups


def _scale(
    dataset: DataSet, weights: Optional["np.ndarray[Any, Any]"], n_elements: int
) -> "np.ndarray[Any, Any]":
    """The per-element factor: the weights (or one), zeroed where the mask is false."""
    scale = np.ones(n_elements) if weights is None else np.asarray(weights, dtype=float)
    if dataset.mask is None:
        return scale
    return scale * np.asarray(dataset.mask, dtype=float)


def _active_mask(dataset: DataSet, n_elements: int) -> "np.ndarray[Any, Any]":
    """The elements an extremum looks at — masked-out ones are skipped, not scaled."""
    if dataset.mask is None:
        return np.ones(n_elements, dtype=bool)
    return np.asarray(dataset.mask, dtype=bool)


def _group_extremum(dataset: DataSet, ufunc: "np.ufunc", empty: float) -> "np.ndarray[Any, Any]":
    """Per-group min/max over the active elements; ``empty`` where a group has none."""
    values = np.asarray(dataset.values, dtype=float)
    groups = _group_index(dataset, len(values))
    active = _active_mask(dataset, len(values))
    shape = (dataset.n_groups,) if values.ndim == 1 else (dataset.n_groups, values.shape[1])
    local = np.full(shape, empty, dtype=float)
    ufunc.at(local, groups[active], values[active])
    return local


def _measure(dataset: DataSet, aggregator: str, quantity: str) -> "np.ndarray[Any, Any]":
    """The geometry's per-element measure, or a TypeError naming the source that lacks one."""
    measure: Optional["np.ndarray[Any, Any]"] = dataset.geometry.measure
    if measure is None:
        raise TypeError(
            f"{aggregator} needs {quantity}; the source of {dataset.name!r} "
            f"({type(dataset.geometry).__name__}) provides none"
        )
    return measure


def _to_aggregated(
    label: str, dataset: DataSet, per_group_values: "np.ndarray[Any, Any]"
) -> AggregatedDataSet:
    """Turn a ``(n_groups,)`` / ``(n_groups, 3)`` result into the writer's headers and rows."""
    values = np.asarray(per_group_values, dtype=float)
    n_components = 1 if values.ndim == 1 else values.shape[1]
    columns = [label] if n_components == 1 else [f"{label}_{i}" for i in range(n_components)]
    grouped = dataset.groups is not None
    rows = [
        ([float(group)] if grouped else []) + [float(value) for value in np.atleast_1d(row)]
        for group, row in enumerate(values)
    ]
    return AggregatedDataSet(name=label, headers=(["bin"] if grouped else []) + columns, rows=rows)


@Node.register
class Sum(Node):
    """Plain sum of the values, one row per bin.

    The aggregator for an extensive quantity that already carries its measure
    (a flux, a force); use :class:`VolIntegrate` or :class:`SurfIntegrate` when
    the cell volume or face area still has to be applied. Masked-out elements
    contribute zero::

        field("p") | Box(min=(0, 0, 0), max=(1, 1, 1)) | Sum(name="p_box")
    """

    type: Literal["sum"] = "sum"
    name: Optional[str] = None

    def compute(self, dataset: DataSet) -> AggregatedDataSet:
        label = self.name or f"{dataset.name}_sum"
        return _to_aggregated(label, dataset, group_sum(dataset, weights=None))


@Node.register
class Mean(Node):
    """Arithmetic mean of the active values, one row per bin.

    Every element counts the same — for a volume-weighted average divide a
    ``VolIntegrate`` of the field by one of a unit field. A bin with no active
    element yields ``GREAT`` (1e15), the value OpenFOAM uses for "no data"::

        field("T") | Mean(name="T_mean")
    """

    type: Literal["mean"] = "mean"
    name: Optional[str] = None

    def compute(self, dataset: DataSet) -> AggregatedDataSet:
        total = group_sum(dataset, weights=None)
        counts = group_sum(dataset.with_values(np.ones(len(dataset.values))), weights=None)
        divisor = counts if total.ndim == 1 else counts[:, None]
        populated = divisor > _SMALL
        # np.where alone still evaluates both branches, so the division itself has
        # to see a non-zero divisor or an empty bin raises a floating point trap.
        mean = np.where(populated, total / np.where(populated, divisor, 1.0), GREAT)
        label = self.name or f"{dataset.name}_mean"
        return _to_aggregated(label, dataset, mean)


@Node.register
class Max(Node):
    """Largest active value, component-wise for a vector field, one row per bin.

    Masked-out elements are skipped rather than counted as zero, so a selector
    upstream never drags the maximum down. A bin with no active element yields
    ``-GREAT`` (-1e15)::

        field("U") | Mag() | Max(name="U_max")
    """

    type: Literal["max"] = "max"
    name: Optional[str] = None

    def compute(self, dataset: DataSet) -> AggregatedDataSet:
        label = self.name or f"{dataset.name}_max"
        local = _group_extremum(dataset, np.maximum, -GREAT)
        return _to_aggregated(label, dataset, reduce_max(local))


@Node.register
class Min(Node):
    """Smallest active value, component-wise for a vector field, one row per bin.

    Masked-out elements are skipped rather than counted as zero, so a selector
    upstream never drags the minimum up. A bin with no active element yields
    ``GREAT`` (1e15)::

        field("p") | Min(name="p_min")
    """

    type: Literal["min"] = "min"
    name: Optional[str] = None

    def compute(self, dataset: DataSet) -> AggregatedDataSet:
        label = self.name or f"{dataset.name}_min"
        local = _group_extremum(dataset, np.minimum, GREAT)
        return _to_aggregated(label, dataset, reduce_min(local))


@Node.register
class VolIntegrate(Node):
    """Cell-volume-weighted sum: the volume integral over the cells.

    The terminal node of a table; needs a geometry that provides a measure (the
    cell source does, a point cloud does not). A vector field yields one column
    per component::

        field("alpha.water") | VolIntegrate(name="water_volume")
    """

    type: Literal["volIntegrate"] = "volIntegrate"
    name: Optional[str] = None

    def compute(self, dataset: DataSet) -> AggregatedDataSet:
        measure = _measure(dataset, "volIntegrate", "cell volumes")
        label = self.name or f"{dataset.name}_volIntegrate"
        return _to_aggregated(label, dataset, group_sum(dataset, weights=measure))


@Node.register
class SurfIntegrate(Node):
    """Face-area-weighted sum: the surface integral over the faces.

    The counterpart of :class:`VolIntegrate` for a face geometry (a patch, a
    sampled plane, an iso-surface), whose measure is the face area magnitude::

        patch("U") | SurfIntegrate(name="wall_force")
    """

    type: Literal["surfIntegrate"] = "surfIntegrate"
    name: Optional[str] = None

    def compute(self, dataset: DataSet) -> AggregatedDataSet:
        measure = _measure(dataset, "surfIntegrate", "face areas")
        label = self.name or f"{dataset.name}_surfIntegrate"
        return _to_aggregated(label, dataset, group_sum(dataset, weights=measure))
