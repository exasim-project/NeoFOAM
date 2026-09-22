# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Aggregators — the terminal nodes that reduce a dataset to table rows."""

from __future__ import annotations

from typing import Any, Callable, Literal, Optional, Union

import numpy as np

from neofoam import neofoam_bindings as nfb  # NeoFOAM Python bindings
from neofoam.postprocess._reduce import reduce_max, reduce_min, reduce_sum
from neofoam.postprocess.node import (
    AggregatedData,
    AggregatedDataSet,
    FieldDataSets,
    InternalDataSet,
    Node,
    PatchDataSet,
    SurfaceDataSet,
)
from neofoam.postprocess.nodes._arrays import ones_like

#: The post-processing kernels: element-wise math and reductions that run on the
#: executor the values live on, host numpy included.
pp = nfb.postprocess

#: The column a binner's index fills; the writer puts it before the values.
BIN_COLUMN = "bin"

GREAT = 1.0e15
"""The sentinel an aggregator writes for a group with no active element (``Foam::GREAT``)."""

_SMALL = 1.0e-15


def group_sum(dataset: FieldDataSets, weights: Optional[Any]) -> "np.ndarray[Any, Any]":
    """Per-group, mask- and weight-scaled sum of a dataset, reduced over all ranks.

    The one kernel call every additive aggregator shares (``Sum``, ``Mean``,
    ``VolIntegrate``, ``SurfIntegrate``); pass ``weights=None`` for a plain sum
    or the geometry's measure for an integral. The sum runs where the values
    live — the weights, the mask and the groups have to be on that executor too
    — and comes back as host numpy, ``(n_groups,)`` for a scalar field and
    ``(n_groups, 3)`` for a vector one. A masked-out element contributes zero
    rather than being dropped::

        group_sum(dataset, weights=dataset.geometry.volumes())
    """
    _reject_groups_outside_the_bins(dataset)
    local = pp.sum(dataset.field, dataset.n_groups, dataset.mask, dataset.groups, scaling=weights)
    return reduce_sum(np.asarray(local, dtype=float))


def _group_extremum(dataset: FieldDataSets, kernel: Callable[..., Any]) -> "np.ndarray[Any, Any]":
    """Per-group ``pp.max``/``pp.min`` over the active elements, as host numpy."""
    _reject_groups_outside_the_bins(dataset)
    local = kernel(dataset.field, dataset.n_groups, dataset.mask, dataset.groups)
    return np.asarray(local, dtype=float)


def _reject_groups_outside_the_bins(dataset: FieldDataSets) -> None:
    """Refuse a bin index the binner never declared, which the kernel would write past.

    A row count that follows the data instead of the binner's spec would differ
    between ranks (R6). Only a host group array is checked: a NeoN one comes
    from ``pp.bin_index``, which places every element in a bin it was given, and
    reading it back would copy the whole field off its executor every step.
    """
    groups = dataset.groups
    if not isinstance(groups, np.ndarray) or groups.size == 0:
        return
    if groups.min() < 0 or groups.max() >= dataset.n_groups:
        raise ValueError(
            f"postProcess: {dataset.name!r} has group indices "
            f"[{groups.min()}, {groups.max()}] outside the {dataset.n_groups} groups "
            "the binner declared"
        )


def _measure(dataset: FieldDataSets, aggregator: str, accessor: str, quantity: str) -> Any:
    """The geometry's per-element measure, or a TypeError naming the source that lacks one."""
    measure = getattr(dataset.geometry, accessor, None)
    if measure is None:
        raise TypeError(
            f"{aggregator} needs {quantity}; the source of {dataset.name!r} "
            f"({type(dataset.geometry).__name__}) provides none"
        )
    return measure()


def _to_aggregated(
    label: str, dataset: FieldDataSets, per_group_values: "np.ndarray[Any, Any]"
) -> AggregatedDataSet:
    """Turn a ``(n_groups,)`` / ``(n_groups, 3)`` result into one row per group."""
    values = np.asarray(per_group_values, dtype=float)
    grouped = dataset.groups is not None
    return AggregatedDataSet(
        name=label,
        values=[
            AggregatedData(
                value=float(row) if values.ndim == 1 else [float(value) for value in row],
                group=[float(group)] if grouped else None,
                group_name=[BIN_COLUMN] if grouped else None,
            )
            for group, row in enumerate(values)
        ],
    )


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

    def compute(self, dataset: FieldDataSets) -> AggregatedDataSet:
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

    def compute(self, dataset: FieldDataSets) -> AggregatedDataSet:
        total = group_sum(dataset, weights=None)
        # the divisor is the same sum over a field of ones: the weight each
        # element carries into `total`, reduced over the same mask and bins.
        counts = group_sum(dataset.with_field(ones_like(dataset.field)), weights=None)
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

    def compute(self, dataset: FieldDataSets) -> AggregatedDataSet:
        label = self.name or f"{dataset.name}_max"
        return _to_aggregated(label, dataset, reduce_max(_group_extremum(dataset, pp.max)))


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

    def compute(self, dataset: FieldDataSets) -> AggregatedDataSet:
        label = self.name or f"{dataset.name}_min"
        return _to_aggregated(label, dataset, reduce_min(_group_extremum(dataset, pp.min)))


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

    def compute(self, dataset: InternalDataSet) -> AggregatedDataSet:
        measure = _measure(dataset, "volIntegrate", "volumes", "cell volumes")
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

    def compute(self, dataset: Union[PatchDataSet, SurfaceDataSet]) -> AggregatedDataSet:
        measure = _measure(dataset, "surfIntegrate", "face_area_magnitudes", "face areas")
        label = self.name or f"{dataset.name}_surfIntegrate"
        return _to_aggregated(label, dataset, group_sum(dataset, weights=measure))
