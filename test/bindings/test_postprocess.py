# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Parity of the NeoN post-processing kernels with the numpy maths they replace.

Every kernel in ``neofoam_bindings.postprocess`` has two front doors — NeoN
vectors in, and host arrays in — and one shared device implementation. The
expectation here is a numpy recomputation rather than a literal: the kernels
are pure maths over random data, so a hand-written table would either be a
transcription of that same formula or too small to catch a per-group or
per-component mix-up. The reference is numpy itself (``bincount``,
``ufunc.at``, ``digitize``), never the code under test, and the data is drawn
from a seeded generator so a failure reproduces.

``n_groups`` deliberately exceeds the largest group index so the empty-group
sentinels (``-GREAT`` for a maximum, ``GREAT`` for a minimum) are exercised on
every aggregator case.

Tolerances: the segmented sums run as Kokkos atomics, so the summation order
differs from ``bincount``'s; ~16 float64 additions per group put the difference
at a few 1e-15 relative, and ``rtol=1e-12`` leaves three decades of headroom.
The extrema and the masks select or compute exact values and are compared bit
for bit.
"""

from __future__ import annotations

from typing import Any, Optional, Union

import neon
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from neofoam.neofoam_bindings import postprocess

GREAT = 1.0e15

RNG = np.random.default_rng(20260922)
N_ELEMENTS = 64
N_GROUPS = 6

SCALAR_VALUES = RNG.normal(size=N_ELEMENTS)
VECTOR_VALUES = RNG.normal(size=(N_ELEMENTS, 3))
POSITIONS = RNG.uniform(-1.0, 1.0, size=(N_ELEMENTS, 3))
MASK = (RNG.random(N_ELEMENTS) > 0.3).astype(np.int32)
OTHER_MASK = (RNG.random(N_ELEMENTS) > 0.5).astype(np.int32)
# Indices stop below N_GROUPS so the last two groups stay empty.
GROUP = RNG.integers(0, 4, size=N_ELEMENTS).astype(np.int32)
SCALING = RNG.random(N_ELEMENTS) + 0.5

VALUES = {"scalar": SCALAR_VALUES, "vector": VECTOR_VALUES}

SELECTIONS = [
    pytest.param(False, False, id="plain"),
    pytest.param(True, False, id="mask"),
    pytest.param(False, True, id="group"),
    pytest.param(True, True, id="mask+group"),
]

# A direction of (0, -4, 0) normalises to (0, -1, 0) exactly, so the signed distance reduces to
# -(y - origin_y) with no rounding on either side of the comparison. Only then is "exactly on an
# edge" a meaningful case: on an oblique direction the dot product is a sum of three rounded
# products and the reference would land on the edge only by luck.
EDGE_DISTANCES = np.array([-1.25, -0.5, 0.75, 2.5, -3.0, 0.0, 4.0])
ON_EDGE_ORIGIN = (0.25, -1.5, 2.0)
ON_EDGE_POSITIONS = np.column_stack(
    [
        np.full(EDGE_DISTANCES.size, 0.25),
        -EDGE_DISTANCES + ON_EDGE_ORIGIN[1],
        np.full(EDGE_DISTANCES.size, 2.0),
    ]
)

BIN_CASES = [
    pytest.param(
        ON_EDGE_POSITIONS,
        (0.0, -4.0, 0.0),
        ON_EDGE_ORIGIN,
        [-1.25, -0.5, 0.75, 2.5],
        id="distances-on-the-edges",
    ),
    pytest.param(
        POSITIONS,
        (1.0, 2.0, -0.5),
        (0.1, -0.2, 0.3),
        [-0.8, -0.15, 0.05, 0.9],
        id="oblique-non-uniform-edges",
    ),
    pytest.param(
        POSITIONS,
        (-2.0, 0.5, 3.0),
        (0.0, 0.0, 0.0),
        [-1.0, 0.0, 1.7],
        id="oblique-at-the-origin",
    ),
    pytest.param(POSITIONS, (0.0, 0.0, 2.0), (0.0, 0.0, -0.4), [0.5], id="single-edge"),
]


def _neon_values(values: "np.ndarray[Any, Any]") -> Any:
    exec_ = neon.SerialExecutor()
    if values.ndim == 1:
        return neon.ScalarVector(exec_, [float(value) for value in values])
    return neon.VectorVector(exec_, [neon.Vec3(*row) for row in values])


def _neon_labels(labels: "np.ndarray[Any, Any]") -> Any:
    return neon.LabelVector(neon.SerialExecutor(), [int(label) for label in labels])


def _values(values: "np.ndarray[Any, Any]", front_door: str) -> Any:
    return values if front_door == "numpy" else _neon_values(values)


def _labels(labels: Optional["np.ndarray[Any, Any]"], front_door: str) -> Any:
    if labels is None:
        return None
    return labels if front_door == "numpy" else _neon_labels(labels)


def _scalars(values: Optional["np.ndarray[Any, Any]"], front_door: str) -> Any:
    if values is None:
        return None
    return values if front_door == "numpy" else _neon_values(values)


def _host(result: Any) -> "np.ndarray[Any, Any]":
    """The result of either front door as a numpy array."""
    return np.asarray(result)


def _reference_sum(
    values: "np.ndarray[Any, Any]",
    mask: Optional["np.ndarray[Any, Any]"],
    group: Optional["np.ndarray[Any, Any]"],
    scaling: Optional["np.ndarray[Any, Any]"],
) -> "np.ndarray[Any, Any]":
    weight = np.ones(len(values))
    if mask is not None:
        weight = weight * mask
    if scaling is not None:
        weight = weight * scaling
    groups = np.zeros(len(values), dtype=np.int64) if group is None else group.astype(np.int64)
    if values.ndim == 1:
        return np.bincount(groups, weights=values * weight, minlength=N_GROUPS)
    return np.stack(
        [np.bincount(groups, weights=values[:, c] * weight, minlength=N_GROUPS) for c in range(3)],
        axis=1,
    )


def _reference_extremum(
    values: "np.ndarray[Any, Any]",
    mask: Optional["np.ndarray[Any, Any]"],
    group: Optional["np.ndarray[Any, Any]"],
    ufunc: "np.ufunc",
    empty: float,
) -> "np.ndarray[Any, Any]":
    active = np.ones(len(values), dtype=bool) if mask is None else mask.astype(bool)
    groups = np.zeros(len(values), dtype=np.int64) if group is None else group.astype(np.int64)
    shape: Union[tuple[int], tuple[int, int]] = (N_GROUPS,) if values.ndim == 1 else (N_GROUPS, 3)
    expected = np.full(shape, empty, dtype=float)
    ufunc.at(expected, groups[active], values[active])
    return expected


@pytest.mark.parametrize("front_door", ["numpy", "neon"])
@pytest.mark.parametrize("kind", ["scalar", "vector"])
@pytest.mark.parametrize("use_mask, use_group", SELECTIONS)
@pytest.mark.parametrize("use_scaling", [False, True])
def test_sum_matches_bincount(
    front_door: str, kind: str, use_mask: bool, use_group: bool, use_scaling: bool
) -> None:
    values = VALUES[kind]
    mask = MASK if use_mask else None
    group = GROUP if use_group else None
    scaling = SCALING if use_scaling else None

    result = postprocess.sum(
        _values(values, front_door),
        N_GROUPS,
        _labels(mask, front_door),
        _labels(group, front_door),
        scaling=_scalars(scaling, front_door),
    )

    assert_allclose(
        _host(result),
        _reference_sum(values, mask, group, scaling),
        rtol=1e-12,
        atol=1e-12,
        err_msg=f"sum({kind}) via {front_door}",
    )


@pytest.mark.parametrize("front_door", ["numpy", "neon"])
@pytest.mark.parametrize("kind", ["scalar", "vector"])
@pytest.mark.parametrize("use_mask, use_group", SELECTIONS)
def test_max_matches_ufunc_maximum(
    front_door: str, kind: str, use_mask: bool, use_group: bool
) -> None:
    values = VALUES[kind]
    mask = MASK if use_mask else None
    group = GROUP if use_group else None

    result = postprocess.max(
        _values(values, front_door),
        N_GROUPS,
        _labels(mask, front_door),
        _labels(group, front_door),
    )

    assert_array_equal(
        _host(result),
        _reference_extremum(values, mask, group, np.maximum, -GREAT),
        err_msg=f"max({kind}) via {front_door}",
    )


@pytest.mark.parametrize("front_door", ["numpy", "neon"])
@pytest.mark.parametrize("kind", ["scalar", "vector"])
@pytest.mark.parametrize("use_mask, use_group", SELECTIONS)
def test_min_matches_ufunc_minimum(
    front_door: str, kind: str, use_mask: bool, use_group: bool
) -> None:
    values = VALUES[kind]
    mask = MASK if use_mask else None
    group = GROUP if use_group else None

    result = postprocess.min(
        _values(values, front_door),
        N_GROUPS,
        _labels(mask, front_door),
        _labels(group, front_door),
    )

    assert_array_equal(
        _host(result),
        _reference_extremum(values, mask, group, np.minimum, GREAT),
        err_msg=f"min({kind}) via {front_door}",
    )


@pytest.mark.parametrize("front_door", ["numpy", "neon"])
def test_mag_matches_linalg_norm(front_door: str) -> None:
    result = postprocess.mag(_values(VECTOR_VALUES, front_door))

    assert_allclose(
        _host(result),
        np.linalg.norm(VECTOR_VALUES, axis=1),
        rtol=1e-14,
        atol=0.0,
        err_msg=f"mag via {front_door}",
    )


@pytest.mark.parametrize("front_door", ["numpy", "neon"])
@pytest.mark.parametrize("index", [0, 1, 2])
def test_component_picks_one_column(front_door: str, index: int) -> None:
    result = postprocess.component(_values(VECTOR_VALUES, front_door), index)

    assert_array_equal(
        _host(result),
        VECTOR_VALUES[:, index],
        err_msg=f"component({index}) via {front_door}",
    )


@pytest.mark.parametrize("front_door", ["numpy", "neon"])
@pytest.mark.parametrize("kind", ["scalar", "vector"])
def test_scale_multiplies_every_value(front_door: str, kind: str) -> None:
    result = postprocess.scale(_values(VALUES[kind], front_door), 2.5)

    assert_array_equal(
        _host(result),
        VALUES[kind] * 2.5,
        err_msg=f"scale({kind}) via {front_door}",
    )


@pytest.mark.parametrize("front_door", ["numpy", "neon"])
def test_box_mask_includes_the_bounds(front_door: str) -> None:
    lo = (-0.5, -0.25, -1.0)
    hi = (0.5, 0.75, 1.0)

    result = postprocess.box_mask(_values(POSITIONS, front_door), lo, hi)

    expected = np.all((POSITIONS >= lo) & (POSITIONS <= hi), axis=1).astype(np.int32)
    assert_array_equal(_host(result), expected, err_msg=f"box_mask via {front_door}")


@pytest.mark.parametrize("front_door", ["numpy", "neon"])
def test_sphere_mask_includes_the_boundary(front_door: str) -> None:
    centre = (0.1, -0.2, 0.3)
    radius = 0.8

    result = postprocess.sphere_mask(_values(POSITIONS, front_door), centre, radius)

    expected = (np.linalg.norm(POSITIONS - np.asarray(centre), axis=1) <= radius).astype(np.int32)
    assert_array_equal(_host(result), expected, err_msg=f"sphere_mask via {front_door}")


@pytest.mark.parametrize("front_door", ["numpy", "neon"])
def test_mask_and_intersects(front_door: str) -> None:
    result = postprocess.mask_and(_labels(MASK, front_door), _labels(OTHER_MASK, front_door))

    assert_array_equal(
        _host(result),
        (MASK.astype(bool) & OTHER_MASK.astype(bool)).astype(np.int32),
        err_msg=f"mask_and via {front_door}",
    )


@pytest.mark.parametrize("front_door", ["numpy", "neon"])
def test_mask_or_unions(front_door: str) -> None:
    result = postprocess.mask_or(_labels(MASK, front_door), _labels(OTHER_MASK, front_door))

    assert_array_equal(
        _host(result),
        (MASK.astype(bool) | OTHER_MASK.astype(bool)).astype(np.int32),
        err_msg=f"mask_or via {front_door}",
    )


@pytest.mark.parametrize("front_door", ["numpy", "neon"])
def test_mask_not_inverts(front_door: str) -> None:
    result = postprocess.mask_not(_labels(MASK, front_door))

    assert_array_equal(
        _host(result),
        (~MASK.astype(bool)).astype(np.int32),
        err_msg=f"mask_not via {front_door}",
    )


@pytest.mark.parametrize("front_door", ["numpy", "neon"])
@pytest.mark.parametrize("positions, direction, origin, edges", BIN_CASES)
def test_bin_index_matches_digitize(
    front_door: str,
    positions: "np.ndarray[Any, Any]",
    direction: tuple[float, float, float],
    origin: tuple[float, float, float],
    edges: list[float],
) -> None:
    result = postprocess.bin_index(_values(positions, front_door), direction, origin, edges)

    normal = np.asarray(direction, dtype=float) / np.linalg.norm(direction)
    distance = (positions - np.asarray(origin, dtype=float)) @ normal
    expected = np.digitize(distance, np.asarray(edges, dtype=float)).astype(np.int32)
    assert_array_equal(_host(result), expected, err_msg=f"bin_index via {front_door}")


@pytest.mark.parametrize("front_door", ["numpy", "neon"])
def test_mask_host_arrays_are_int32(front_door: str) -> None:
    result = postprocess.mask_not(_labels(MASK, front_door))

    assert _host(result).dtype == np.int32
