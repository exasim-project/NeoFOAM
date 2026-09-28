# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Unit tests for the ``Directional`` binner.

Two rules distinguish it from the pyOFTools node it ports: the direction is
normalised, so the bin edges are distances and the length of the direction
vector cannot silently rescale them, and ``n_groups`` comes from the spec rather
than from the data, so every rank of a parallel run emits the same rows (R6).
Positions are the only mesh data a binner reads, so a fake geometry is the whole
backend here.

The kernel follows ``np.digitize``, which is what the oblique, shifted and
unevenly binned cases below are compared against — a reference the test computes
on the raw positions, never with the node under test. The same case runs on a
NeoN ``VectorVector``, since binning on the executor the field lives on is the
point of the kernel.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from pydantic import ValidationError

from neofoam.postprocess.node import InternalDataSet
from neofoam.postprocess.nodes.binning import Directional

#: Points along x at, between and outside the bin edges 0.0 and 1.0.
POSITIONS = np.array(
    [
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
    ]
)

#: Points with something in every coordinate, for a direction that is not an axis.
SCATTERED = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [-2.0, 0.5, 1.0],
        [3.0, -1.0, 2.0],
        [0.25, 0.25, 0.25],
    ]
)


class FakeGeometry:
    """A geometry: a binner reads only the element positions."""

    def __init__(self, positions: np.ndarray) -> None:
        self._positions = positions

    def positions(self) -> np.ndarray:
        return self._positions

    def volumes(self) -> np.ndarray:
        return np.ones(len(POSITIONS))


def _dataset(positions: np.ndarray = POSITIONS) -> InternalDataSet:
    return InternalDataSet(
        name="p",
        field=np.arange(len(POSITIONS), dtype=float),
        geometry=FakeGeometry(positions),
    )


def _digitized(
    positions: np.ndarray,
    bins: list[float],
    direction: tuple[float, float, float],
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """The bin of every position, computed on the raw arrays as numpy would."""
    normal = np.asarray(direction, dtype=float)
    normal = normal / np.linalg.norm(normal)
    distance = (positions - np.asarray(origin, dtype=float)) @ normal
    return np.digitize(distance, np.asarray(bins, dtype=float))


@pytest.mark.parametrize(
    "bins",
    [[1.0, 0.0], [0.0, 0.0]],
    ids=["decreasing", "repeated"],
)
def test_bin_edges_that_do_not_increase_are_rejected(bins: list[float]) -> None:
    # the kernel follows np.digitize, which would switch to its decreasing
    # convention and reverse the profile with no warning
    with pytest.raises(ValidationError, match="strictly increasing"):
        Directional(bins=bins, direction=(1.0, 0.0, 0.0))


def test_bin_edges_are_distances_along_the_normalised_direction() -> None:
    binner = Directional(bins=[0.0, 1.0], direction=(1.0, 0.0, 0.0))

    result = binner.compute(_dataset())

    assert_array_equal(np.asarray(result.groups), np.array([0, 1, 1, 2, 2]))


def test_a_longer_direction_vector_bins_exactly_like_the_unit_one() -> None:
    scaled = Directional(bins=[0.0, 1.0], direction=(2.0, 0.0, 0.0)).compute(_dataset())

    unit = Directional(bins=[0.0, 1.0], direction=(1.0, 0.0, 0.0)).compute(_dataset())

    assert_array_equal(np.asarray(scaled.groups), np.asarray(unit.groups))


def test_the_origin_shifts_the_distances() -> None:
    binner = Directional(bins=[0.0, 1.0], direction=(1.0, 0.0, 0.0), origin=(1.0, 0.0, 0.0))

    result = binner.compute(_dataset())

    assert_array_equal(np.asarray(result.groups), np.array([0, 0, 0, 1, 2]))


# --- the general spec: any direction, any origin, any increasing edges -----

GENERAL = [
    pytest.param((1.0, 1.0, 0.0), (0.0, 0.0, 0.0), [0.0, 1.0], id="diagonal"),
    pytest.param((1.0, 2.0, -2.0), (0.0, 0.0, 0.0), [-0.5, 0.5, 1.5], id="oblique"),
    pytest.param((1.0, 1.0, 1.0), (0.25, 0.25, 0.25), [0.0, 1.0], id="oblique-shifted-origin"),
    pytest.param((0.0, 1.0, 0.0), (0.0, -0.5, 0.0), [0.0, 0.1, 2.0], id="uneven-edges"),
    pytest.param((1.0, 0.0, 0.0), (0.0, 0.0, 0.0), [-1.5, 0.25, 0.3, 10.0], id="uneven-and-wide"),
]


@pytest.mark.parametrize("direction,origin,bins", GENERAL)
def test_the_bins_are_what_digitize_makes_of_the_signed_distance(
    direction: tuple[float, float, float],
    origin: tuple[float, float, float],
    bins: list[float],
) -> None:
    binner = Directional(bins=bins, direction=direction, origin=origin)

    result = binner.compute(_dataset(SCATTERED))

    assert_array_equal(np.asarray(result.groups), _digitized(SCATTERED, bins, direction, origin))
    assert result.n_groups == len(bins) + 1


@pytest.mark.parametrize("direction,origin,bins", GENERAL)
def test_a_neon_field_is_binned_on_its_executor(
    direction: tuple[float, float, float],
    origin: tuple[float, float, float],
    bins: list[float],
) -> None:
    # the positions of a NeoN dataset are a NeoN vector, and the bins have to
    # come back as one — on the same executor, without a host round trip
    neon = pytest.importorskip("neon")
    executor = neon.SerialExecutor()
    positions = neon.VectorVector(executor, [neon.Vec3(*point) for point in SCATTERED])
    binner = Directional(bins=bins, direction=direction, origin=origin)

    result = binner.compute(_dataset(positions))

    assert isinstance(result.groups, neon.LabelVector)
    assert repr(result.groups.exec()) == repr(executor)
    assert_array_equal(
        np.asarray(result.groups.copy_to_host()), _digitized(SCATTERED, bins, direction, origin)
    )


def test_n_groups_comes_from_the_spec_even_when_the_data_fills_one_bin() -> None:
    binner = Directional(bins=[10.0, 20.0], direction=(1.0, 0.0, 0.0))

    result = binner.compute(_dataset())

    assert_array_equal(np.asarray(result.groups), np.zeros(len(POSITIONS)))
    assert result.n_groups == 3


def test_binning_leaves_the_input_dataset_untouched() -> None:
    dataset = _dataset()

    result = Directional(bins=[0.0], direction=(1.0, 0.0, 0.0)).compute(dataset)

    assert result is not dataset
    assert dataset.groups is None
    assert dataset.n_groups == 1


def test_a_single_edge_splits_the_elements_in_two() -> None:
    result = Directional(bins=[0.5], direction=(1.0, 0.0, 0.0)).compute(_dataset())

    assert_array_equal(np.asarray(result.groups), np.array([0, 0, 1, 1, 1]))
    assert result.n_groups == 2


def test_a_zero_length_direction_is_rejected() -> None:
    with pytest.raises(ValidationError, match="direction must have a non-zero length"):
        Directional(bins=[0.0], direction=(0.0, 0.0, 0.0))
