# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Unit tests for the field functions ``Mag``, ``Component`` and ``Area``.

All three replace the values and rename the dataset — the name is what the
aggregator downstream turns into a CSV header, so it is part of the behaviour.
The values of the fake geometry are chosen so every expectation is an exact
literal (3-4-5 triangles), and each test also proves the transform carries the
mask and the groups of a selector or binner upstream through unchanged.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from pydantic import ValidationError

from neofoam.postprocess.node import InternalDataSet, Node, PointDataSet, SurfaceDataSet
from neofoam.postprocess.nodes.field_functions import Area, Component, Mag

VELOCITY = np.array([[3.0, 4.0, 0.0], [0.0, -6.0, 8.0]])
FACE_AREAS = np.array([0.25, 0.75])
CELL_VOLUMES = np.array([0.125, 0.5])
MASK = np.array([1, 0], dtype=np.int32)
GROUPS = np.array([0, 1], dtype=np.int32)


class FakeSurface:
    """A sampled surface: ``Area`` reads its face areas, a selector its positions."""

    def positions(self) -> np.ndarray:
        return np.zeros((len(VELOCITY), 3))

    def face_areas(self) -> np.ndarray:
        return np.zeros((len(VELOCITY), 3))

    def face_area_magnitudes(self) -> np.ndarray:
        return FACE_AREAS

    def total_area(self) -> float:
        return float(FACE_AREAS.sum())


class FakeCells:
    """The cells: their measure is the volume, not an area."""

    def positions(self) -> np.ndarray:
        return np.zeros((len(VELOCITY), 3))

    def volumes(self) -> np.ndarray:
        return CELL_VOLUMES


class FakePointSet:
    """A probe's geometry — the one with no measure on it at all."""

    def positions(self) -> np.ndarray:
        return np.zeros((len(VELOCITY), 3))

    def distance(self) -> np.ndarray:
        return np.arange(float(len(VELOCITY)))


def _dataset(values: Optional[np.ndarray] = None) -> SurfaceDataSet:
    return SurfaceDataSet(
        name="U",
        field=VELOCITY if values is None else values,
        geometry=FakeSurface(),
        mask=MASK,
        groups=GROUPS,
        n_groups=2,
    )


def test_mag_reduces_a_vector_field_to_its_magnitudes() -> None:
    result = Mag().compute(_dataset())

    assert_allclose(np.asarray(result.field), [5.0, 10.0], rtol=1e-12, err_msg="|U|")


def test_mag_renames_the_dataset_after_the_function() -> None:
    assert Mag().compute(_dataset()).name == "mag(U)"


def test_component_picks_one_component_of_a_vector_field() -> None:
    result = Component(index=1).compute(_dataset())

    assert_allclose(np.asarray(result.field), [4.0, -6.0], rtol=1e-12, err_msg="U_1")


def test_component_renames_the_dataset_after_the_index() -> None:
    assert Component(index=2).compute(_dataset()).name == "U_2"


@pytest.mark.parametrize("index", [-1, 3])
def test_component_rejects_an_index_outside_a_vector(index: int) -> None:
    with pytest.raises(ValidationError):
        Component(index=index)


@pytest.mark.parametrize(
    "node",
    [pytest.param(Mag(), id="mag"), pytest.param(Component(index=0), id="component")],
)
def test_a_vector_function_on_a_scalar_field_names_the_dataset(node: Node) -> None:
    scalar = _dataset(values=np.array([1.0, 2.0]))

    with pytest.raises(TypeError, match=r"needs a vector field; 'U' has scalar values"):
        node.compute(scalar)


def test_area_replaces_the_values_by_the_face_areas() -> None:
    result = Area().compute(_dataset())

    assert_allclose(np.asarray(result.field), FACE_AREAS, rtol=1e-12, err_msg="face areas")
    assert result.name == "area"


def test_area_on_a_geometry_without_a_measure_names_the_source() -> None:
    probe = PointDataSet(name="U", field=VELOCITY, geometry=FakePointSet())

    with pytest.raises(TypeError, match=r"area needs a per-element measure; the source of 'U'"):
        Area().compute(probe)


def test_area_on_the_cells_is_their_volumes() -> None:
    # the measure belongs to the geometry: on the cells ``| Area() | Sum()`` is
    # the mesh volume, which is what cases/sources declares
    cells = InternalDataSet(name="p", field=np.array([1.0, 2.0]), geometry=FakeCells())

    result = Area().compute(cells)

    assert_allclose(np.asarray(result.field), CELL_VOLUMES, rtol=1e-12, err_msg="cell volumes")
    assert result.name == "area"


@pytest.mark.parametrize(
    "node",
    [
        pytest.param(Mag(), id="mag"),
        pytest.param(Component(index=0), id="component"),
        pytest.param(Area(), id="area"),
    ],
)
def test_a_field_function_carries_the_mask_and_the_groups_through(node: Node) -> None:
    dataset = _dataset()

    result: Any = node.compute(dataset)

    assert result is not dataset
    assert_array_equal(np.asarray(result.mask), MASK)
    assert_array_equal(np.asarray(result.groups), GROUPS)
    assert result.n_groups == 2
