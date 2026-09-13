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
from numpy.testing import assert_allclose
from pydantic import ValidationError

from neofoam.postprocess.node import DataSet, Node
from neofoam.postprocess.nodes.field_functions import Area, Component, Mag

VELOCITY = np.array([[3.0, 4.0, 0.0], [0.0, -6.0, 8.0]])
FACE_AREAS = np.array([0.25, 0.75])
MASK = np.array([True, False])
GROUPS = np.array([0, 1])


class FakeGeometry:
    """A geometry whose measure the ``Area`` node reads; ``None`` for a point cloud."""

    def __init__(self, measure: Optional[np.ndarray]) -> None:
        self._measure = measure

    @property
    def positions(self) -> np.ndarray:
        return np.zeros((len(VELOCITY), 3))

    @property
    def measure(self) -> Optional[np.ndarray]:
        return self._measure


def _dataset(measure: Optional[np.ndarray] = FACE_AREAS) -> DataSet:
    return DataSet(
        name="U",
        values=VELOCITY,
        geometry=FakeGeometry(measure),
        mask=MASK,
        groups=GROUPS,
        n_groups=2,
    )


def test_mag_reduces_a_vector_field_to_its_magnitudes() -> None:
    result = Mag().compute(_dataset())

    assert_allclose(result.values, [5.0, 10.0], rtol=1e-12, err_msg="|U|")


def test_mag_renames_the_dataset_after_the_function() -> None:
    assert Mag().compute(_dataset()).name == "mag(U)"


def test_component_picks_one_component_of_a_vector_field() -> None:
    result = Component(index=1).compute(_dataset())

    assert_allclose(result.values, [4.0, -6.0], rtol=1e-12, err_msg="U_1")


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
    scalar = _dataset().with_values(np.array([1.0, 2.0]))

    with pytest.raises(TypeError, match=r"needs a vector field; 'U' has scalar values"):
        node.compute(scalar)


def test_area_replaces_the_values_by_the_face_areas() -> None:
    result = Area().compute(_dataset())

    assert_allclose(result.values, FACE_AREAS, rtol=1e-12, err_msg="face areas")
    assert result.name == "area"


def test_area_on_a_geometry_without_a_measure_names_the_source() -> None:
    with pytest.raises(TypeError, match=r"area needs face areas; the source of 'U'"):
        Area().compute(_dataset(measure=None))


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
    assert list(result.mask) == [True, False]
    assert list(result.groups) == [0, 1]
    assert result.n_groups == 2
