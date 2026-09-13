# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Unit tests for the spatial selectors ``Box``, ``Sphere``, ``Not`` and ``Binary``.

The case matrix is pyOFTools' (a box, a sphere overlapping one of its corners,
and points inside / outside / on the boundary of each), extended with the two
NeoFOAM-specific rules: a selector ANDs with the mask it is handed instead of
replacing it, and it returns a new DataSet rather than mutating the one it got.
Positions are the only mesh data a selector reads, so a fake geometry is the
whole backend here.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pytest
from pydantic import ValidationError

from neofoam.postprocess.node import DataSet, Node
from neofoam.postprocess.nodes.selectors import Binary, Box, Not, Selector, Sphere

#: Points ordered: inside the box only, inside both, inside the sphere only,
#: outside both, on the box boundary, on the sphere boundary.
POSITIONS = np.array(
    [
        [0.25, 0.25, 0.25],
        [0.75, 0.75, 0.75],
        [1.25, 1.0, 1.0],
        [2.0, 2.0, 2.0],
        [1.0, 1.0, 1.0],
        [0.5, 1.0, 1.0],
    ]
)

BOX = Box(min=(0.0, 0.0, 0.0), max=(1.0, 1.0, 1.0))
#: Centred on a corner of the box; the radius and the boundary point below are
#: exact binary fractions, so "on the boundary" is a decision and not a rounding.
SPHERE = Sphere(center=(1.0, 1.0, 1.0), radius=0.5)


class FakeGeometry:
    """A geometry: a selector reads only the element positions."""

    def __init__(self, positions: np.ndarray) -> None:
        self._positions = positions

    @property
    def positions(self) -> np.ndarray:
        return self._positions

    @property
    def measure(self) -> Optional[np.ndarray]:
        return None


def _dataset(mask: Optional[np.ndarray] = None) -> DataSet:
    return DataSet(
        name="p",
        values=np.arange(len(POSITIONS), dtype=float),
        geometry=FakeGeometry(POSITIONS),
        mask=mask,
    )


# --- the region case matrix -----------------------------------------------


@pytest.mark.parametrize(
    ("selector", "expected"),
    [
        pytest.param(BOX, [True, True, False, False, True, True], id="box"),
        pytest.param(SPHERE, [False, True, True, False, True, True], id="sphere"),
        pytest.param(Not(region=BOX), [False, False, True, True, False, False], id="not_box"),
        pytest.param(Not(region=SPHERE), [True, False, False, True, False, False], id="not_sphere"),
        pytest.param(
            Binary(op="and", left=BOX, right=SPHERE),
            [False, True, False, False, True, True],
            id="box_and_sphere",
        ),
        pytest.param(
            Binary(op="or", left=BOX, right=SPHERE),
            [True, True, True, False, True, True],
            id="box_or_sphere",
        ),
        pytest.param(
            BOX & ~SPHERE, [True, False, False, False, False, False], id="box_and_not_sphere"
        ),
    ],
)
def test_selector_masks_the_elements_inside_the_region(
    selector: Selector, expected: list[bool]
) -> None:
    result = selector.compute(_dataset())

    assert list(result.mask) == expected


@pytest.mark.parametrize(
    ("built_by_operator", "built_by_class"),
    [
        pytest.param(BOX & SPHERE, Binary(op="and", left=BOX, right=SPHERE), id="and"),
        pytest.param(BOX | SPHERE, Binary(op="or", left=BOX, right=SPHERE), id="or"),
        pytest.param(~BOX, Not(region=BOX), id="invert"),
    ],
)
def test_the_operators_build_the_same_selector_as_the_classes(
    built_by_operator: Selector, built_by_class: Selector
) -> None:
    assert built_by_operator == built_by_class


# --- the mask contract ----------------------------------------------------


def test_selector_keeps_elements_masked_out_upstream_masked_out() -> None:
    inbound = np.array([False, True, True, True, True, True])

    result = BOX.compute(_dataset(mask=inbound))

    assert list(result.mask) == [False, True, False, False, True, True]


def test_selector_leaves_the_input_dataset_untouched() -> None:
    inbound = np.array([True, True, True, True, True, True])
    dataset = _dataset(mask=inbound)

    result = BOX.compute(dataset)

    assert result is not dataset
    assert list(dataset.mask) == [True] * 6
    assert list(inbound) == [True] * 6


# --- nested selectors from a spec mapping ---------------------------------


def test_a_nested_selector_resolves_from_a_spec_mapping() -> None:
    spec = {"type": "not", "region": {"type": "box", "min": (0, 0, 0), "max": (1, 1, 1)}}

    node: Any = Node.create(node=spec).node  # type: ignore[attr-defined]

    assert isinstance(node, Not)
    assert isinstance(node.region, Box)
    assert list(node.compute(_dataset()).mask) == [False, False, True, True, False, False]


def test_a_nested_selector_round_trips_back_to_its_spec_mapping() -> None:
    spec = {
        "type": "binary",
        "op": "and",
        "left": {"type": "box", "min": (0.0, 0.0, 0.0), "max": (1.0, 1.0, 1.0)},
        "right": {
            "type": "not",
            "region": {"type": "sphere", "center": (1.0, 1.0, 1.0), "radius": 0.5},
        },
    }

    node: Any = Node.create(node=spec).node  # type: ignore[attr-defined]

    assert node.model_dump() == spec


def test_a_nested_field_rejects_a_node_that_is_not_a_selector() -> None:
    spec = {"type": "not", "region": {"type": "scale", "factor": 2.0}}

    with pytest.raises(ValidationError, match="expected a selector, 'scale' is a Scale"):
        Node.create(node=spec)  # type: ignore[attr-defined]
