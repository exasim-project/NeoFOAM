# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Unit tests for the value objects a pipeline passes around.

A field dataset is frozen on purpose: the same ``Pipeline`` object is evaluated
every write step, so a node that mutated its input would poison every later
step. The ``with_*`` helpers are the only way to derive one, and each test below
asserts both halves of that contract — the derived object carries the change and
the original does not.

:meth:`InternalDataSet.from_field` is the one branch on the field library, so
the two shapes it accepts are the whole contract and stand-ins spell them out
better than either library does: a pybFoam field answers ``internalField()``, a
NeoN one hands out a ``Vector`` that stays on its executor. The NeoN case uses a
*real* ``neon.ScalarVector`` on the serial executor — a fake would not prove that
the values are handed on untouched and that the cell geometry is mirrored onto
the same executor, which is what keeps a GPU field off the host. The live NeoN
read is ``test/solver/incompressibleFluidNeoN/test_post_process.py``.

``AggregatedDataSet`` is checked for the column layout the writers turn into a
header; the aggregators that produce it are covered in
``nodes/test_aggregators.py``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from neofoam.postprocess.node import AggregatedData, AggregatedDataSet, InternalDataSet
from neofoam.postprocess.sources.geometry import CellGeometry, NeonCellGeometry

PRESSURE = np.array([1.0, 2.0, 3.0])
MASK = np.array([1, 0, 1], dtype=np.int32)
GROUPS = np.array([0, 0, 1], dtype=np.int32)

CELL_CENTRES = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
CELL_VOLUMES = np.array([0.5, 0.25, 1.0])


class FakeCellField:
    """A pybFoam volume field: the cell geometry reads only its internal values."""

    def __init__(self, values: "np.ndarray[Any, Any]") -> None:
        self._values = values

    def internalField(self) -> "np.ndarray[Any, Any]":  # noqa: N802  # pybFoam's spelling
        return self._values


class _NeonField:
    """A NeoN ``VolumeField``: its cell values live in ``internal_vector()``."""

    def __init__(self, vector: Any) -> None:
        self._vector = vector

    def internal_vector(self) -> Any:
        return self._vector


class FakeMesh:
    """An fvMesh: the cell geometry reads only cell centres and cell volumes."""

    def C(self) -> FakeCellField:  # noqa: N802  # OpenFOAM's spelling
        return FakeCellField(CELL_CENTRES)

    def V(self) -> "np.ndarray[Any, Any]":  # noqa: N802  # OpenFOAM's spelling
        return CELL_VOLUMES


def _dataset() -> InternalDataSet:
    return InternalDataSet(name="p", field=PRESSURE, geometry=CellGeometry(FakeMesh()))


# --- the dataset contract -------------------------------------------------


def test_a_fresh_dataset_has_no_mask_and_one_group() -> None:
    dataset = _dataset()

    assert dataset.mask is None
    assert dataset.groups is None
    assert dataset.n_groups == 1


def test_with_field_returns_a_new_dataset_leaving_the_original_untouched() -> None:
    dataset = _dataset()

    scaled = dataset.with_field(PRESSURE * 2.0)

    assert scaled is not dataset
    np.testing.assert_allclose(dataset.field, PRESSURE, rtol=1e-12)
    np.testing.assert_allclose(scaled.field, [2.0, 4.0, 6.0], rtol=1e-12)


def test_with_mask_returns_a_new_dataset_leaving_the_original_unmasked() -> None:
    dataset = _dataset()

    masked = dataset.with_mask(MASK)

    assert masked is not dataset
    assert dataset.mask is None
    np.testing.assert_array_equal(masked.mask, MASK)


def test_with_groups_returns_a_new_dataset_carrying_the_binner_s_group_count() -> None:
    dataset = _dataset()

    binned = dataset.with_groups(GROUPS, n_groups=2)

    assert binned is not dataset
    assert dataset.groups is None and dataset.n_groups == 1
    np.testing.assert_array_equal(binned.groups, GROUPS)
    assert binned.n_groups == 2


def test_with_mask_and_with_groups_compose_without_losing_each_other() -> None:
    combined = _dataset().with_mask(MASK).with_groups(GROUPS, n_groups=2)

    np.testing.assert_array_equal(combined.mask, MASK)
    np.testing.assert_array_equal(combined.groups, GROUPS)
    np.testing.assert_allclose(combined.field, PRESSURE, rtol=1e-12)


def test_a_derived_dataset_keeps_the_class_of_the_one_it_came_from() -> None:
    assert isinstance(_dataset().with_mask(MASK), InternalDataSet)


@pytest.mark.parametrize("attribute", ["field", "mask", "groups", "n_groups"])
def test_a_derived_dataset_is_frozen_too(attribute: str) -> None:
    masked = _dataset().with_mask(MASK)

    with pytest.raises(ValidationError):
        setattr(masked, attribute, None)


# --- reading a volume field off either field library -----------------------


def test_a_pybfoam_field_is_read_as_host_numpy_on_the_host_cell_geometry() -> None:
    dataset = InternalDataSet.from_field("p", FakeCellField(PRESSURE), FakeMesh())

    assert dataset.name == "p"
    assert isinstance(dataset.geometry, CellGeometry)
    np.testing.assert_array_equal(dataset.field, PRESSURE)
    np.testing.assert_allclose(dataset.geometry.volumes(), CELL_VOLUMES, rtol=1e-12)


def test_a_neon_field_keeps_its_vector_and_gets_the_cells_on_its_executor() -> None:
    neon = pytest.importorskip("neon")
    executor = neon.SerialExecutor()
    vector = neon.ScalarVector(executor, list(PRESSURE))

    dataset = InternalDataSet.from_field("p", _NeonField(vector), FakeMesh())

    assert dataset.field is vector, "the field's own vector must be handed on, not a copy"
    assert isinstance(dataset.geometry, NeonCellGeometry)
    np.testing.assert_allclose(
        np.asarray(dataset.geometry.volumes().copy_to_host()), CELL_VOLUMES, rtol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(dataset.geometry.positions().copy_to_host()), CELL_CENTRES, rtol=1e-12
    )


def test_the_cell_geometry_of_a_neon_field_is_built_once_per_mesh() -> None:
    # a static mesh, so the copy onto the executor must not be repeated every
    # write step
    neon = pytest.importorskip("neon")
    executor = neon.SerialExecutor()
    mesh = FakeMesh()
    field = _NeonField(neon.ScalarVector(executor, list(PRESSURE)))

    first = InternalDataSet.from_field("p", field, mesh)
    second = InternalDataSet.from_field("p", field, mesh)

    assert first.geometry is second.geometry


# --- the AggregatedDataSet layout ----------------------------------------

# The layout an aggregator emits: one entry per row, its value and the labels
# that name it. ``headers`` and ``grouped_values`` list the value columns first;
# a writer puts the labels in front of them (``test_writer.py``).
LAYOUTS = [
    pytest.param([AggregatedData(value=10.0)], ["p_sum"], [[10.0]], id="scalar"),
    pytest.param(
        [AggregatedData(value=[10.0, 0.0, -10.0])],
        ["p_sum_0", "p_sum_1", "p_sum_2"],
        [[10.0, 0.0, -10.0]],
        id="vector",
    ),
    pytest.param(
        [
            AggregatedData(value=3.0, group=[0.0], group_name=["bin"]),
            AggregatedData(value=7.0, group=[1.0], group_name=["bin"]),
        ],
        ["p_sum", "bin"],
        [[3.0, 0.0], [7.0, 1.0]],
        id="grouped-scalar",
    ),
    pytest.param(
        [
            AggregatedData(value=[3.0, 0.0, -3.0], group=[0.0], group_name=["bin"]),
            AggregatedData(value=[7.0, 0.0, -7.0], group=[1.0], group_name=["bin"]),
        ],
        ["p_sum_0", "p_sum_1", "p_sum_2", "bin"],
        [[3.0, 0.0, -3.0, 0.0], [7.0, 0.0, -7.0, 1.0]],
        id="grouped-vector",
    ),
]


@pytest.mark.parametrize("values,headers,rows", LAYOUTS)
def test_an_aggregation_names_one_column_per_value_component_and_group(
    values: list[AggregatedData], headers: list[str], rows: list[list[Any]]
) -> None:
    result = AggregatedDataSet(name="p_sum", values=values)

    assert result.headers == headers


@pytest.mark.parametrize("values,headers,rows", LAYOUTS)
def test_every_row_of_an_aggregation_fills_every_column(
    values: list[AggregatedData], headers: list[str], rows: list[list[Any]]
) -> None:
    result = AggregatedDataSet(name="p_sum", values=values)

    assert result.grouped_values == rows
    assert all(len(row) == len(result.headers) for row in result.grouped_values)


def test_an_aggregation_with_nothing_in_it_has_no_columns() -> None:
    # a residuals step that solved nothing: the columns of a row are only known
    # once there is one, and the writer leaves the file alone until then
    result = AggregatedDataSet(name="value", values=[])

    assert result.headers == []
    assert result.grouped_values == []
