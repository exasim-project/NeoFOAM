# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Unit tests for the two value objects a pipeline passes around.

``DataSet`` is frozen on purpose: the same ``Pipeline`` object is evaluated
every write step, so a node that mutated its input would poison every later
step. The ``with_*`` helpers are the only way to derive one, and each test below
asserts both halves of that contract — the derived object carries the change and
the original does not. ``AggregatedDataSet`` is checked for the column layout
the CSV writer relies on; the aggregators that produce it are covered in
``test_aggregators.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from neofoam.postprocess.node import AggregatedDataSet, DataSet

PRESSURE = np.array([1.0, 2.0, 3.0, 4.0])
MASK = np.array([True, False, True, False])
GROUPS = np.array([0, 0, 1, 1])


def _dataset() -> DataSet:
    return DataSet(name="p", values=PRESSURE, geometry=None)


# --- the DataSet contract -------------------------------------------------


def test_a_fresh_dataset_has_no_mask_and_one_group() -> None:
    dataset = _dataset()

    assert dataset.mask is None
    assert dataset.groups is None
    assert dataset.n_groups == 1


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
    np.testing.assert_allclose(combined.values, PRESSURE, rtol=1e-12)


@pytest.mark.parametrize("attribute", ["mask", "groups", "n_groups"])
def test_a_derived_dataset_is_frozen_too(attribute: str) -> None:
    masked = _dataset().with_mask(MASK)

    with pytest.raises(ValidationError):
        setattr(masked, attribute, None)


# --- the AggregatedDataSet layout ----------------------------------------

# The layout an aggregator emits and ``CsvWriter`` turns into lines: value
# columns only when nothing binned the elements, a leading ``bin`` column
# holding the group index and one row per group when something did.
LAYOUTS = [
    pytest.param(["p_sum"], [[10.0]], id="scalar"),
    pytest.param(["U_sum_0", "U_sum_1", "U_sum_2"], [[10.0, 0.0, -10.0]], id="vector"),
    pytest.param(["bin", "p_sum"], [[0.0, 3.0], [1.0, 7.0]], id="grouped-scalar"),
    pytest.param(
        ["bin", "U_sum_0", "U_sum_1", "U_sum_2"],
        [[0.0, 3.0, 0.0, -3.0], [1.0, 7.0, 0.0, -7.0]],
        id="grouped-vector",
    ),
]


@pytest.mark.parametrize("headers,rows", LAYOUTS)
def test_every_row_of_an_aggregation_fills_every_column(
    headers: list[str], rows: list[list[float]]
) -> None:
    result = AggregatedDataSet(name="p_sum", headers=headers, rows=rows)

    assert all(len(row) == len(result.headers) for row in result.rows)


def test_the_group_index_is_the_first_column_of_a_binned_aggregation() -> None:
    result = AggregatedDataSet(
        name="p_sum", headers=["bin", "p_sum"], rows=[[0.0, 3.0], [1.0, 7.0]]
    )

    assert result.headers[0] == "bin"
    assert [row[0] for row in result.rows] == [0.0, 1.0]
