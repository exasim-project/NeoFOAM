# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Unit tests for ``Rows`` — the terminal node that writes elements instead of reducing them.

``Rows`` reads only positions, values and the mask, so a fake geometry is the
whole backend. What it has to get right is the column layout the CsvWriter turns
into a header (``x,y,z`` then one column per component) and that a masked-out
element leaves no row behind — which is how a line probe that pokes out of the
mesh still writes a clean file. The rows are asserted in file order
(``table_rows``), the order the CSV shows.

The one thing it cannot do is run decomposed — its elements are spread over the
ranks and there is no gather — so the refusal is pinned here with a patched
``Pstream.parRun``, the same way ``test_reduce.py`` reaches the parallel branch.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pybFoam as pyf
import pytest

from neofoam.postprocess.node import PointDataSet
from neofoam.postprocess.nodes.rows import Rows
from neofoam.postprocess.writers.writer import table_headers, table_rows

POSITIONS = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
SCALARS = np.array([10.0, 20.0, 30.0])
VECTORS = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])


class FakeGeometry:
    """A probe's geometry: ``Rows`` reads only the element positions."""

    def positions(self) -> "np.ndarray[Any, Any]":
        return POSITIONS

    def distance(self) -> "np.ndarray[Any, Any]":
        return np.arange(float(len(POSITIONS)))


def _dataset(values: "np.ndarray[Any, Any]", mask: Optional[Any] = None) -> PointDataSet:
    return PointDataSet(name="U", field=values, geometry=FakeGeometry(), mask=mask)


def _binned(values: "np.ndarray[Any, Any]") -> PointDataSet:
    return _dataset(values).with_groups(np.array([0, 1, 1], dtype=np.int32), n_groups=2)


def test_a_scalar_field_gets_one_position_column_set_and_one_value_column() -> None:
    result = Rows().compute(_dataset(SCALARS))

    assert table_headers(result) == ["x", "y", "z", "U"]
    assert table_rows(result) == [
        [0.0, 0.0, 0.0, 10.0],
        [1.0, 2.0, 3.0, 20.0],
        [4.0, 5.0, 6.0, 30.0],
    ]


def test_a_vector_field_gets_one_value_column_per_component() -> None:
    result = Rows().compute(_dataset(VECTORS))

    assert table_headers(result) == ["x", "y", "z", "U_0", "U_1", "U_2"]
    assert table_rows(result)[1] == [1.0, 2.0, 3.0, 0.0, 2.0, 0.0]


def test_a_masked_out_element_gets_no_row() -> None:
    result = Rows().compute(_dataset(SCALARS, mask=np.array([1, 0, 1], dtype=np.int32)))

    assert table_rows(result) == [[0.0, 0.0, 0.0, 10.0], [4.0, 5.0, 6.0, 30.0]]


def test_the_columns_are_named_after_the_node_when_it_carries_a_name() -> None:
    result = Rows(name="U_profile").compute(_dataset(SCALARS))

    assert result.name == "U_profile"
    assert table_headers(result) == ["x", "y", "z", "U_profile"]


def test_a_binned_dataset_keeps_its_bin_as_a_leading_column() -> None:
    result = Rows().compute(_binned(SCALARS))

    assert table_headers(result) == ["bin", "x", "y", "z", "U"]
    assert table_rows(result)[2] == [1.0, 4.0, 5.0, 6.0, 30.0]


@pytest.mark.parametrize("values", [SCALARS, VECTORS], ids=["scalar", "vector"])
@pytest.mark.parametrize("build", [_dataset, _binned], ids=["ungrouped", "binned"])
def test_every_row_has_as_many_entries_as_there_are_headers(
    build: Any, values: "np.ndarray[Any, Any]"
) -> None:
    result = Rows().compute(build(values))

    assert {len(row) for row in table_rows(result)} == {len(table_headers(result))}


def test_a_decomposed_run_is_refused_rather_than_writing_the_master_ranks_share(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pyf.Pstream, "parRun", lambda: True)

    with pytest.raises(NotImplementedError, match="rows table 'U_profile' cannot run decomposed"):
        Rows(name="U_profile").compute(_dataset(SCALARS))
