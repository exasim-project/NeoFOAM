# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Unit tests for the aggregators on a host geometry.

A node sees the mesh only through a geometry's accessors (``positions()`` plus
the measure that geometry has), so the integrals below run on plain arrays
without OpenFOAM — the NeoN kernels behind them take host numpy as readily as a
NeoN vector, which is what makes a mesh-free test of the real kernel possible.
The expected sums are written out by hand in the test — never recomputed with
the code under test — and compared at ``rtol=1e-12``, i.e. a handful of float64
ulps, since the kernel is one weighted sum over four cells.

The tables below are asserted in *file* order (``table_headers`` /
``table_rows``: the bin index before the value it labels), because that is the
layout a reader of the CSV sees; the aggregation's own value-first order is
pinned in ``test_dataset.py``.

The case table is the pyOFTools semantics, one row per (aggregator, input)
pair: a mask *scales* the dropped elements to zero for the additive aggregators
and *skips* them for the extrema, and a bin with nothing in it reports
``GREAT``/``-GREAT`` rather than a NaN. Covering a new combination is a new
entry in ``CASES``, never a new test body.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import numpy as np
import pytest
from numpy.testing import assert_allclose

from neofoam.postprocess.node import InternalDataSet, InternalMesh, Node, PointDataSet
from neofoam.postprocess.nodes.aggregators import (
    GREAT,
    Max,
    Mean,
    Min,
    Sum,
    SurfIntegrate,
    VolIntegrate,
)
from neofoam.postprocess.writers.writer import table_headers, table_rows

CELL_VOLUMES = np.array([0.5, 0.25, 0.25, 1.0])
# 1*0.5 + 2*0.25 + 3*0.25 + 4*1.0
SCALAR_INTEGRAL = 5.75


class ArrayGeometry:
    """A geometry made of plain arrays — the numpy stand-in for a cell zone.

    It answers both ``volumes()`` and ``face_area_magnitudes()`` with the same
    numbers, so one fake serves the cell and the surface aggregators.
    """

    def __init__(self, positions: np.ndarray, measure: np.ndarray) -> None:
        self._positions = positions
        self._measure = measure

    def positions(self) -> np.ndarray:
        return self._positions

    def volumes(self) -> np.ndarray:
        return self._measure

    def face_area_magnitudes(self) -> np.ndarray:
        return self._measure


class MeasurelessGeometry:
    """A probe's geometry: positions and nothing to weigh with.

    A dataset validates its geometry against the protocol its kind declares, so
    this is what an aggregator that needs a measure actually meets — a
    ``PointDataSet``, the one dataset whose geometry has no measure at all.
    """

    def positions(self) -> np.ndarray:
        return np.zeros((4, 3))

    def distance(self) -> np.ndarray:
        return np.arange(4.0)


def _cells() -> ArrayGeometry:
    return ArrayGeometry(positions=np.zeros((4, 3)), measure=CELL_VOLUMES)


def test_array_geometry_satisfies_the_internal_mesh_protocol() -> None:
    assert isinstance(_cells(), InternalMesh)


def test_vol_integrate_is_the_volume_weighted_sum() -> None:
    dataset = InternalDataSet(name="p", field=np.array([1.0, 2.0, 3.0, 4.0]), geometry=_cells())

    result = VolIntegrate().compute(dataset)

    assert table_headers(result) == ["p_volIntegrate"]
    assert_allclose(
        table_rows(result), [[SCALAR_INTEGRAL]], rtol=1e-12, err_msg="volume integral of p"
    )


def test_vol_integrate_labels_the_column_after_the_field_by_default() -> None:
    dataset = InternalDataSet(name="alpha.water", field=np.ones(4), geometry=_cells())

    result = VolIntegrate().compute(dataset)

    assert result.name == "alpha.water_volIntegrate"
    assert table_headers(result) == ["alpha.water_volIntegrate"]


def test_vol_integrate_uses_the_given_name_for_the_column() -> None:
    dataset = InternalDataSet(name="alpha.water", field=np.ones(4), geometry=_cells())

    result = VolIntegrate(name="water_volume").compute(dataset)

    assert result.name == "water_volume"
    assert table_headers(result) == ["water_volume"]


def test_vol_integrate_of_a_vector_field_yields_one_column_per_component() -> None:
    values = np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [3.0, 0.0, -3.0], [4.0, 0.0, -4.0]])
    dataset = InternalDataSet(name="U", field=values, geometry=_cells())

    result = VolIntegrate().compute(dataset)

    assert table_headers(result) == [
        "U_volIntegrate_0",
        "U_volIntegrate_1",
        "U_volIntegrate_2",
    ]
    assert_allclose(
        table_rows(result),
        [[SCALAR_INTEGRAL, 0.0, -SCALAR_INTEGRAL]],
        rtol=1e-12,
        atol=1e-15,
        err_msg="component-wise volume integral of U",
    )


def test_vol_integrate_without_a_measure_reports_the_offending_source() -> None:
    dataset = PointDataSet(name="p", field=np.ones(4), geometry=MeasurelessGeometry())

    with pytest.raises(
        TypeError, match=r"volIntegrate needs cell volumes.*'p'.*MeasurelessGeometry"
    ):
        VolIntegrate().compute(dataset)


# --- mask, groups and the empty bin ---------------------------------------

VALUES = np.array([1.0, 2.0, 3.0, 4.0])
VECTORS = np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [3.0, 0.0, -3.0], [4.0, 0.0, -4.0]])
# Keeps cells 0 and 2 (values 1 and 3, volumes 0.5 and 0.25): a sum scales the
# dropped cells to zero while an extremum skips them, so `Max` is 3, not 4. A
# mask is 0/1 labels, which is what the kernels read.
MASK = np.array([1, 0, 1, 0], dtype=np.int32)
NOTHING_ACTIVE = np.zeros(4, dtype=np.int32)
PAIRS = np.array([0, 0, 1, 1], dtype=np.int32)
# Everything in bin 0 while the binner declared two: bin 1 is the empty group.
ALL_IN_FIRST = np.zeros(4, dtype=np.int32)
# All negative, so an extremum that scaled the masked cells to zero instead of
# skipping them would report 0 rather than the largest surviving value.
NEGATIVES = -VALUES


class Case(NamedTuple):
    """One aggregator run: the dataset to build, and the table it must produce."""

    node: Node
    values: np.ndarray
    mask: Optional[np.ndarray]
    groups: Optional[np.ndarray]
    n_groups: int
    headers: list[str]
    rows: list[list[float]]


def _case(
    node: Node,
    headers: list[str],
    rows: list[list[float]],
    *,
    values: np.ndarray = VALUES,
    mask: Optional[np.ndarray] = None,
    groups: Optional[np.ndarray] = None,
    n_groups: int = 1,
) -> Case:
    return Case(node, values, mask, groups, n_groups, headers, rows)


CASES = [
    # plain: every cell active, one group
    pytest.param(_case(Sum(), ["p_sum"], [[10.0]]), id="sum"),
    pytest.param(_case(Mean(), ["p_mean"], [[2.5]]), id="mean"),
    pytest.param(_case(Max(), ["p_max"], [[4.0]]), id="max"),
    pytest.param(_case(Min(), ["p_min"], [[1.0]]), id="min"),
    pytest.param(_case(VolIntegrate(), ["p_volIntegrate"], [[5.75]]), id="volIntegrate"),
    pytest.param(_case(SurfIntegrate(), ["p_surfIntegrate"], [[5.75]]), id="surfIntegrate"),
    # masked: sums scale the dropped cells to zero, extrema skip them
    pytest.param(_case(Sum(), ["p_sum"], [[4.0]], mask=MASK), id="sum-masked"),
    pytest.param(_case(Mean(), ["p_mean"], [[2.0]], mask=MASK), id="mean-masked"),
    pytest.param(_case(Max(), ["p_max"], [[3.0]], mask=MASK), id="max-masked"),
    pytest.param(
        _case(Max(), ["p_max"], [[-1.0]], values=NEGATIVES, mask=MASK), id="max-masked-negative"
    ),
    pytest.param(_case(Min(), ["p_min"], [[1.0]], mask=MASK), id="min-masked"),
    pytest.param(
        _case(VolIntegrate(), ["p_volIntegrate"], [[1.25]], mask=MASK), id="volIntegrate-masked"
    ),
    # binned: one row per bin, bin index first
    pytest.param(
        _case(Sum(), ["bin", "p_sum"], [[0.0, 3.0], [1.0, 7.0]], groups=PAIRS, n_groups=2),
        id="sum-binned",
    ),
    pytest.param(
        _case(Mean(), ["bin", "p_mean"], [[0.0, 1.5], [1.0, 3.5]], groups=PAIRS, n_groups=2),
        id="mean-binned",
    ),
    pytest.param(
        _case(Max(), ["bin", "p_max"], [[0.0, 2.0], [1.0, 4.0]], groups=PAIRS, n_groups=2),
        id="max-binned",
    ),
    pytest.param(
        _case(Min(), ["bin", "p_min"], [[0.0, 1.0], [1.0, 3.0]], groups=PAIRS, n_groups=2),
        id="min-binned",
    ),
    pytest.param(
        _case(
            VolIntegrate(),
            ["bin", "p_volIntegrate"],
            [[0.0, 1.0], [1.0, 4.75]],
            groups=PAIRS,
            n_groups=2,
        ),
        id="volIntegrate-binned",
    ),
    # binned and masked at once: one cell survives per bin
    pytest.param(
        _case(
            Sum(),
            ["bin", "p_sum"],
            [[0.0, 1.0], [1.0, 3.0]],
            mask=MASK,
            groups=PAIRS,
            n_groups=2,
        ),
        id="sum-binned-masked",
    ),
    pytest.param(
        _case(
            Mean(),
            ["bin", "p_mean"],
            [[0.0, 1.0], [1.0, 3.0]],
            mask=MASK,
            groups=PAIRS,
            n_groups=2,
        ),
        id="mean-binned-masked",
    ),
    # empty bin: zero for a sum, the OpenFOAM sentinel for a mean or an extremum
    pytest.param(
        _case(Sum(), ["bin", "p_sum"], [[0.0, 10.0], [1.0, 0.0]], groups=ALL_IN_FIRST, n_groups=2),
        id="sum-empty-bin",
    ),
    pytest.param(
        _case(
            Mean(),
            ["bin", "p_mean"],
            [[0.0, 2.5], [1.0, GREAT]],
            groups=ALL_IN_FIRST,
            n_groups=2,
        ),
        id="mean-empty-bin",
    ),
    pytest.param(
        _case(
            Max(), ["bin", "p_max"], [[0.0, 4.0], [1.0, -GREAT]], groups=ALL_IN_FIRST, n_groups=2
        ),
        id="max-empty-bin",
    ),
    pytest.param(
        _case(Min(), ["bin", "p_min"], [[0.0, 1.0], [1.0, GREAT]], groups=ALL_IN_FIRST, n_groups=2),
        id="min-empty-bin",
    ),
    # every cell masked out: the whole table is the empty-group case
    pytest.param(_case(Sum(), ["p_sum"], [[0.0]], mask=NOTHING_ACTIVE), id="sum-all-masked"),
    pytest.param(_case(Mean(), ["p_mean"], [[GREAT]], mask=NOTHING_ACTIVE), id="mean-all-masked"),
    pytest.param(_case(Max(), ["p_max"], [[-GREAT]], mask=NOTHING_ACTIVE), id="max-all-masked"),
    pytest.param(_case(Min(), ["p_min"], [[GREAT]], mask=NOTHING_ACTIVE), id="min-all-masked"),
    # vectors: one column per component, component-wise extrema
    pytest.param(
        _case(
            Sum(),
            ["p_sum_0", "p_sum_1", "p_sum_2"],
            [[10.0, 0.0, -10.0]],
            values=VECTORS,
        ),
        id="sum-vector",
    ),
    pytest.param(
        _case(
            Mean(),
            ["p_mean_0", "p_mean_1", "p_mean_2"],
            [[2.5, 0.0, -2.5]],
            values=VECTORS,
        ),
        id="mean-vector",
    ),
    pytest.param(
        _case(
            Max(),
            ["p_max_0", "p_max_1", "p_max_2"],
            [[4.0, 0.0, -1.0]],
            values=VECTORS,
        ),
        id="max-vector",
    ),
    pytest.param(
        _case(
            Min(),
            ["p_min_0", "p_min_1", "p_min_2"],
            [[1.0, 0.0, -4.0]],
            values=VECTORS,
        ),
        id="min-vector",
    ),
    pytest.param(
        _case(
            Mean(),
            ["p_mean_0", "p_mean_1", "p_mean_2"],
            [[2.0, 0.0, -2.0]],
            values=VECTORS,
            mask=MASK,
        ),
        id="mean-vector-masked",
    ),
    pytest.param(
        _case(
            Max(),
            ["p_max_0", "p_max_1", "p_max_2"],
            [[3.0, 0.0, -1.0]],
            values=VECTORS,
            mask=MASK,
        ),
        id="max-vector-masked",
    ),
    pytest.param(
        _case(
            Sum(),
            ["bin", "p_sum_0", "p_sum_1", "p_sum_2"],
            [[0.0, 3.0, 0.0, -3.0], [1.0, 7.0, 0.0, -7.0]],
            values=VECTORS,
            groups=PAIRS,
            n_groups=2,
        ),
        id="sum-vector-binned",
    ),
    pytest.param(
        _case(
            Mean(),
            ["bin", "p_mean_0", "p_mean_1", "p_mean_2"],
            [[0.0, 2.5, 0.0, -2.5], [1.0, GREAT, GREAT, GREAT]],
            values=VECTORS,
            groups=ALL_IN_FIRST,
            n_groups=2,
        ),
        id="mean-vector-empty-bin",
    ),
    pytest.param(
        _case(
            VolIntegrate(),
            ["bin", "p_volIntegrate_0", "p_volIntegrate_1", "p_volIntegrate_2"],
            [[0.0, 1.0, 0.0, -1.0], [1.0, 4.75, 0.0, -4.75]],
            values=VECTORS,
            groups=PAIRS,
            n_groups=2,
        ),
        id="volIntegrate-vector-binned",
    ),
]


@pytest.mark.parametrize("case", CASES)
def test_an_aggregator_reproduces_the_hand_computed_table(case: Case) -> None:
    dataset = InternalDataSet(
        name="p",
        field=case.values,
        geometry=_cells(),
        mask=case.mask,
        groups=case.groups,
        n_groups=case.n_groups,
    )

    result = case.node.compute(dataset)

    assert table_headers(result) == case.headers
    # One weighted sum (or one comparison) over four cells: a handful of float64
    # ulps, and the exact zeros of the second vector component need an atol.
    assert_allclose(
        table_rows(result), case.rows, rtol=1e-12, atol=1e-15, err_msg=f"{case.node.type} on p"
    )


# --- naming, immutability and the rejected inputs -------------------------


@pytest.mark.parametrize(
    "node,label",
    [
        pytest.param(Sum(), "p_sum", id="sum"),
        pytest.param(Mean(), "p_mean", id="mean"),
        pytest.param(Max(), "p_max", id="max"),
        pytest.param(Min(), "p_min", id="min"),
        pytest.param(VolIntegrate(), "p_volIntegrate", id="volIntegrate"),
        pytest.param(SurfIntegrate(), "p_surfIntegrate", id="surfIntegrate"),
    ],
)
def test_an_aggregator_names_its_column_after_the_field_and_itself(node: Node, label: str) -> None:
    dataset = InternalDataSet(name="p", field=VALUES, geometry=_cells())

    result = node.compute(dataset)

    assert result.name == label
    assert table_headers(result) == [label]


def test_an_aggregator_leaves_its_input_dataset_untouched() -> None:
    dataset = InternalDataSet(name="p", field=VALUES.copy(), geometry=_cells(), mask=MASK.copy())

    Mean().compute(dataset)

    assert_allclose(dataset.field, VALUES, rtol=1e-12)
    np.testing.assert_array_equal(dataset.mask, MASK)


def test_surf_integrate_without_a_measure_reports_the_offending_source() -> None:
    dataset = PointDataSet(name="p", field=VALUES, geometry=MeasurelessGeometry())

    with pytest.raises(
        TypeError, match=r"surfIntegrate needs face areas.*'p'.*MeasurelessGeometry"
    ):
        SurfIntegrate().compute(dataset)


def test_a_group_index_beyond_the_declared_bins_is_refused() -> None:
    # A row count read off the data would differ between ranks (plan risk R6).
    dataset = InternalDataSet(
        name="p",
        field=VALUES,
        geometry=_cells(),
        groups=np.array([0, 1, 2, 2], dtype=np.int32),
        n_groups=2,
    )

    with pytest.raises(ValueError, match=r"'p' has group indices \[0, 2\] outside the 2 groups"):
        Sum().compute(dataset)


def test_a_probe_geometry_is_not_an_internal_mesh() -> None:
    # what makes the "needs cell volumes" error possible rather than an
    # AttributeError deep in the kernel call
    assert not isinstance(MeasurelessGeometry(), InternalMesh)


def test_an_aggregated_value_is_plain_python() -> None:
    # the writers and the MCP catalog serialise these, so a numpy scalar leaking
    # into a row would show up as its repr in the CSV
    dataset = InternalDataSet(name="p", field=VALUES, geometry=_cells())

    result = Sum().compute(dataset)

    assert type(result.values[0].value) is float
