# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Unit tests for the debugging nodes ``Scale`` and ``Print``.

Both are pure numpy: ``Scale`` proves the never-mutate-your-input rule that lets
a pipeline be evaluated every time step, ``Print`` proves the pass-through and
its one line of output (captured, since that line *is* the behaviour).
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from neofoam.postprocess.node import AggregatedDataSet, DataSet
from neofoam.postprocess.nodes.debug import Print, Scale


def _dataset(values: np.ndarray) -> DataSet:
    return DataSet(name="p", values=values, geometry=None)


def test_scale_multiplies_the_values() -> None:
    result = Scale(factor=1000.0).compute(_dataset(np.array([1.0, 2.0, 3.0])))

    assert_allclose(result.values, [1000.0, 2000.0, 3000.0], rtol=1e-12, err_msg="scaled p")


def test_scale_leaves_the_input_dataset_untouched() -> None:
    values = np.array([1.0, 2.0, 3.0])
    dataset = _dataset(values)

    scaled = Scale(factor=2.0).compute(dataset)

    assert scaled is not dataset
    assert_allclose(values, [1.0, 2.0, 3.0], rtol=1e-12, err_msg="input array mutated in place")
    assert_allclose(dataset.values, [1.0, 2.0, 3.0], rtol=1e-12)


def test_scale_defaults_to_the_identity() -> None:
    assert_allclose(Scale().compute(_dataset(np.array([1.0, 2.0]))).values, [1.0, 2.0], rtol=1e-12)


def test_print_passes_a_dataset_through_unchanged(capsys: pytest.CaptureFixture[str]) -> None:
    dataset = _dataset(np.array([1.0, 2.0, 3.0]))

    result = Print(label="cells").compute(dataset)

    assert result is dataset
    out = capsys.readouterr().out
    assert out.splitlines() == ["[cells] p: shape=(3,) min=1 max=3"]


def test_print_says_so_when_the_dataset_has_no_elements(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # A plane can cut no cell on this rank; a debug node must not raise there.
    Print(label="cut").compute(_dataset(np.array([])))

    assert capsys.readouterr().out.splitlines() == ["[cut] p: shape=(0,) no elements"]


def test_print_passes_an_aggregation_through_unchanged(
    capsys: pytest.CaptureFixture[str],
) -> None:
    aggregated = AggregatedDataSet(name="volume_p", headers=["volume_p"], rows=[[2.5]])

    result = Print().compute(aggregated)

    assert result is aggregated
    out = capsys.readouterr().out
    assert out.splitlines() == ["[postProcess] volume_p: {'volume_p': 2.5}"]


def test_print_says_so_when_the_aggregation_has_no_rows(
    capsys: pytest.CaptureFixture[str],
) -> None:
    empty = AggregatedDataSet(name="volume_p", headers=["volume_p"], rows=[])

    Print().compute(empty)

    assert capsys.readouterr().out.splitlines() == ["[postProcess] volume_p: no rows"]
