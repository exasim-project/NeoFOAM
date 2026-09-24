# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Unit tests for the debugging nodes ``Scale`` and ``Print``.

``Scale`` proves the never-mutate-your-input rule that lets a pipeline be
evaluated every time step, ``Print`` proves the pass-through and its one line of
output (captured, since that line *is* the behaviour). ``Print`` is also the one
node besides ``Rows`` that reads the numbers itself, so it copies a field to the
host — on a host field that copy is a no-op, which is what the summary below is
computed from.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from neofoam.postprocess.node import AggregatedData, AggregatedDataSet, InternalDataSet
from neofoam.postprocess.nodes.debug import Print, Scale


class FakeGeometry:
    """A geometry: neither debug node reads more than the values."""

    def positions(self) -> np.ndarray:
        return np.zeros((3, 3))

    def volumes(self) -> np.ndarray:
        return np.ones(3)


def _dataset(values: np.ndarray) -> InternalDataSet:
    return InternalDataSet(name="p", field=values, geometry=FakeGeometry())


def test_scale_multiplies_the_values() -> None:
    result = Scale(factor=1000.0).compute(_dataset(np.array([1.0, 2.0, 3.0])))

    assert_allclose(
        np.asarray(result.field), [1000.0, 2000.0, 3000.0], rtol=1e-12, err_msg="scaled p"
    )


def test_scale_leaves_the_input_dataset_untouched() -> None:
    values = np.array([1.0, 2.0, 3.0])
    dataset = _dataset(values)

    scaled = Scale(factor=2.0).compute(dataset)

    assert scaled is not dataset
    assert_allclose(values, [1.0, 2.0, 3.0], rtol=1e-12, err_msg="input array mutated in place")
    assert_allclose(np.asarray(dataset.field), [1.0, 2.0, 3.0], rtol=1e-12)


def test_scale_defaults_to_the_identity() -> None:
    result = Scale().compute(_dataset(np.array([1.0, 2.0, 3.0])))

    assert_allclose(np.asarray(result.field), [1.0, 2.0, 3.0], rtol=1e-12)


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
    aggregated = AggregatedDataSet(name="volume_p", values=[AggregatedData(value=2.5)])

    result = Print().compute(aggregated)

    assert result is aggregated
    out = capsys.readouterr().out
    assert out.splitlines() == ["[postProcess] volume_p: {'volume_p': 2.5}"]


def test_print_says_so_when_the_aggregation_has_no_rows(
    capsys: pytest.CaptureFixture[str],
) -> None:
    empty = AggregatedDataSet(name="volume_p", values=[])

    Print().compute(empty)

    assert capsys.readouterr().out.splitlines() == ["[postProcess] volume_p: no rows"]
