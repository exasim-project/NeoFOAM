# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Unit tests for the global reductions an aggregator applies to its local result.

Serially every reduction is the identity, which is what makes the aggregators
testable without OpenFOAM at all. The parallel branch cannot be exercised for
real here (one ``Foam::Time`` per process, and no MPI in a unit test), so
``pybFoam.Pstream.parRun`` — exactly what ``_is_parallel_run`` reads — and the
pybFoam collective behind the reduction are both patched, and what is asserted
is the *delegation*: one collective per element of the flattened array, the
caller's array untouched, the shape restored. The values themselves are pinned
by the decomposed end-to-end run in
``test/solver/incompressibleFluid/test_post_process_parallel.py``.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pybFoam as pyf
import pytest

from neofoam.postprocess._reduce import reduce_max, reduce_min, reduce_sum

REDUCTIONS = [
    pytest.param(reduce_sum, "sum", id="sum"),
    pytest.param(reduce_max, "gMax", id="max"),
    pytest.param(reduce_min, "gMin", id="min"),
]

PER_GROUP_SCALARS = np.array([1.0, -2.0, 3.5])
PER_GROUP_VECTORS = np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0]])


@pytest.mark.parametrize("reduce,collective", REDUCTIONS)
@pytest.mark.parametrize("values", [PER_GROUP_SCALARS, PER_GROUP_VECTORS], ids=["scalar", "vector"])
def test_a_serial_reduction_hands_the_local_result_straight_back(
    reduce: Callable[[Any], Any], collective: str, values: "np.ndarray[Any, Any]"
) -> None:
    reduced = reduce(values)

    assert reduced is values


@pytest.mark.parametrize("reduce,collective", REDUCTIONS)
def test_a_reduction_is_the_identity_when_the_backend_runs_serially(
    reduce: Callable[[Any], Any], collective: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pyf.Pstream, "parRun", lambda: False)

    np.testing.assert_allclose(reduce(PER_GROUP_SCALARS), PER_GROUP_SCALARS, rtol=1e-12)


@pytest.mark.parametrize("reduce,collective", REDUCTIONS)
@pytest.mark.parametrize("values", [PER_GROUP_SCALARS, PER_GROUP_VECTORS], ids=["scalar", "vector"])
def test_a_decomposed_reduction_calls_the_collective_once_per_element(
    reduce: Callable[[Any], Any],
    collective: str,
    values: "np.ndarray[Any, Any]",
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[list[float]] = []

    def fake_collective(field: Any) -> float:
        seen.append([float(value) for value in field])
        return 10.0 * seen[-1][0]

    monkeypatch.setattr(pyf.Pstream, "parRun", lambda: True)
    monkeypatch.setattr(pyf, collective, fake_collective)
    local = values.copy()

    reduced = reduce(local)

    assert seen == [[value] for value in values.ravel()]
    np.testing.assert_allclose(reduced, 10.0 * values, rtol=1e-12)
    np.testing.assert_array_equal(local, values)
