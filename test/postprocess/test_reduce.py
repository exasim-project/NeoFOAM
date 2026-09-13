# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Unit tests for the global reductions an aggregator applies to its local result.

Serially every reduction is the identity, which is what makes the aggregators
testable without OpenFOAM at all. The parallel branch cannot be exercised with a
real decomposed run here (one ``Foam::Time`` per process, and no MPI in a unit
test), so ``pybFoam.Pstream.parRun`` — exactly what ``_is_parallel_run`` reads —
is patched, and the refusal is asserted rather than a result: pybFoam binds no
element-wise reduction yet (plan risk R1).
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pybFoam as pyf
import pytest

from neofoam.postprocess._reduce import reduce_max, reduce_min, reduce_sum

REDUCTIONS = [
    pytest.param(reduce_sum, "sum", id="sum"),
    pytest.param(reduce_max, "max", id="max"),
    pytest.param(reduce_min, "min", id="min"),
]

PER_GROUP_SCALARS = np.array([1.0, -2.0, 3.5])
PER_GROUP_VECTORS = np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0]])


@pytest.mark.parametrize("reduce,name", REDUCTIONS)
@pytest.mark.parametrize("values", [PER_GROUP_SCALARS, PER_GROUP_VECTORS], ids=["scalar", "vector"])
def test_a_serial_reduction_hands_the_local_result_straight_back(
    reduce: Callable[[Any], Any], name: str, values: "np.ndarray[Any, Any]"
) -> None:
    reduced = reduce(values)

    assert reduced is values


@pytest.mark.parametrize("reduce,name", REDUCTIONS)
def test_a_reduction_is_the_identity_when_the_backend_runs_serially(
    reduce: Callable[[Any], Any], name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pyf.Pstream, "parRun", lambda: False)

    np.testing.assert_allclose(reduce(PER_GROUP_SCALARS), PER_GROUP_SCALARS, rtol=1e-12)


@pytest.mark.parametrize("reduce,name", REDUCTIONS)
def test_a_reduction_refuses_a_decomposed_run_naming_the_missing_binding(
    reduce: Callable[[Any], Any], name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pyf.Pstream, "parRun", lambda: True)

    with pytest.raises(NotImplementedError, match=rf"global {name}.*not bound in pybFoam"):
        reduce(PER_GROUP_SCALARS)
