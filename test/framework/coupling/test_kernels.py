# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The numpy kernels of a coupling, each on a few hand-checked values.

The relaxations are fed the fixed-point iteration ``x -> 0.5 * x + 1`` (fixed
point 2), the way a coupling feeds them: what the other side sends back for the
value last delivered. Aitken's factor is exact for a linear map of one value, so
its second relaxed pass lands on the fixed point (to round-off, hence 1e-12).
"""

from typing import Any

import numpy as np
import pytest

from neofoam.framework.coupling import (
    AitkenRelaxation,
    ConstantRelaxation,
    CoupledSide,
    MappedTransfer,
    RelativeConvergence,
    SideChange,
    nearest_neighbor,
)


def _side(coordinates: list[list[float]]) -> CoupledSide:
    return CoupledSide(send=dict, receive=lambda: None, coordinates=np.array(coordinates))


def _passes(relaxation: ConstantRelaxation, count: int) -> list[float]:
    """The values delivered in ``count`` passes of one time step, starting from 0."""
    delivered = [float(relaxation([{"x": np.zeros(1)}], 1)[0]["x"][0])]
    for iteration in range(2, count + 1):
        sent_back = 0.5 * delivered[-1] + 1.0
        delivered.append(float(relaxation([{"x": np.full(1, sent_back)}], iteration)[0]["x"][0]))
    return delivered


def test_nearest_neighbor_picks_the_closest_source_point() -> None:
    source = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    target = np.array([[1.9, 0.1], [0.2, 0.0], [1.1, -0.3]])

    assert nearest_neighbor(source, target).tolist() == [2, 0, 1]


def test_constant_relaxation_moves_by_the_factor() -> None:
    delivered = _passes(ConstantRelaxation(omega=0.5), 3)

    np.testing.assert_allclose(delivered, [0.0, 0.5, 0.875], rtol=1e-15, err_msg="omega 0.5")


def test_aitken_relaxation_reaches_the_fixed_point_of_a_linear_map() -> None:
    delivered = _passes(AitkenRelaxation(initial=0.5), 3)

    np.testing.assert_allclose(delivered, [0.0, 0.5, 2.0], rtol=1e-12, err_msg="Aitken")


def test_aitken_relaxation_restarts_from_initial_each_time_step() -> None:
    relaxation = AitkenRelaxation(initial=0.5)
    _passes(relaxation, 3)

    first_pass = relaxation([{"x": np.full(1, 4.0)}], 1)[0]["x"]

    np.testing.assert_allclose(first_pass, [3.0], rtol=1e-12, err_msg="2 + 0.5 * (4 - 2)")


def test_aitken_relaxation_restarts_after_a_round_off_residual() -> None:
    relaxation = AitkenRelaxation(initial=0.5)
    relaxation([{"x": np.full(1, 2.0)}], 1)
    relaxation([{"x": np.full(1, 2.0)}], 2)

    relaxed = relaxation([{"x": np.full(1, 4.0)}], 3)[0]["x"]

    np.testing.assert_allclose(relaxed, [3.0], rtol=1e-12, err_msg="2 + 0.5 * (4 - 2)")


def test_relaxation_keeps_the_names_and_shapes_of_the_data() -> None:
    sent = [{"T_sink": np.zeros(2), "h": np.ones(2)}, {"T_sink": np.full(2, 3.0)}]

    relaxed = ConstantRelaxation(omega=0.5)(sent, 1)

    assert [{name: values.tolist() for name, values in data.items()} for data in relaxed] == [
        {"T_sink": [0.0, 0.0], "h": [1.0, 1.0]},
        {"T_sink": [3.0, 3.0]},
    ]


def test_mapped_transfer_maps_each_target_point_to_its_nearest_source() -> None:
    source = _side([[0.0], [1.0], [2.0]])
    target = _side([[1.9], [0.1]])
    transfer = MappedTransfer()

    received = transfer({"T": np.array([10.0, 20.0, 30.0])}, source, target, 1)

    assert received["T"].tolist() == [30.0, 10.0]


@pytest.mark.parametrize(
    ("current", "converged"),
    [(100.0 + 1e-4, True), (100.0 + 1e-2, False)],
    ids=["change 1e-6 relative", "change 1e-4 relative"],
)
def test_relative_convergence_compares_the_change_with_the_limit(
    current: float, converged: bool
) -> None:
    change = SideChange("T", {"T_sink": np.array([100.0])}, {"T_sink": np.array([current])})

    assert RelativeConvergence(data="T", limit=1e-5)([change]) is converged


def test_relative_convergence_ignores_other_fields() -> None:
    changes: list[Any] = [SideChange("U", {"u": np.array([1.0])}, {"u": np.array([2.0])})]

    assert RelativeConvergence(data="T", limit=1e-5)(changes) is True
