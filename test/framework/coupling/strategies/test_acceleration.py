# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The relaxations of a coupling, each on a few hand-checked values.

The relaxations are fed the fixed-point iteration ``x -> 0.5 * x + 1`` (fixed
point 2), the way a coupling feeds them: what the other side sends back for the
value last delivered. Aitken's factor is exact for a linear map of one value, so
its second relaxed pass lands on the fixed point (to round-off, hence 1e-12).
"""

import numpy as np
import pytest

from neofoam.framework.coupling import (
    AitkenRelaxation,
    ConstantRelaxation,
)


def _passes(relaxation: ConstantRelaxation, count: int) -> list[float]:
    """The values delivered in ``count`` passes of one time step, starting from 0."""
    delivered = [float(relaxation([{"x": np.zeros(1)}], 1)[0]["x"][0])]
    for iteration in range(2, count + 1):
        sent_back = 0.5 * delivered[-1] + 1.0
        delivered.append(float(relaxation([{"x": np.full(1, sent_back)}], iteration)[0]["x"][0]))
    return delivered


def test_constant_relaxation_moves_by_the_factor() -> None:
    delivered = _passes(ConstantRelaxation(relaxation=0.5), 3)

    np.testing.assert_allclose(delivered, [0.0, 0.5, 0.875], rtol=1e-15, err_msg="omega 0.5")


def test_aitken_relaxation_reaches_the_fixed_point_of_a_linear_map() -> None:
    delivered = _passes(AitkenRelaxation(relaxation=0.5), 3)

    np.testing.assert_allclose(delivered, [0.0, 0.5, 2.0], rtol=1e-12, err_msg="Aitken")


def test_aitken_relaxation_restarts_from_initial_each_time_step() -> None:
    relaxation = AitkenRelaxation(relaxation=0.5)
    _passes(relaxation, 3)

    first_pass = relaxation([{"x": np.full(1, 4.0)}], 1)[0]["x"]

    np.testing.assert_allclose(first_pass, [4.0], rtol=0, atol=0, err_msg="unchanged, no 2 kept")


@pytest.mark.parametrize(
    "relaxation",
    [ConstantRelaxation(relaxation=0.5), AitkenRelaxation(relaxation=0.5)],
    ids=["constant", "aitken"],
)
def test_relaxation_passes_the_first_data_of_every_time_step_unchanged(
    relaxation: ConstantRelaxation,
) -> None:
    _passes(relaxation, 3)

    first_pass = relaxation([{"x": np.full(1, 7.0)}], 1)[0]["x"]

    np.testing.assert_array_equal(first_pass, [7.0], err_msg=type(relaxation).__name__)


def test_aitken_relaxation_restarts_after_a_round_off_residual() -> None:
    relaxation = AitkenRelaxation(relaxation=0.5)
    relaxation([{"x": np.full(1, 2.0)}], 1)
    relaxation([{"x": np.full(1, 2.0)}], 2)

    relaxed = relaxation([{"x": np.full(1, 4.0)}], 3)[0]["x"]

    np.testing.assert_allclose(relaxed, [3.0], rtol=1e-12, err_msg="2 + 0.5 * (4 - 2)")


def test_relaxation_keeps_the_names_and_shapes_of_the_data() -> None:
    sent = [{"T_sink": np.zeros(2), "h": np.ones(2)}, {"T_sink": np.full(2, 3.0)}]

    relaxed = ConstantRelaxation(relaxation=0.5)(sent, 1)

    assert [{name: values.tolist() for name, values in data.items()} for data in relaxed] == [
        {"T_sink": [0.0, 0.0], "h": [1.0, 1.0]},
        {"T_sink": [3.0, 3.0]},
    ]
