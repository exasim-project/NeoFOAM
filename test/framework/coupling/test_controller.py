# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The leader's controller on hand-made sides: convergence, relaxation, final pass, residuals."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
from framework.coupling.regions import never

from neofoam.framework.coupling import (
    CouplingConvergence,
    CouplingNode,
    CouplingNotConvergedWarning,
    RelativeConvergence,
    SideChange,
)
from neofoam.framework.coupling.controller import CouplingController, GroupSide, relative_changes


def test_relative_changes_is_the_largest_relative_change_per_field() -> None:
    changes = [
        SideChange("T", {"T": np.array([1.0, 0.0])}, {"T": np.array([2.0, 0.0])}),
        SideChange("T", {"q": np.array([0.0])}, {"q": np.array([0.0])}),
        SideChange("U", {"U": np.array([1.0])}, {"U": np.array([0.0])}),
    ]

    assert relative_changes(changes) == {"T": 0.5, "U": 1.0}


def _group() -> list[GroupSide]:
    return [
        GroupSide("a", "p", "x", ("b", "p"), "a-b"),
        GroupSide("b", "p", "x", ("a", "p"), "a-b"),
    ]


def test_a_convergence_on_a_field_no_side_sends_raises_as_the_controller_is_built() -> None:
    with pytest.raises(
        ValueError,
        match=r"coupling b.coupling: convergence RelativeConvergence on T, which no side of the "
        r"node sends \(its sides couple \['x'\]\)",
    ):
        CouplingController(
            _group(),
            "b.coupling",
            CouplingNode(
                "coupling", ("a-b",), convergence=(RelativeConvergence(data="T", limit=1e-5),)
            ),
        )


def test_a_field_reports_an_infinite_residual_until_a_pass_measured_it() -> None:
    controller = CouplingController(
        _group(), "b.coupling", CouplingNode("coupling", ("a-b",), convergence=(never,))
    )
    assert controller.residuals == {"x": math.inf}

    # the first decision starts the first pass: nothing measured yet
    controller.decide({("a", "p"): {"x": np.ones(1)}, ("b", "p"): {"x": np.ones(1)}}, 0)
    assert controller.residuals == {"x": math.inf}

    controller.decide({("a", "p"): {"x": np.ones(1)}, ("b", "p"): {"x": np.full(1, 2.0)}}, 1)
    assert controller.residuals == {"x": 0.5}


def test_relative_changes_keep_a_nan() -> None:
    changes = [
        SideChange("x", {"x": np.ones(1)}, {"x": np.full(1, 2.0)}),
        SideChange("x", {"x": np.ones(1)}, {"x": np.full(1, np.nan)}),
    ]

    assert math.isnan(relative_changes(changes)["x"])


def _sides() -> list[GroupSide]:
    """Two exchanges x and y between a and b."""
    return [
        GroupSide("a", "p", "x", ("b", "p"), "x"),
        GroupSide("b", "p", "x", ("a", "p"), "x"),
        GroupSide("a", "q", "y", ("b", "q"), "y"),
        GroupSide("b", "q", "y", ("a", "q"), "y"),
    ]


def _data(*values: float) -> dict[tuple[str, str], dict[str, Any]]:
    """The data of the sides of :func:`_sides`, in their order."""
    return {side.end: {"v": np.array([value])} for side, value in zip(_sides(), values)}


def test_an_acceleration_relaxes_only_the_sides_of_its_exchange() -> None:
    relaxed: list[list[float]] = []

    def halve(sent: list[dict[str, Any]], iteration: int) -> list[dict[str, Any]]:
        relaxed.append([float(data["v"][0]) for data in sent])
        return [{"v": data["v"] / 2} for data in sent]

    node = CouplingNode("n", ("x", "y"), scheme="parallel", acceleration={"x": halve})
    controller = CouplingController(_sides(), "b.n", node)

    carried = controller.decide(_data(1.0, 2.0, 10.0, 20.0), 0)

    assert carried is not None
    assert relaxed == [[1.0, 2.0]]
    assert [float(data["v"][0]) for data in carried.values()] == [0.5, 1.0, 10.0, 20.0]


def test_a_node_converges_once_every_measure_of_its_convergence_holds() -> None:
    verdicts = {"first": True, "second": False}

    class _Verdict(CouplingConvergence):
        name: str

        def __call__(self, changes: list[SideChange]) -> bool:
            return verdicts[self.name]

    node = CouplingNode(
        "n", ("x", "y"), convergence=(_Verdict(name="first"), _Verdict(name="second"))
    )
    controller = CouplingController(_sides(), "b.n", node)
    assert controller.decide(_data(1.0, 2.0, 3.0, 4.0), 0) is not None
    assert controller.decide(_data(1.0, 2.0, 3.0, 4.0), 1) is not None

    verdicts["second"] = True

    assert controller.decide(_data(1.0, 2.0, 3.0, 4.0), 2) is not None
    assert controller.final
    assert controller.decide(_data(1.0, 2.0, 3.0, 4.0), 3) is None
    assert controller.outcomes[-1].converged


def test_a_node_without_convergence_converges_once_the_data_is_unchanged() -> None:
    controller = CouplingController(_sides(), "b.n", CouplingNode("n", ("x", "y")))

    assert controller.decide(_data(1.0, 2.0, 3.0, 4.0), 0) is not None
    assert controller.decide(_data(1.0, 2.0, 3.0, 5.0), 1) is not None
    assert controller.decide(_data(1.0, 2.0, 3.0, 5.0), 2) is not None
    assert controller.final
    assert controller.decide(_data(1.0, 2.0, 3.0, 5.0), 3) is None


def test_the_controller_tells_which_pass_is_final() -> None:
    node = CouplingNode("coupling", ("a-b",), convergence=(never,), max_iterations=2)
    controller = CouplingController([], "c.coupling", node)

    assert controller.decide({}, 0) == {} and not controller.final
    assert controller.decide({}, 1) == {} and controller.final
    with pytest.warns(CouplingNotConvergedWarning) as warned:
        assert controller.decide({}, 2) is None
    # the warning points at the caller outside the framework
    assert warned[0].filename == __file__
    assert not controller.final
