# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The leader's controller on hand-made streams: convergence, relaxation, final pass, residuals."""

from __future__ import annotations

import math
import warnings
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
from neofoam.framework.coupling.controller import (
    CouplingController,
    GroupStream,
    relative_changes,
)


def test_relative_changes_is_the_largest_relative_change_per_data_field() -> None:
    changes = [
        SideChange({"T": np.array([1.0, 0.0])}, {"T": np.array([2.0, 0.0])}),
        SideChange({"T": np.array([1.0])}, {"T": np.array([1.0])}),
        SideChange({"q": np.array([0.0])}, {"q": np.array([0.0])}),
        SideChange({"U": np.array([1.0])}, {"U": np.array([0.0])}),
    ]

    assert relative_changes(changes) == {"T": 0.5, "q": 0.0, "U": 1.0}


def _group() -> list[GroupStream]:
    return [
        GroupStream("a", "p", "x", ("x",), ("b", "p"), "a-b"),
        GroupStream("b", "p", "x", ("x",), ("a", "p"), "a-b"),
    ]


def test_a_convergence_on_a_field_no_side_sends_raises_as_the_controller_is_built() -> None:
    with pytest.raises(
        ValueError,
        match=r"coupling b.coupling: convergence RelativeConvergence on T, which no port of the "
        r"node sends \(they send \['x'\]\)",
    ):
        CouplingController(
            _group(),
            "b.coupling",
            CouplingNode(
                "coupling", ("a-b",), convergence=(RelativeConvergence(data="T", limit=1e-5),)
            ),
            ("a", "b"),
        )


def test_a_field_reports_an_infinite_residual_until_a_pass_measured_it() -> None:
    controller = CouplingController(
        _group(), "b.coupling", CouplingNode("coupling", ("a-b",), convergence=(never,)), ("a", "b")
    )
    assert controller.residuals == {"x": math.inf}

    # the first decision starts the first pass: nothing measured yet
    controller.decide({("a", "p", "x"): {"x": np.ones(1)}, ("b", "p", "x"): {"x": np.ones(1)}}, 0)
    assert controller.residuals == {"x": math.inf}

    controller.decide(
        {("a", "p", "x"): {"x": np.ones(1)}, ("b", "p", "x"): {"x": np.full(1, 2.0)}}, 1
    )
    assert controller.residuals == {"x": 0.5}


def test_relative_changes_keep_a_nan() -> None:
    changes = [
        SideChange({"x": np.ones(1)}, {"x": np.full(1, 2.0)}),
        SideChange({"x": np.ones(1)}, {"x": np.full(1, np.nan)}),
    ]

    assert math.isnan(relative_changes(changes)["x"])


def _streams() -> list[GroupStream]:
    """Two exchanges x and y between a and b."""
    return [
        GroupStream("a", "p", "v", ("v",), ("b", "p"), "x"),
        GroupStream("b", "p", "v", ("v",), ("a", "p"), "x"),
        GroupStream("a", "q", "v", ("v",), ("b", "q"), "y"),
        GroupStream("b", "q", "v", ("v",), ("a", "q"), "y"),
    ]


def _data(*values: float) -> dict[tuple[str, str, str], dict[str, Any]]:
    """The data of the streams of :func:`_streams`, in their order."""
    return {stream.id: {"v": np.array([value])} for stream, value in zip(_streams(), values)}


def test_an_acceleration_relaxes_only_the_streams_of_its_exchange() -> None:
    relaxed: list[list[float]] = []

    def halve(sent: list[dict[str, Any]], iteration: int) -> list[dict[str, Any]]:
        relaxed.append([float(data["v"][0]) for data in sent])
        return [{"v": data["v"] / 2} for data in sent]

    node = CouplingNode("n", ("x", "y"), scheme="parallel", acceleration={"x": halve})
    controller = CouplingController(_streams(), "b.n", node, ("a", "b"))

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
    controller = CouplingController(_streams(), "b.n", node, ("a", "b"))
    assert controller.decide(_data(1.0, 2.0, 3.0, 4.0), 0) is not None
    assert controller.decide(_data(1.0, 2.0, 3.0, 4.0), 1) is not None

    verdicts["second"] = True

    assert controller.decide(_data(1.0, 2.0, 3.0, 4.0), 2) is not None
    assert controller.final
    assert controller.decide(_data(1.0, 2.0, 3.0, 4.0), 3) is None
    assert controller.outcomes[-1].converged


def test_a_node_without_convergence_converges_once_the_data_is_unchanged() -> None:
    controller = CouplingController(_streams(), "b.n", CouplingNode("n", ("x", "y")), ("a", "b"))

    assert controller.decide(_data(1.0, 2.0, 3.0, 4.0), 0) is not None
    assert controller.decide(_data(1.0, 2.0, 3.0, 5.0), 1) is not None
    assert controller.decide(_data(1.0, 2.0, 3.0, 5.0), 2) is not None
    assert controller.final
    assert controller.decide(_data(1.0, 2.0, 3.0, 5.0), 3) is None


def test_the_controller_tells_which_pass_is_final() -> None:
    node = CouplingNode("coupling", ("a-b",), convergence=(never,), max_iterations=2)
    controller = CouplingController([], "c.coupling", node, ("c",))

    assert controller.decide({}, 0) == {} and not controller.final
    assert controller.decide({}, 1) == {} and controller.final
    with pytest.warns(CouplingNotConvergedWarning) as warned:
        assert controller.decide({}, 2) is None
    # the warning points at the caller outside the framework
    assert warned[0].filename == __file__
    assert not controller.final


def test_explicit_coupling_runs_one_pass_without_a_not_converged_warning() -> None:
    node = CouplingNode("coupling", ("a-b",), convergence=(never,), max_iterations=1)
    controller = CouplingController(_group(), "b.coupling", node, ("a", "b"))
    data = {("a", "p", "x"): {"x": np.ones(1)}, ("b", "p", "x"): {"x": np.ones(1)}}

    assert controller.decide(data, 0) is not None and controller.final
    with warnings.catch_warnings():
        warnings.simplefilter("error", CouplingNotConvergedWarning)
        assert controller.decide(data, 1) is None

    assert controller.outcomes[-1].passes == 1


def test_a_stream_is_received_fresh_when_its_sender_steps_earlier_in_a_serial_node() -> None:
    """``b`` only receives (it sends nothing): its rank comes from the node's group."""
    streams = [GroupStream("a", "p", "x", ("x",), ("b", "p"), "a-b")]

    controller = CouplingController(streams, "b.n", CouplingNode("n", ("a-b",)), ("a", "b"))

    assert controller.fresh(("a", "p", "x"))
