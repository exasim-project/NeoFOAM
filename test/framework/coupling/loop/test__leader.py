# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The Leader on hand-made streams: convergence, relaxation, final iteration, residuals."""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
import pytest
from framework.coupling.regions import never

from neofoam.framework.coupling import (
    CouplingConvergence,
    CouplingLoop,
    CouplingNotConvergedWarning,
    RelativeConvergence,
    StreamChange,
)
from neofoam.framework.coupling.loop._leader import (
    CouplingOutcome,
    GroupStream,
    Leader,
    relative_changes,
)


def test_relative_changes_is_the_largest_relative_change_per_data_field() -> None:
    changes = [
        StreamChange({"T": np.array([1.0, 0.0])}, {"T": np.array([2.0, 0.0])}),
        StreamChange({"T": np.array([1.0])}, {"T": np.array([1.0])}),
        StreamChange({"q": np.array([0.0])}, {"q": np.array([0.0])}),
        StreamChange({"U": np.array([1.0])}, {"U": np.array([0.0])}),
    ]

    assert relative_changes(changes) == {"T": 0.5, "q": 0.0, "U": 1.0}


def _group() -> list[GroupStream]:
    return [
        GroupStream("a", "p", "x", ("x",), ("b", "p"), "a-b"),
        GroupStream("b", "p", "x", ("x",), ("a", "p"), "a-b"),
    ]


def test_a_convergence_on_a_field_no_port_sends_raises_as_the_leader_is_built() -> None:
    with pytest.raises(
        ValueError,
        match=r"coupling b.coupling: convergence RelativeConvergence on T, which no port of the "
        r"loop sends \(they send \['x'\]\)",
    ):
        Leader(
            _group(),
            "b.coupling",
            CouplingLoop(
                "coupling", ("a-b",), convergence=(RelativeConvergence(data="T", limit=1e-5),)
            ),
            ("a", "b"),
        )


def test_a_field_reports_an_infinite_residual_until_an_iteration_measured_it() -> None:
    leader = Leader(
        _group(), "b.coupling", CouplingLoop("coupling", ("a-b",), convergence=(never,)), ("a", "b")
    )
    assert leader.residuals == {"x": math.inf}

    # the first decision starts the first iteration: nothing measured yet
    leader.decide({("a", "p", "x"): {"x": np.ones(1)}, ("b", "p", "x"): {"x": np.ones(1)}}, 0)
    assert leader.residuals == {"x": math.inf}

    leader.decide({("a", "p", "x"): {"x": np.ones(1)}, ("b", "p", "x"): {"x": np.full(1, 2.0)}}, 1)
    assert leader.residuals == {"x": 0.5}


def test_relative_changes_keep_a_nan() -> None:
    changes = [
        StreamChange({"x": np.ones(1)}, {"x": np.full(1, 2.0)}),
        StreamChange({"x": np.ones(1)}, {"x": np.full(1, np.nan)}),
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

    loop = CouplingLoop("n", ("x", "y"), scheme="parallel", acceleration={"x": halve})
    leader = Leader(_streams(), "b.n", loop, ("a", "b"))

    carried = leader.decide(_data(1.0, 2.0, 10.0, 20.0), 0)

    assert carried is not None
    assert relaxed == [[1.0, 2.0]]
    assert [float(data["v"][0]) for data in carried.values()] == [0.5, 1.0, 10.0, 20.0]


def test_a_loop_converges_once_every_measure_of_its_convergence_holds() -> None:
    verdicts = {"first": True, "second": False}

    class _Verdict(CouplingConvergence):
        name: str

        def __call__(self, changes: list[StreamChange]) -> bool:
            return verdicts[self.name]

    loop = CouplingLoop(
        "n", ("x", "y"), convergence=(_Verdict(name="first"), _Verdict(name="second"))
    )
    leader = Leader(_streams(), "b.n", loop, ("a", "b"))
    assert leader.decide(_data(1.0, 2.0, 3.0, 4.0), 0) is not None
    assert leader.decide(_data(1.0, 2.0, 3.0, 4.0), 1) is not None

    verdicts["second"] = True

    assert leader.decide(_data(1.0, 2.0, 3.0, 4.0), 2) is not None
    assert leader.final
    assert leader.decide(_data(1.0, 2.0, 3.0, 4.0), 3) is None
    assert leader.outcome is not None and leader.outcome.converged


def test_a_loop_without_convergence_converges_once_the_data_is_unchanged() -> None:
    leader = Leader(_streams(), "b.n", CouplingLoop("n", ("x", "y")), ("a", "b"))

    assert leader.decide(_data(1.0, 2.0, 3.0, 4.0), 0) is not None
    assert leader.decide(_data(1.0, 2.0, 3.0, 5.0), 1) is not None
    assert leader.decide(_data(1.0, 2.0, 3.0, 5.0), 2) is not None
    assert leader.final
    assert leader.decide(_data(1.0, 2.0, 3.0, 5.0), 3) is None


def test_the_leader_tells_which_iteration_is_final() -> None:
    loop = CouplingLoop("coupling", ("a-b",), convergence=(never,), max_iterations=2)
    leader = Leader([], "c.coupling", loop, ("c",))

    assert leader.decide({}, 0) == {} and not leader.final
    assert leader.decide({}, 1) == {} and leader.final
    with pytest.warns(CouplingNotConvergedWarning) as warned:
        assert leader.decide({}, 2) is None
    # the warning points at the caller outside the framework
    assert warned[0].filename == __file__
    assert not leader.final


def test_explicit_coupling_runs_one_iteration_without_a_not_converged_warning() -> None:
    loop = CouplingLoop("coupling", ("a-b",), convergence=(never,), max_iterations=1)
    leader = Leader(_group(), "b.coupling", loop, ("a", "b"))
    data = {("a", "p", "x"): {"x": np.ones(1)}, ("b", "p", "x"): {"x": np.ones(1)}}

    assert leader.decide(data, 0) is not None and leader.final
    with warnings.catch_warnings():
        warnings.simplefilter("error", CouplingNotConvergedWarning)
        assert leader.decide(data, 1) is None

    assert leader.outcome is not None and leader.outcome.iterations == 1


def test_a_stream_is_received_fresh_when_its_sender_steps_earlier_in_a_serial_loop() -> None:
    """``b`` only receives (it sends nothing): its rank comes from the loop's group."""
    streams = [GroupStream("a", "p", "x", ("x",), ("b", "p"), "a-b")]

    leader = Leader(streams, "b.n", CouplingLoop("n", ("a-b",)), ("a", "b"))

    assert leader.fresh(("a", "p", "x"))


def test_the_leader_keeps_the_outcome_of_the_last_time_step_only() -> None:
    loop = CouplingLoop("coupling", ("a-b",), max_iterations=1)
    leader = Leader(_group(), "b.coupling", loop, ("a", "b"))
    for value in (1.0, 3.0):
        data = {("a", "p", "x"): {"x": np.ones(1)}, ("b", "p", "x"): {"x": np.full(1, value)}}
        leader.decide(data, 0)
        leader.decide({**data, ("b", "p", "x"): {"x": np.full(1, value + 1.0)}}, 1)

    # b's x went 3 -> 4 in the second time step (1 -> 2 in the first); observers get both
    assert leader.outcome == CouplingOutcome(iterations=1, converged=False, residuals={"x": 0.25})
    assert not hasattr(leader, "outcomes")
