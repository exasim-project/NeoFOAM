# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The final iteration of a coupling loop, as its leader decides it for every region.

Regions ``a``, ``b``, ``c`` in a row, one loop, c leads; ``write`` is marked
``final`` and runs in the final iteration only.
"""

from __future__ import annotations

import warnings
from typing import Any, Literal

import numpy as np
import pytest
from framework.coupling.regions import (
    CoupledInstance,
    Outcomes,
    RecordingCommChannel,
    add_coupled,
    context,
    contexts,
    couple,
    never,
    run,
    solve,
)

from neofoam.framework.context import Context, LoopEntry
from neofoam.framework.coupling import (
    Array,
    CouplingLoop,
    CouplingNotConvergedWarning,
    Exchange,
    FaceData,
)
from neofoam.framework.coupling.exchanges import participant_exchanges
from neofoam.framework.coupling.loop._leader import CouplingOutcome
from neofoam.framework.execution_observer import observe
from neofoam.framework.operations import Operation, SequentialOp
from neofoam.framework.types import OperationMetadata

EXCHANGES = [Exchange("a-b", (("a", "p"), ("b", "p"))), Exchange("b-c", (("b", "q"), ("c", "q")))]
PATCHES = {"a": ["p"], "b": ["p", "q"], "c": ["q"]}
SCHEMES = pytest.mark.parametrize("scheme", ["serial", "parallel"])


class _Data(FaceData):
    x: Array


class _Region:
    """A test region: solving sets its ports' value to ``fixed``; logs its iterations and writes."""

    def __init__(self, name: str, fixed: float) -> None:
        self.name = name
        self.fixed = fixed
        self.value = 0.0
        self.iterations: list[LoopEntry] = []
        self.writes = 0

    def solve(self, ctx: Context) -> None:
        self.iterations.append(ctx.loops[-1])
        self.value = self.fixed

    def write(self, ctx: Context) -> None:
        self.writes += 1

    def instance(self) -> CoupledInstance:
        return CoupledInstance(
            _Data, _Data, lambda: _Data(x=[self.value]), lambda data: None, np.zeros((1, 1))
        )

    def ops(self) -> list[Operation]:
        write = Operation(
            func=SequentialOp(self.write),
            metadata=OperationMetadata(op_name="write", final=True, depends_on=["solve_x"]),
        )
        return [solve("x", self.solve), write]


def _couple(
    scheme: Literal["serial", "parallel"] = "serial", **loop: Any
) -> tuple[dict[str, _Region], tuple[CouplingOutcome, ...]]:
    """Couple a, b and c in one loop and run one time step; the regions and c's outcomes."""
    regions = contexts(list(PATCHES), EXCHANGES, scheme=scheme, **loop)
    tests = {name: _Region(name, fixed) for name, fixed in zip(PATCHES, (1.0, 2.0, 3.0))}
    for name, ctx in regions.items():
        for patch in PATCHES[name]:
            add_coupled(ctx, patch, tests[name].instance())
    outcomes = Outcomes()
    with observe(outcomes):
        run({name: (ctx, tests[name].ops()) for name, ctx in regions.items()})
    assert couple(regions["c"]).controller is not None
    return tests, outcomes.of("c.coupling")


def _flags(region: _Region) -> list[tuple[int, bool, bool]]:
    return [(entry.iteration, entry.first, entry.final) for entry in region.iterations]


@SCHEMES
def test_every_region_runs_one_final_iteration_after_the_data_converged(scheme: Any) -> None:
    tests, outcomes = _couple(scheme, max_iterations=10)

    # the values change in the first iteration, not in the second: the third is final
    for region in tests.values():
        assert _flags(region) == [(0, True, False), (1, False, False), (2, False, True)]
        assert region.writes == 1
    assert outcomes == (CouplingOutcome(iterations=3, converged=True, residuals={"x": 0.0}),)


@SCHEMES
def test_the_last_iteration_max_iterations_allows_is_final_and_warns(scheme: Any) -> None:
    with pytest.warns(CouplingNotConvergedWarning, match="did not converge in 3 iterations"):
        tests, outcomes = _couple(scheme, convergence=[never], max_iterations=3)

    for region in tests.values():
        assert _flags(region) == [(0, True, False), (1, False, False), (2, False, True)]
        assert region.writes == 1
    assert [(o.iterations, o.converged) for o in outcomes] == [(3, False)]


@SCHEMES
def test_a_single_iteration_is_first_and_final_and_explicit(scheme: Any) -> None:
    """``max_iterations: 1`` is explicit coupling: no not-converged warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", CouplingNotConvergedWarning)
        tests, outcomes = _couple(scheme, max_iterations=1)

    for region in tests.values():
        assert _flags(region) == [(0, True, True)]
        assert region.writes == 1
    assert [o.iterations for o in outcomes] == [1]


def test_the_leader_sends_every_region_whether_the_iteration_is_final() -> None:
    comm_channel = RecordingCommChannel()
    loops = [CouplingLoop("coupling", ("a-b", "b-c"))]
    couplings = participant_exchanges(list(PATCHES), EXCHANGES, comm_channel, loops)
    regions = {name: context(name, couplings.get(name)) for name in PATCHES}
    tests = {name: _Region(name, 1.0) for name in PATCHES}
    for name, ctx in regions.items():
        for patch in PATCHES[name]:
            add_coupled(ctx, patch, tests[name].instance())
    run({name: (ctx, tests[name].ops()) for name, ctx in regions.items()})

    decisions = [
        (target, message[0], message[1])
        for _source, target, tag, message in comm_channel.sent
        if tag.endswith("/decision")
    ]
    assert decisions == [
        (region, go_on, final)
        for go_on, final in [(True, False), (True, False), (True, True), (False, False)]
        for region in ("a", "b")
    ]


def test_the_leader_relaxes_the_data_carried_into_the_final_iteration() -> None:
    iterations: list[int] = []

    def acceleration(sent: list[dict[str, Any]], iteration: int) -> list[dict[str, Any]]:
        iterations.append(iteration)
        return sent

    _tests, outcomes = _couple("parallel", max_iterations=10, acceleration={"a-b": acceleration})

    assert iterations == [1, 2, 3]
    assert outcomes[-1].iterations == 3
