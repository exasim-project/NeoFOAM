# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The final pass of a coupling node, as its leader decides it for every region.

Regions ``a``, ``b``, ``c`` in a row, one node, c leads; ``write`` is marked
``final`` and runs in the final pass only.
"""

from __future__ import annotations

import warnings
from typing import Any, Literal

import numpy as np
import pytest
from framework.coupling.regions import (
    RecordingCommChannel,
    Side,
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
    CouplingNode,
    CouplingNotConvergedWarning,
    Exchange,
    FaceData,
)
from neofoam.framework.coupling.exchanges import region_couplings
from neofoam.framework.coupling.iteration._leader import CouplingController, CouplingOutcome
from neofoam.framework.operations import Operation, SequentialOp
from neofoam.framework.types import OperationMetadata

EXCHANGES = [Exchange("a-b", (("a", "p"), ("b", "p"))), Exchange("b-c", (("b", "q"), ("c", "q")))]
PATCHES = {"a": ["p"], "b": ["p", "q"], "c": ["q"]}
SCHEMES = pytest.mark.parametrize("scheme", ["serial", "parallel"])


class _Data(FaceData):
    x: Array


class _Region:
    """A test region: solving sets its sides' value to ``fixed``; it logs its passes and writes."""

    def __init__(self, name: str, fixed: float) -> None:
        self.name = name
        self.fixed = fixed
        self.value = 0.0
        self.passes: list[LoopEntry] = []
        self.writes = 0

    def solve(self, ctx: Context) -> None:
        self.passes.append(ctx.loops[-1])
        self.value = self.fixed

    def write(self, ctx: Context) -> None:
        self.writes += 1

    def side(self) -> Side:
        return Side(
            _Data, _Data, lambda: _Data(x=[self.value]), lambda data: None, np.zeros((1, 1))
        )

    def ops(self) -> list[Operation]:
        write = Operation(
            func=SequentialOp(self.write),
            metadata=OperationMetadata(op_name="write", final=True, depends_on=["solve_x"]),
        )
        return [solve("x", self.solve), write]


def _couple(
    scheme: Literal["serial", "parallel"] = "serial", **node: Any
) -> tuple[dict[str, _Region], CouplingController]:
    """Couple a, b and c in one node and run one time step; the regions and c's controller."""
    regions = contexts(list(PATCHES), EXCHANGES, scheme=scheme, **node)
    tests = {name: _Region(name, fixed) for name, fixed in zip(PATCHES, (1.0, 2.0, 3.0))}
    for name, ctx in regions.items():
        for patch in PATCHES[name]:
            add_coupled(ctx, patch, tests[name].side())
    run({name: (ctx, tests[name].ops()) for name, ctx in regions.items()})
    controller = couple(regions["c"]).controller
    assert controller is not None
    return tests, controller


def _flags(region: _Region) -> list[tuple[int, bool, bool]]:
    return [(entry.iteration, entry.first, entry.final) for entry in region.passes]


@SCHEMES
def test_every_region_runs_one_final_pass_after_the_data_converged(scheme: Any) -> None:
    tests, controller = _couple(scheme, max_iterations=10)

    # the values change in the first pass, not in the second: the third is final
    for region in tests.values():
        assert _flags(region) == [(0, True, False), (1, False, False), (2, False, True)]
        assert region.writes == 1
    assert controller.outcomes == (CouplingOutcome(passes=3, converged=True, residuals={"x": 0.0}),)


@SCHEMES
def test_the_last_pass_max_iterations_allows_is_final_and_warns(scheme: Any) -> None:
    with pytest.warns(CouplingNotConvergedWarning, match="did not converge in 3 passes"):
        tests, controller = _couple(scheme, convergence=[never], max_iterations=3)

    for region in tests.values():
        assert _flags(region) == [(0, True, False), (1, False, False), (2, False, True)]
        assert region.writes == 1
    assert [(o.passes, o.converged) for o in controller.outcomes] == [(3, False)]


@SCHEMES
def test_a_single_pass_is_first_and_final_and_explicit(scheme: Any) -> None:
    """``max_iterations: 1`` is explicit coupling: no not-converged warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", CouplingNotConvergedWarning)
        tests, controller = _couple(scheme, max_iterations=1)

    for region in tests.values():
        assert _flags(region) == [(0, True, True)]
        assert region.writes == 1
    assert [o.passes for o in controller.outcomes] == [1]


def test_the_leader_sends_every_region_whether_the_pass_is_final() -> None:
    comm_channel = RecordingCommChannel()
    nodes = [CouplingNode("coupling", ("a-b", "b-c"))]
    couplings = region_couplings(list(PATCHES), EXCHANGES, comm_channel, nodes)
    regions = {name: context(name, couplings.get(name)) for name in PATCHES}
    tests = {name: _Region(name, 1.0) for name in PATCHES}
    for name, ctx in regions.items():
        for patch in PATCHES[name]:
            add_coupled(ctx, patch, tests[name].side())
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


def test_the_controller_relaxes_the_data_carried_into_the_final_pass() -> None:
    iterations: list[int] = []

    def acceleration(sent: list[dict[str, Any]], iteration: int) -> list[dict[str, Any]]:
        iterations.append(iteration)
        return sent

    _tests, controller = _couple("parallel", max_iterations=10, acceleration={"a-b": acceleration})

    assert iterations == [1, 2, 3]
    assert controller.outcomes[-1].passes == 3
