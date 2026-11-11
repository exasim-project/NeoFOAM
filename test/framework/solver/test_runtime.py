# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

# NB: no ``from __future__ import annotations`` — the dependency resolver matches
# ``param.annotation is Context``.

"""A runtime solves one region: its context, case files and ``Depends`` cache are its own.

``shared`` records a cached ``Depends`` (the context's region); the regions run
one after the other, each on its own thread, so a cache on the spec would hand
the second region the first one's value. ``cases/regions`` holds a different
increment per region and at the single-region path, so a wrong file shows.
``uncoupling`` adds no coupling model, so a region the case couples raises.
"""

import threading
from typing import Annotated, Any, Optional, Sequence

import pytest

from framework.solver.region_counter import counter
from neofoam.framework.context import Context
from neofoam.framework.coupling import CouplingNode, InProcessCommChannel, Link
from neofoam.framework.coupling.region import region_couplings
from neofoam.framework.initialization.depends import Depends
from neofoam.framework.operations import Operation, Operations, SequentialOp, StepBuilder
from neofoam.framework.solver import Solver
from neofoam.framework.types import OperationMetadata


def region_of(ctx: Context) -> Any:
    return ctx.region


shared = Solver("shared_by_regions")


@shared.initializer
def _initialize(self: Any) -> Context:
    return Context(fields={}, models={"seen": []})


@shared.execution_graph_step
def _execution_graph(self: Any, ctx: Context) -> tuple[StepBuilder, Operations]:
    builder = StepBuilder()
    builder.step(self.operations["record"])
    return builder, Operations()


@shared.operation()
def record(self: Any, ctx: Context, region: Annotated[Any, Depends(region_of)]) -> None:
    ctx.models["seen"].append(region)


def _run(region: str, contexts: dict[str, Context]) -> None:
    runtime = shared.instantiate(region=region)
    ctx = runtime.initialize()
    runtime.resolve_operations(ctx).run(ctx)
    contexts[region] = ctx


def test_two_regions_of_one_solver_spec_resolve_their_own_cached_dependency() -> None:
    contexts: dict[str, Context] = {}
    for region in ("left", "right"):
        thread = threading.Thread(target=_run, args=(region, contexts))
        thread.start()
        thread.join()

    assert {region: ctx.models["seen"] for region, ctx in contexts.items()} == {
        "left": ["left"],
        "right": ["right"],
    }


@pytest.mark.parametrize(("region", "expected"), [("left", 2.0), ("right", 3.0), (None, 1.0)])
def test_runtime_region_selects_the_region_configs(region: Optional[str], expected: float) -> None:
    ctx = counter.instantiate(region=region).initialize()

    assert ctx.models["increment"] == expected


def test_runtime_initialize_hands_its_region_to_the_context() -> None:
    ctx = counter.instantiate(region="left").initialize()

    assert ctx.region == "left"


def test_runtime_run_initializes_and_runs_its_region() -> None:
    ctx = counter.instantiate(region="right").run()

    assert ctx.region == "right"
    assert ctx.fields["total"] == 3.0


uncoupling = Solver("uncoupling")


@uncoupling.initializer
def _initialize_uncoupling(self: Any) -> Context:
    return Context(fields={}, models={}, region=self.region)


@uncoupling.execution_graph_step
def _execution_graph_uncoupling(self: Any, ctx: Context) -> tuple[StepBuilder, Operations]:
    builder = StepBuilder()
    builder.step(self.operations["step"])
    return builder, Operations()


@uncoupling.operation()
def step(self: Any, ctx: Context) -> None:
    pass


def test_a_region_the_case_does_not_couple_resolves_without_a_coupling_model() -> None:
    runtime = uncoupling.instantiate(region="left")
    ctx = runtime.initialize()

    assert [op.operation_name for op in runtime.resolve_operations(ctx)] == ["step"]


def test_a_coupled_region_whose_solver_added_no_coupling_model_raises() -> None:
    link = Link("x", (("left", "p"), ("right", "p")))
    nodes = [CouplingNode("n", ("x",))]
    coupling = region_couplings(["left", "right"], [link], InProcessCommChannel(), nodes)["left"]
    runtime = uncoupling.instantiate(region="left", coupling=coupling)
    ctx = runtime.initialize()

    with pytest.raises(
        RuntimeError,
        match=r"region left is coupled in regionCoupling.yaml, but solver uncoupling added no "
        r"coupling model; add ctx.models\['coupling'\] = Coupling\(self.coupling\) in initialize",
    ):
        runtime.resolve_operations(ctx)


class _Contribution:
    """A model that adds ``extra`` to its region's graph and records the resolved graph."""

    def __init__(self) -> None:
        self.checked: list[str] = []

    def graph_operations(self, ctx: Context, graph: Sequence[Operation]) -> Operations:
        return Operations(
            [
                Operation(
                    func=SequentialOp(lambda ctx: None), metadata=OperationMetadata(op_name="extra")
                )
            ]
        )

    def check_graph(self, graph: Operations) -> None:
        self.checked = [op.operation_name for op in graph]


def test_a_model_contributing_to_the_graph_adds_its_operations_and_checks_the_result() -> None:
    contribution = _Contribution()
    runtime = uncoupling.instantiate()
    ctx = runtime.initialize()
    ctx.models["anything"] = contribution

    names = [op.operation_name for op in runtime.resolve_operations(ctx)]

    assert names == ["step", "extra"]
    assert contribution.checked == names
