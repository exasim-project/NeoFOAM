# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""One solver spec run as several regions, each a time-step body of a shared loop.

The ``counter`` solver adds its region's ``increment`` (``constant/<region>/
increment.yaml`` of ``cases/regions``, 2 on the left and 3 on the right) to a
``total`` field once per time step. The increments differ, so a region that read
the other's file, shared its context or ran on the driver's context shows up in
its total. A region's graph edits (``ctx.graph``) apply to its own time step
only: a loop of two passes around ``add_increment`` doubles that region's total.
``region_operations`` hands the same body to a driver as single operations named
``<region>.<op>``; a driver that includes the region's edits loops them the same way.
"""

from framework.conftest import MaxIterations
from framework.solver.region_counter import counter
from neofoam.framework.context import Context
from neofoam.framework.operations import IterativeOp, StepBuilder
from neofoam.framework.solver import region_operations, region_step


def _run_steps(steps: int, driver_ctx: Context) -> dict[str, Context]:
    region_ctxs: dict[str, Context] = {}
    operations = []
    for region in ("left", "right"):
        runtime = counter.instantiate(region=region)
        region_ctxs[region] = runtime.initialize()
        operations.append(region_step(runtime, region_ctxs[region]))
    for _ in range(steps):
        for operation in operations:
            operation.run(driver_ctx)
    return region_ctxs


def test_region_step_runs_on_the_region_context() -> None:
    region_ctxs = _run_steps(3, Context(fields={}, models={}))

    assert {region: ctx.fields["total"] for region, ctx in region_ctxs.items()} == {
        "left": 6.0,
        "right": 9.0,
    }


def test_region_step_leaves_the_driver_context_alone() -> None:
    driver_ctx = Context(fields={}, models={})

    _run_steps(3, driver_ctx)

    assert driver_ctx.fields == {}


def test_region_step_applies_the_region_graph_edits() -> None:
    left = counter.instantiate(region="left")
    left_ctx = left.initialize()
    left_ctx.graph.wrap(ops=left_ctx.graph.solving("total"), with_=IterativeOp(MaxIterations(2)))
    right = counter.instantiate(region="right")
    right_ctx = right.initialize()

    for operation in (region_step(left, left_ctx), region_step(right, right_ctx)):
        operation.run(Context(fields={}, models={}))

    assert (left_ctx.fields["total"], right_ctx.fields["total"]) == (4.0, 3.0)


def test_region_operations_prefix_the_names_with_the_region() -> None:
    left = counter.instantiate(region="left")

    operations = region_operations(left, left.initialize())

    assert [(op.operation_name, op.metadata.solves) for op in operations] == [
        ("left.add_increment", ["left.total"])
    ]


def test_region_operations_run_on_the_region_context() -> None:
    left = counter.instantiate(region="left")
    left_ctx = left.initialize()
    driver_ctx = Context(fields={}, models={})

    for operation in region_operations(left, left_ctx):
        operation.run(driver_ctx)

    assert (left_ctx.fields["total"], driver_ctx.fields) == (2.0, {})


def test_region_operations_join_the_loop_of_an_included_region_edit() -> None:
    left = counter.instantiate(region="left")
    left_ctx = left.initialize()
    left_ctx.graph.wrap(ops=left_ctx.graph.solving("total"), with_=IterativeOp(MaxIterations(2)))
    driver_ctx = Context(fields={}, models={})
    driver_ctx.graph.include_region(left_ctx.graph, "left")

    graph = driver_ctx.graph.apply(StepBuilder(region_operations(left, left_ctx)))
    graph.operations.run(driver_ctx)

    assert left_ctx.fields["total"] == 4.0
