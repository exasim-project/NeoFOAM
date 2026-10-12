# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Several solvers, one per region, stepped by one shared time loop.

Each region is a :class:`SolverRuntime` instantiated with ``region=`` and
initialized into its own :class:`Context`. A driver owns the time loop and runs
every region's time-step body (``@spec.time_step``) once per step.
"""

from dataclasses import replace
from typing import Union

from neofoam.framework.context import Context
from neofoam.framework.graph import DAGResolver
from neofoam.framework.operations import (
    ConditionalOp,
    IterativeOp,
    Operation,
    Operations,
    SequentialOp,
    StepBuilder,
)
from neofoam.framework.types import OperationMetadata

from .runtime import SolverRuntime


def region_step(runtime: SolverRuntime, ctx: Context) -> Operation:
    """One operation running the region's time-step body on the region's own context.

    The driver runs it inside its time loop with the driver's context, which the
    operation ignores: fields and models stay per region. The region reads the
    driver's time only if the driver shares it (``ctx.time = driver_ctx.time``).
    The region's graph edits (``ctx.graph``) apply to its time-step body.

    Example::

        left = slab.instantiate(region="left")
        left_ctx = left.initialize()
        left_ctx.time = driver_ctx.time
        time_builder.step(region_step(left, left_ctx))
    """
    builder = StepBuilder()
    runtime.time_step(ctx, builder)
    resolved = DAGResolver().resolve(builder, Operations())
    body: Operations = ctx.graph.apply(resolved).operations
    return Operation(
        func=SequentialOp(lambda _driver_ctx: body.run(ctx)),
        metadata=OperationMetadata(op_name=f"time_step_{runtime.region}"),
    )


def region_operations(runtime: SolverRuntime, ctx: Context) -> list[Operation]:
    """The operations of the region's time-step body, for the driver's own graph.

    Unlike :func:`region_step`, the driver steps every operation itself, so the
    graph edits of several regions can join one loop. Each operation runs on the
    region's context and is renamed ``<region>.<name>``, as are its
    ``depends_on`` / ``before`` targets and the fields it ``solves``. The driver
    takes over the region's edits with ``driver_ctx.graph.include_region``.

    Example::

        for operation in region_operations(left, left_ctx):
            time_builder.step(operation)
        driver_ctx.graph.include_region(left_ctx.graph, "left")
    """
    builder = StepBuilder()
    runtime.time_step(ctx, builder)
    resolved = DAGResolver().resolve(builder, Operations())
    return [_on_region(op, runtime.region or "", ctx) for op in resolved.operations]


def _on_region(op: Operation, region: str, ctx: Context) -> Operation:
    """*op* renamed into the region and run on the region's context, loops included."""

    def prefixed(names: list[str] | None) -> list[str] | None:
        return None if names is None else [f"{region}.{name}" for name in names]

    # The region's body is resolved already: dropping the operation numbers keeps
    # its order in the driver, where numbered operations would sort first.
    metadata = replace(
        op.metadata,
        op_name=f"{region}.{op.operation_name}",
        operation_number=None,
        depends_on=prefixed(op.metadata.depends_on),
        before=prefixed(op.metadata.before),
        solves=prefixed(op.metadata.solves) or [],
    )
    return Operation(
        func=_run_on(op.func, ctx),
        metadata=metadata,
        sub_operations=[_on_region(sub, region, ctx) for sub in op.sub_operations],
    )


def _run_on(
    func: Union[ConditionalOp, IterativeOp, SequentialOp], ctx: Context
) -> Union[ConditionalOp, IterativeOp, SequentialOp]:
    """*func* of the same kind, called with *ctx* whatever context it is run with."""
    if isinstance(func, IterativeOp):
        return IterativeOp(lambda _driver_ctx: func(ctx))
    if isinstance(func, ConditionalOp):
        return ConditionalOp(lambda _driver_ctx: func(ctx))
    return SequentialOp(lambda _driver_ctx: func(ctx))
