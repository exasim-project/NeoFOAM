# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Several solvers, one per region, stepped by one shared time loop.

Each region is a :class:`SolverRuntime` instantiated with ``region=`` and
initialized into its own :class:`Context`. A driver owns the time loop and runs
every region's time-step body (``@spec.time_step``) once per step.
"""

from neofoam.framework.context import Context
from neofoam.framework.graph import DAGResolver
from neofoam.framework.operations import Operation, Operations, SequentialOp, StepBuilder
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
