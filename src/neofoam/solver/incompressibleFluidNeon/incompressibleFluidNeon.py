# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""
incompressibleFluidNeon — NeoN-backed incompressible Navier-Stokes solver.

Uses the framework Solver/StagedInit/ModelSpec pattern with NeoN bindings
for equation assembly and solving. Currently supports PISO (laminar) only.
"""

from typing import Annotated, Any, Optional

import neon._neon as nn
from neofoam import neofoam_bindings as nfb
from pybFoam import Info

from neofoam.framework.context import Context
from neofoam.framework.graph import DAGResolver
from neofoam.framework.initialization import Depends, StagedInit
from neofoam.framework.operations import (
    IterativeOp,
    Operation,
    Operations,
    StepBuilder,
)
from neofoam.framework.solver import Solver
from neofoam.framework.types import OperationMetadata
from .create_fields import create_init


class TimeLoop:
    """Helper class for managing the main time loop."""

    def __call__(self, ctx: Context) -> bool:
        return bool(ctx.runtime.run())


# Create Solver instance
incompressibleFluidNeon = Solver("incompressibleFluidNeon")


@incompressibleFluidNeon.initializer
def initialize(self: Any, init: Annotated[StagedInit, Depends(create_init)]) -> Context:
    """Initialize using create_init factory with dependency injection."""
    ctx = init.run()
    return ctx


@incompressibleFluidNeon.execution_graph_step
def execution_graph(
    self: Any,
    domain_name: Optional[str] = None,
) -> tuple[StepBuilder, Operations]:
    """Build solver structure and collect model operations."""
    _ = domain_name

    ops = self.operations
    builder = StepBuilder()

    # Get the PISO algorithm model from core_models
    algorithm_model = self.state.core_models[0]
    algo_ops = Operations(algorithm_model._build_operations_for(algorithm_model))

    # Time loop structure
    time_loop_op = Operation(
        func=IterativeOp(TimeLoop()),
        metadata=OperationMetadata(op_name="time_loop"),
    )

    with builder.loop(time_loop_op) as time_builder:
        time_builder.step(ops["set_time_step"])
        time_builder.step(ops["increment_time"])

        # PISO inner loop (single pass)
        with time_builder.loop(algo_ops["inner_loop"]) as inner_builder:
            inner_builder.step(algo_ops["momentum"])
            inner_builder.step(algo_ops["continuity"])

        time_builder.step(ops["write_output"])

    return builder, Operations()


def run(
    argv: Optional[list[str]] = None,
    log_file: Optional[Any] = None,
) -> Context:
    """Run the complete simulation with NeoN lifecycle management."""
    import os
    import sys
    from pathlib import Path

    effective_argv = argv or ["incompressibleFluidNeon"]

    nn.initialize(effective_argv)

    saved_fd: Optional[int] = None
    redirect = log_file is not None
    if redirect:
        log_path = Path(str(log_file))
        sys.stdout.flush()
        saved_fd = os.dup(1)
        log_fd = os.open(str(log_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        os.dup2(log_fd, 1)
        os.close(log_fd)

    try:
        solver = incompressibleFluidNeon.instantiate(argv=effective_argv)
        ctx = solver.initialize()

        Info("Starting time loop")

        builder, model_ops = solver.execution_graph()
        resolver = DAGResolver()
        resolved = resolver.resolve(builder, model_ops)

        resolved.operations.run(ctx)

        Info("End")

        return ctx
    finally:
        if redirect and saved_fd is not None:
            sys.stdout.flush()
            os.dup2(saved_fd, 1)
            os.close(saved_fd)
        nn.finalize()


@incompressibleFluidNeon.operation()
def set_time_step(
    self: Any,
    ctx: Context,
    neon_runtime: Annotated[Optional[Any], "models"],
    cfl_condition: Annotated[Optional[Any], "models"],
) -> None:
    """Compute Courant number and synchronize runtimes."""
    if neon_runtime is not None:
        phi = ctx.fields["phi"]
        rt = neon_runtime
        max_co, mean_co = nn.compute_co_num(rt.nf_mesh, phi.internal_vector(), rt.dt)
        Info(f"Courant Number mean: {mean_co:.6f} max: {max_co:.6f}")
        nfb.sync_run_times(ctx.runtime, rt, max_co)

    if cfl_condition is not None:
        cfl_condition(ctx)


@incompressibleFluidNeon.operation()
def increment_time(self: Any, ctx: Context) -> None:
    """Rotate old times, print current time, and increment."""
    U = ctx.fields["U"]
    phi = ctx.fields["phi"]

    nn.rotate_old_times(U)
    nn.rotate_old_times(phi)

    Info(f"Time = {ctx.runtime.timeName()}")
    ctx.runtime.increment()


@incompressibleFluidNeon.operation(depends_on=["continuity"])
def write_output(
    self: Any,
    ctx: Context,
    neon_runtime: Annotated[Optional[Any], "models"],
) -> None:
    """Write fields to disk via NeoN field writers."""
    if ctx.runtime.outputTime():
        rt = neon_runtime
        if rt is not None:
            nfb.write_scalar_field(ctx.fields["p"], rt)
            nfb.write_vector_field(ctx.fields["U"], rt)

    ctx.runtime.printExecutionTime()
