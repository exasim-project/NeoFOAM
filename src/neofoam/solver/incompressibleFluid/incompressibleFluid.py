# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""incompressibleFluid — solver entrypoint (minimal port).

Adapted from ``feat/python_solvers`` to the SolverSpec/ModelSpec API in
``stack/python_arch``. Only PIMPLE is wired up; SIMPLE/PISO/boussinesq/SA
and the CFLCondition adjust-time-step path from the source branch are
omitted in this minimal version.
"""

from typing import Annotated, Any, Optional

from pybFoam import Info

from neofoam.framework.context import Context, FieldUpdates
from neofoam.framework.graph import DAGResolver
from neofoam.framework.initialization import Depends, StagedInitRunner
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
    """Helper class for managing the main time loop predicate."""

    def __call__(self, ctx: Context) -> bool:
        return bool(ctx.runtime.run())


incompressibleFluid = Solver("incompressibleFluid")


@incompressibleFluid.initializer
def initialize(
    self: Any, init: Annotated[StagedInitRunner, Depends(create_init)]
) -> Context:
    """Initialize using create_init factory with dependency injection."""
    return init.run()


@incompressibleFluid.execution_graph_step
def execution_graph(
    self: Any,
    domain_name: Optional[str] = None,
) -> tuple[StepBuilder, Operations]:
    """Build the time-loop + PIMPLE inner-loop execution graph."""
    _ = domain_name

    ops = self.operations
    builder = StepBuilder()

    # The pressure-velocity algorithm (pimple) is the first core model.
    # We pass the spec as its own runtime — it stores configured
    # algorithm_type / use_boussinesq attributes directly.
    algorithm_model = self.state.core_models[0]
    algo_ops = Operations(algorithm_model._build_operations_for(algorithm_model))

    time_loop_op = Operation(
        func=IterativeOp(TimeLoop()),
        metadata=OperationMetadata(op_name="time_loop"),
    )

    with builder.loop(time_loop_op) as time_builder:
        time_builder.step(ops["increment_time"])

        with time_builder.loop(algo_ops["inner_loop"]) as inner_builder:
            inner_builder.step(algo_ops["momentum"])
            inner_builder.step(algo_ops["continuity"])
            inner_builder.step(ops["turbulence_correction"])

        time_builder.step(ops["write_output"])

    # Collect operations from optional models (empty by default).
    model_ops = Operations()
    for opt in self.state.optional_models:
        model_ops.add(opt.operations)

    return builder, model_ops


def run(
    argv: Optional[list[str]] = None,
    log_file: Optional[Any] = None,
) -> Context:
    """Run one full simulation and return the final Context.

    If ``log_file`` is given, fd 1 (stdout) is redirected to that file
    for the duration of the solve so C++ ``Info`` output ends up there
    instead of the calling process's stdout.
    """
    import os
    import sys
    from pathlib import Path

    redirect = log_file is not None
    saved_fd: Optional[int] = None
    if redirect:
        log_path = Path(str(log_file))
        sys.stdout.flush()
        saved_fd = os.dup(1)
        log_fd = os.open(str(log_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        os.dup2(log_fd, 1)
        os.close(log_fd)

    try:
        solver = incompressibleFluid.instantiate(argv=argv or [])
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


@incompressibleFluid.operation()
def increment_time(self: Any, ctx: Context) -> None:
    """Print current simulation time and increment."""
    Info(f"Time = {ctx.runtime.timeName()}")
    ctx.runtime.increment()


@incompressibleFluid.operation(depends_on=["continuity"])
def turbulence_correction(
    self: Any,
    laminarTransport: Annotated[Optional[Any], "models"],
    turbulence: Annotated[Optional[Any], "models"],
) -> FieldUpdates:
    """Correct laminar transport + turbulence after pressure-velocity coupling."""
    if laminarTransport is not None:
        laminarTransport.correct()
    if turbulence is not None:
        turbulence.correct()
    return FieldUpdates({})


@incompressibleFluid.operation(depends_on=["turbulence_correction"])
def write_output(self: Any, ctx: Context) -> None:
    """Write fields to disk and report execution time."""
    ctx.runtime.write(True)
    ctx.runtime.printExecutionTime()
