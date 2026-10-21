# SPDX-License-Identifier: GPL-3.0-or-later
# Reference implementation matching doc/tutorials/03-build-a-solver.

from typing import Annotated, Any, Optional

from pybFoam import (
    Info,
    fvm,
    fvScalarMatrix,
    surfaceScalarField,
    volScalarField,
    volVectorField,
)

from neofoam import Depends, FieldUpdates, Solver, StagedInit
from neofoam.framework.context import Context
from neofoam.framework.operations import (
    IterativeOp,
    Operation,
    Operations,
    StepBuilder,
)
from neofoam.framework.types import OperationMetadata

from .create_fields import create_init


class TimeLoop:
    def __call__(self, ctx: Context) -> bool:
        return bool(ctx.time.run())


scalar_transport = Solver("scalar_transport")


@scalar_transport.initializer
def initialize(
    self: Any,
    init: Annotated[StagedInit, Depends(create_init)],
) -> Context:
    return init.run()


@scalar_transport.execution_graph_step
def execution_graph(
    self: Any,
    domain_name: Optional[str] = None,
) -> tuple[StepBuilder, Operations]:
    ops = self.operations

    builder = StepBuilder()
    time_loop_op = Operation(
        func=IterativeOp(TimeLoop()),
        metadata=OperationMetadata(op_name="time_loop"),
    )
    with builder.loop(time_loop_op) as time_builder:
        time_builder.step(ops["increment_time"])
        time_builder.step(ops["solve_T"])
        time_builder.step(ops["write_output"])

    return builder, Operations()


@scalar_transport.operation()
def increment_time(self: Any, ctx: Context) -> None:
    Info(f"Time = {ctx.time.timeName()}")
    ctx.time.increment()


@scalar_transport.operation(depends_on=["increment_time"])
def solve_T(
    self: Any,
    T: volScalarField,
    U: volVectorField,
    phi: surfaceScalarField,
    D: volScalarField,
) -> FieldUpdates:
    TEqn = fvScalarMatrix(fvm.ddt(T) + fvm.div(phi, T) - fvm.laplacian(D, T))
    TEqn.solve()
    return FieldUpdates({"T": T})


@scalar_transport.operation(depends_on=["solve_T"])
def write_output(self: Any, ctx: Context) -> None:
    ctx.time.write(True)
    ctx.time.printExecutionTime()
