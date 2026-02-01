# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""Complex integration tests for DAG resolver."""

from foamadapter.framework.operations import (
    Operation,
    OperationCollection,
    StepBuilder,
    SequentialOp,
    IterativeOp,
    DAGResolver,
)
from .dag_test_helpers import (
    set_time_step,
    increment_time,
    write_output,
    momentum,
    continuity,
    update_buoyancy,
    turbulence_correction,
    time_loop_condition,
    inner_loop_condition,
)


def test_solver_algorithm_model_integration():
    """Test realistic solver + algorithm + model scenario."""
    # Build StepBuilder (solver + algorithm)
    builder = StepBuilder()
    time_loop = Operation(
        func=IterativeOp(time_loop_condition),
        operation_name="time_loop",
    )

    with builder.loop(time_loop) as tloop:
        tloop.step(
            Operation(
                func=SequentialOp(set_time_step),
                operation_name="set_time_step",
            )
        )
        tloop.step(
            Operation(
                func=SequentialOp(increment_time),
                operation_name="increment_time",
                depends_on=["set_time_step"],
            )
        )

        inner_loop = Operation(
            func=IterativeOp(inner_loop_condition),
            operation_name="inner_loop",
        )
        with tloop.loop(inner_loop) as iloop:
            iloop.step(
                Operation(
                    func=SequentialOp(momentum),
                    operation_name="momentum",
                )
            )
            iloop.step(
                Operation(
                    func=SequentialOp(continuity),
                    operation_name="continuity",
                    depends_on=["momentum"],
                )
            )

        tloop.step(
            Operation(
                func=SequentialOp(write_output),
                operation_name="write_output",
            )
        )

    # Model operations
    buoyancy_op = Operation(
        func=SequentialOp(update_buoyancy),
        operation_name="update_buoyancy",
        depends_on=[],
    )
    buoyancy_op.loop_context = "inner_loop"

    turbulence_op = Operation(
        func=SequentialOp(turbulence_correction),
        operation_name="turbulence_correction",
        depends_on=["continuity"],
    )
    turbulence_op.loop_context = "inner_loop"

    model_ops = OperationCollection([buoyancy_op, turbulence_op])

    # Resolve
    resolver = DAGResolver()
    result = resolver.resolve(builder, model_ops)

    # Verify structure using nested dict-like access
    assert len(result.operations) == 1
    time_loop = result.operations[0]
    assert time_loop.operation_name == "time_loop"

    # Get time_loop operations by name
    time_loop_ops = {op.operation_name: op for op in time_loop.sub_operations}
    assert set(time_loop_ops.keys()) == {
        "set_time_step",
        "increment_time",
        "inner_loop",
        "write_output",
    }

    # Get inner_loop operation names as list for indexing
    inner_loop = [
        op.operation_name for op in time_loop_ops["inner_loop"].sub_operations
    ]
    assert len(inner_loop) == 4

    # Verify ordering by index
    # Operations with dependencies come first in dependency order,
    # operations without dependencies come last
    assert inner_loop[0] == "momentum"
    assert inner_loop[1] == "continuity"  # depends on momentum
    assert inner_loop[2] == "turbulence_correction"  # depends on continuity
    assert inner_loop[3] == "update_buoyancy"  # no dependencies, comes last
