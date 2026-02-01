# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""Basic DAG resolver functionality tests."""

from foamadapter.framework.operations import (
    Operation,
    OperationCollection,
    StepBuilder,
    SequentialOp,
    DAGResolver,
)
from .dag_test_helpers import increment_count, noop


def test_resolver_empty_inputs():
    """Test resolver with empty inputs."""
    builder = StepBuilder()
    model_ops = OperationCollection()

    resolver = DAGResolver()
    result = resolver.resolve(builder, model_ops)

    assert len(result.operations) == 0


def test_resolver_no_model_ops():
    """Test resolver with only StepBuilder operations."""
    builder = StepBuilder()
    operation = Operation(
        func=SequentialOp(increment_count),
        operation_name="op1",
    )
    builder.step(operation)

    model_ops = OperationCollection()

    resolver = DAGResolver()
    result = resolver.resolve(builder, model_ops)

    assert len(result.operations) == 1
    assert result.operations[0].operation_name == "op1"


def test_resolver_no_builder_ops():
    """Test resolver with only model operations."""
    builder = StepBuilder()

    model_op = Operation(
        func=SequentialOp(noop),
        operation_name="model_op",
        depends_on=[],
    )
    model_op.loop_context = "root"
    model_ops = OperationCollection([model_op])

    resolver = DAGResolver()
    result = resolver.resolve(builder, model_ops)

    assert len(result.operations) == 1
    assert result.operations[0].operation_name == "model_op"
