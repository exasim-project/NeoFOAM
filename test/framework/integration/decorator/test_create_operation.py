# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors
from neofoam.framework.context import Context, FieldUpdates
from neofoam.framework.decorator import condition, operation
from neofoam.framework.operations import (
    IterativeOp,
    Operation,
    SequentialOp,
    context_adapter,
)


class SomeClass:
    @operation
    def member_function(self, a: int) -> FieldUpdates:
        a += 1
        return FieldUpdates({"a": a})

    @operation
    def another_member_function(self, a: int, b: int) -> FieldUpdates:
        b += 2
        return FieldUpdates({"b": b})

    @operation
    def another_member_function_kwargs(
        self, a: int, *, b: int, c: float
    ) -> FieldUpdates:
        b += 2
        c += 3.0
        return FieldUpdates({"b": b, "c": c})


def test_context_adapter_step():
    """
    Tests the `context_adapter` decorator by verifying that it correctly
    injects the function parameters including dataclasses from a `Context`
    object and returns the a runnable function that updates the context fields.
    """

    sc = SomeClass()

    # create a new function that takes the context as input for a single parameter
    adapted_func = context_adapter(sc.member_function)

    ctx = Context(fields={"a": 0}, models={})
    # run the adapted function
    adapted_func(ctx)

    # verify that the context fields have been updated correctly
    assert ctx.fields["a"] == 1

    # create a new function that takes the context as input for several parameters
    adapted_func = context_adapter(sc.another_member_function)

    ctx = Context(fields={"a": 0, "b": 0}, models={})
    adapted_func(ctx)

    assert ctx.fields["b"] == 2

    # create a new function that takes the context as input for several parameters including kwargs
    adapted_func = context_adapter(sc.another_member_function_kwargs)

    ctx = Context(fields={"a": 0, "b": 0, "c": 0.0}, models={})
    adapted_func(ctx)

    assert ctx.fields["b"] == 2
    assert ctx.fields["c"] == 3.0


def test_context_adapter_condition():
    class ConditionClass:
        @condition
        def my_condition(self, a: int) -> bool:
            return a > 5

    cc = ConditionClass()

    adapted_func = context_adapter(cc.my_condition)

    ctx = Context(fields={"a": 10}, models={})
    condition_result = adapted_func(ctx)

    assert isinstance(condition_result, bool)
    assert condition_result

    ctx = Context(fields={"a": 3}, models={})
    condition_result = adapted_func(ctx)

    assert isinstance(condition_result, bool)
    assert not condition_result


def test_sequential_op_free_function():
    @operation
    def function1(a: int) -> int:
        a += 1
        return FieldUpdates({"a": a})

    # init from components
    seq_op = SequentialOp(func=context_adapter(function1))
    op1 = Operation(func=seq_op, operation_name="step1", operation_number=1)

    ctx = Context(fields={"a": 1}, models={})
    op1.run(ctx)
    assert ctx.fields["a"] == 2

    op2 = Operation.create_SeqOp(function1)
    ctx = Context(fields={"a": 1}, models={})
    op2.run(ctx)
    assert ctx.fields["a"] == 2


def test_sequential_op_member_function():
    class MyClass:
        @operation
        def function1(self, a: int) -> FieldUpdates:
            a += 1
            return FieldUpdates({"a": a})

    my_instance = MyClass()

    seq_op = SequentialOp(func=context_adapter(my_instance.function1))
    op1 = Operation(func=seq_op, operation_name="step1", operation_number=1)

    ctx = Context(fields={"a": 1}, models={})
    op1.run(ctx)
    assert ctx.fields["a"] == 2

    op2 = Operation.create_SeqOp(my_instance.function1)
    assert op2.operation_name == "function1"
    assert op2.operation_number is None

    ctx = Context(fields={"a": 1}, models={})
    op2.run(ctx)
    assert ctx.fields["a"] == 2


def test_iterative_op_member_function():
    class MyClass:
        @operation
        def function1(self, a: int) -> FieldUpdates:
            a += 1
            return FieldUpdates({"a": a})

        @condition
        def condition1(self, a: int) -> bool:
            return a < 5

    my_instance = MyClass()

    iter_op = IterativeOp(func=context_adapter(my_instance.condition1))
    increment_op = Operation.create_SeqOp(my_instance.function1)

    op1 = Operation(
        func=iter_op,
        operation_name="step1",
        operation_number=1,
        sub_operations=[increment_op],
    )
    assert op1.operation_name == "step1"

    ctx = Context(fields={"a": 0}, models={})
    op1.run(ctx)
    assert ctx.fields["a"] == 5
