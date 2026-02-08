# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors
import pytest

from foamadapter.framework.decorator import (
    OpType,
    condition,
    decorated_member_functions,
    operation,
)


def test_decorator_free_function():
    @operation
    def free_step_function(a: float) -> float:
        a = a + 1.0
        return a

    @condition
    def free_condition_function() -> bool:
        return True

    assert free_step_function(1.0) == 2.0

    free_step_function._metadata.op_type == OpType.OPERATION
    free_step_function._metadata.name == "free_step_function"

    assert free_condition_function()

    free_condition_function._metadata.op_type == OpType.CONDITION
    free_condition_function._metadata.name == "free_condition_function"


def test_decorator_free_function_type_error():
    msg = "Return type of free_condition_function must be bool not a <class 'int'>"
    with pytest.raises(TypeError) as excinfo:

        @condition
        def free_condition_function() -> int:
            return 1

    assert msg in str(excinfo.value)

    msg = (
        "Function free_condition_function must have a return type annotation of 'bool'"
    )
    with pytest.raises(TypeError) as excinfo:

        @condition
        def free_condition_function():
            return True

    assert msg in str(excinfo.value)


def test_decorator_member_function():
    class SomeClass:
        @operation
        def member_function(self):
            return True

        @condition
        def member_condition(self) -> bool:
            return True

        def list_functions(self):
            funcs = decorated_member_functions(self)
            return funcs

    some_instance = SomeClass()
    assert some_instance.member_function()

    some_instance.member_function._metadata.op_type == OpType.OPERATION
    some_instance.member_function._metadata.name == "member_function"

    funcs = some_instance.list_functions()
    assert len(funcs) == 2
    assert funcs[0]._metadata.name == "member_function"
    assert funcs[1]._metadata.name == "member_condition"

    assert some_instance.member_condition()

    some_instance.member_condition._metadata.op_type == OpType.CONDITION
    some_instance.member_condition._metadata.name == "member_condition"
