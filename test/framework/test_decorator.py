# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors
import pytest

from typing import Any

from neofoam.framework.decorator import (
    condition,
    decorated_member_functions,
    operation,
)
from neofoam.framework.types import OpType


def test_decorator_free_function() -> None:
    @operation
    def free_step_function(a: float) -> float:
        a = a + 1.0
        return a

    @condition
    def free_condition_function() -> bool:
        return True

    assert free_step_function(1.0) == 2.0  # type: ignore[arg-type]

    assert free_step_function._metadata.op_type == OpType.OPERATION  # type: ignore[union-attr]
    assert free_step_function._metadata.name == "free_step_function"  # type: ignore[union-attr]

    assert free_condition_function()  # type: ignore[call-arg]

    assert free_condition_function._metadata.op_type == OpType.CONDITION  # type: ignore[union-attr]
    assert free_condition_function._metadata.name == "free_condition_function"  # type: ignore[union-attr]


def test_decorator_free_function_non_bool_return_annotation() -> None:
    msg = "Return type of free_condition_function must be bool not a <class 'int'>"
    with pytest.raises(TypeError) as excinfo:

        @condition
        def free_condition_function() -> int:
            return 1

    assert msg in str(excinfo.value)


def test_decorator_free_function_missing_return_annotation() -> None:
    msg = (
        "Function free_condition_function must have a return type annotation of 'bool'"
    )
    with pytest.raises(TypeError) as excinfo:

        @condition
        def free_condition_function():  # type: ignore[no-untyped-def]
            return True

    assert msg in str(excinfo.value)


def test_decorator_member_function() -> None:
    class SomeClass:
        @operation
        def member_function(self) -> bool:
            return True

        @condition
        def member_condition(self) -> bool:
            return True

        def list_functions(self) -> list[Any]:
            funcs = decorated_member_functions(self)
            return funcs

    some_instance = SomeClass()
    assert some_instance.member_function()  # type: ignore[misc]

    assert some_instance.member_function._metadata.op_type == OpType.OPERATION  # type: ignore[misc, union-attr]
    assert some_instance.member_function._metadata.name == "member_function"  # type: ignore[misc, union-attr]

    funcs = some_instance.list_functions()
    assert len(funcs) == 2
    assert funcs[0]._metadata.name == "member_function"
    assert funcs[1]._metadata.name == "member_condition"

    assert some_instance.member_condition()  # type: ignore[misc]

    assert some_instance.member_condition._metadata.op_type == OpType.CONDITION  # type: ignore[misc, union-attr]
    assert some_instance.member_condition._metadata.name == "member_condition"  # type: ignore[misc, union-attr]
