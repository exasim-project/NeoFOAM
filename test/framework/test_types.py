# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

import pytest

from neofoam.framework.types import OperationMetadata, OperationNumber


def test_step_number() -> None:
    # Basic comparisons
    assert OperationNumber("1") < OperationNumber("1.10")
    assert OperationNumber("1") == OperationNumber(1)
    assert OperationNumber("1.10.0") > OperationNumber("1.9.0")
    assert OperationNumber("1.10.0") < OperationNumber("1.10.1")
    assert OperationNumber("1") > OperationNumber("1.-1")

    # Equality and padding
    assert OperationNumber("1.10") == OperationNumber("1.10.0")
    assert OperationNumber("1.0.0") == OperationNumber("1")
    assert OperationNumber("1.0.1") != OperationNumber("1")

    # Mixed input types
    assert OperationNumber([1, 2, 3]) == OperationNumber("1.2.3")
    assert OperationNumber((1, 2)) < OperationNumber("1.2.1")

    # Greater / less / equal checks
    a = OperationNumber("2.0")
    b = OperationNumber("1.9.9")
    c = OperationNumber("2.0.0")
    assert a > b
    assert a >= c
    assert c <= a
    assert not a < b

    # Invalid inputs
    with pytest.raises(TypeError):
        OperationNumber(1.23)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        OperationNumber("1.a.3")


def test_operation_metadata() -> None:
    node = OperationMetadata(
        op_name="test_node",
        depends_on=["dep1", "dep2"],
        shape="circle",
        color="red",
        operation_number=OperationNumber("1.0.0"),
    )
    assert node.op_name == "test_node"
    assert node.depends_on == ["dep1", "dep2"]
    assert node.shape == "circle"
    assert node.color == "red"
    assert node.operation_number == OperationNumber("1.0.0")

    node_no_color = OperationMetadata(
        op_name="test_node2",
        depends_on=[],
        shape="square",
        operation_number=OperationNumber("1.0.1"),
    )
    assert node_no_color.op_name == "test_node2"
    assert node_no_color.depends_on == []
    assert node_no_color.shape == "square"
    assert node_no_color.color is None
    assert node_no_color.operation_number == OperationNumber("1.0.1")
