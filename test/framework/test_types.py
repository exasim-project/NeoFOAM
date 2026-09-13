# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

import pytest

from neofoam.framework.types import OperationMetadata, OperationNumber


@pytest.mark.parametrize(
    "smaller, larger",
    [
        ("1", "1.10"),
        ("1.9.0", "1.10.0"),
        ("1.10.0", "1.10.1"),
        ("1.-1", "1"),
        ("1.9.9", "2.0"),
        ((1, 2), "1.2.1"),
    ],
)
def test_operation_number_orders_by_value(smaller, larger) -> None:
    assert OperationNumber(smaller) < OperationNumber(larger)
    assert OperationNumber(larger) > OperationNumber(smaller)
    assert not OperationNumber(larger) < OperationNumber(smaller)


@pytest.mark.parametrize(
    "left, right, are_equal",
    [
        ("1", 1, True),
        ([1, 2, 3], "1.2.3", True),
        ("1.0.1", "1", False),
    ],
)
def test_operation_number_equality(left, right, are_equal) -> None:
    if are_equal:
        assert OperationNumber(left) == OperationNumber(right)
    else:
        assert OperationNumber(left) != OperationNumber(right)


@pytest.mark.parametrize(
    "shorter, longer",
    [
        ("1.10", "1.10.0"),
        ("1", "1.0.0"),
        ("2.0", "2.0.0"),
    ],
)
def test_operation_number_zero_pads(shorter, longer) -> None:
    assert OperationNumber(shorter) == OperationNumber(longer)
    assert OperationNumber(longer) == OperationNumber(shorter)
    assert OperationNumber(shorter) >= OperationNumber(longer)
    assert OperationNumber(longer) <= OperationNumber(shorter)


def test_operation_number_rejects_float() -> None:
    with pytest.raises(TypeError):
        OperationNumber(1.23)  # type: ignore[arg-type]


def test_operation_number_rejects_nonnumeric() -> None:
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
