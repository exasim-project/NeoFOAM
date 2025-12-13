# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors
from foamadapter.framework.types import OperationMetadata, OperationNumber


def test_node_data():
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
