from foamadapter.framework.dag import NodeData, StepNumber


def test_node_data():
    node = NodeData(name="test_node", depends_on=["dep1", "dep2"], shape="circle", color="red", step_number=StepNumber("1.0.0"))
    assert node.name == "test_node"
    assert node.depends_on == ["dep1", "dep2"]
    assert node.shape == "circle"
    assert node.color == "red"
    assert node.dependencies == ["dep1", "dep2"]
    assert node.step_number == StepNumber("1.0.0")

    node_no_color = NodeData(name="test_node2", depends_on=[], shape="square", step_number=StepNumber("1.0.1"))
    assert node_no_color.name == "test_node2"
    assert node_no_color.depends_on == []
    assert node_no_color.shape == "square"
    assert node_no_color.color is None
    assert node_no_color.dependencies == []
    assert node_no_color.step_number == StepNumber("1.0.1")