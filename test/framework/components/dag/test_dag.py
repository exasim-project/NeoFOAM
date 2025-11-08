import networkx as nx
from foamadapter.framework.dag import (
    build_dag,
    NodeData,
    build_global_dag,
    compute_nodes_order,
    compute_steps_order,
    StepNumber,
)
from foamadapter.framework.pyvis_utils import digraph_to_pyvis_html
import matplotlib.pyplot as plt
from pathlib import Path

PLOT_DAG = True


def test_build_dag():
    parent_dir = Path(__file__).parent
    node1 = NodeData(
        name="node1",
        depends_on=[],
        shape="circle",
        color="red",
        step_number=StepNumber("1.0.0"),
    )
    node2 = NodeData(
        name="node2",
        depends_on=["node1"],
        shape="square",
        color="blue",
        step_number=StepNumber("1.0.1"),
    )
    node3 = NodeData(
        name="node3",
        depends_on=["node1", "node2"],
        shape="triangle",
        color="green",
        step_number=StepNumber("1.1.0"),
    )
    node4 = NodeData(
        name="node4",
        depends_on=["node2"],
        shape="triangle",
        color="green",
        step_number=StepNumber("1.1.1"),
    )

    dag = build_dag([node1, node2, node3, node4])
    assert len(dag.nodes) == 4
    assert len(dag.edges) == 4
    assert ("node1", "node2") in dag.edges
    assert ("node1", "node3") in dag.edges
    assert ("node2", "node3") in dag.edges
    assert ("node2", "node4") in dag.edges

    if PLOT_DAG:
        path = str(parent_dir / "test_dag.html")
        digraph_to_pyvis_html(dag, html_path=path)


def test_build_global_dag():
    parent_dir = Path(__file__).parent
    domain_a_nodes = [
        NodeData(
            name="a1",
            depends_on=[],
            shape="circle",
            color="red",
            step_number=StepNumber("1.0.0"),
        ),
        NodeData(
            name="a2",
            depends_on=["a1"],
            shape="square",
            color="blue",
            step_number=StepNumber("1.0.0"),
        ),
    ]
    domain_b_nodes = [
        NodeData(
            name="b1",
            depends_on=[],
            shape="triangle",
            color="green",
            step_number=StepNumber("1.0.0"),
        ),
        NodeData(
            name="b2",
            depends_on=["b1", "a1"],
            shape="diamond",
            color="orange",
            step_number=StepNumber("1.0.0"),
        ),
    ]

    domains = {"domain_a": domain_a_nodes, "domain_b": domain_b_nodes}
    dag = build_global_dag(domains)
    assert len(dag.nodes) == 4
    assert len(dag.edges) == 3
    assert ("a1", "a2") in dag.edges
    assert ("b1", "b2") in dag.edges
    assert ("a1", "b2") in dag.edges

    if PLOT_DAG:
        path = str(parent_dir / "test_global_dag.html")
        digraph_to_pyvis_html(dag, html_path=path)


def test_compute_steps_order():
    node1 = NodeData(
        name="node1",
        depends_on=[],
        shape="circle",
        color="red",
        step_number=StepNumber("1.0.0"),
    )
    node2 = NodeData(
        name="node2",
        depends_on=["node1"],
        shape="square",
        color="blue",
        step_number=StepNumber("2.0.0"),
    )
    node3 = NodeData(
        name="node3",
        depends_on=["node1", "node2"],
        shape="triangle",
        color="green",
        step_number=StepNumber("3.2.0"),
    )
    node4 = NodeData(
        name="node4",
        depends_on=["node2"],
        shape="triangle",
        color="green",
        step_number=StepNumber("3.1.0"),
    )
    node5 = NodeData(
        name="node5",
        depends_on=["node3"],
        shape="circle",
        color="red",
        step_number=StepNumber("3.2.0"),
    )
    node6 = NodeData(
        name="node6",
        depends_on=["node3"],
        shape="circle",
        color="red",
        step_number=StepNumber("3.2.0"),
    )
    node7 = NodeData(
        name="node7",
        depends_on=["node3"],
        shape="circle",
        color="red",
        step_number=StepNumber("3.2.1"),
    )

    nodes = [node1, node2, node3, node4, node5, node6, node7]
    order = compute_nodes_order(nodes)
    assert order == ["node1", "node2", "node4", "node3", "node5", "node6", "node7"]

    if PLOT_DAG:
        dag = build_dag(nodes)
        parent_dir = Path(__file__).parent
        path = str(parent_dir / "test_dag_order.html")
        digraph_to_pyvis_html(dag, html_path=path)

