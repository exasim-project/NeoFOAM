from pathlib import Path

from neofoam.framework.graph import (
    build_dag,
    build_global_dag,
)
from neofoam.framework.graph.visualization import digraph_to_pyvis_html
from neofoam.framework.types import OperationMetadata, OperationNumber

PLOT_DAG = True


def test_build_dag():
    parent_dir = Path(__file__).parent
    node1 = OperationMetadata(
        op_name="node1",
        depends_on=[],
        shape="circle",
        color="red",
        operation_number=OperationNumber("1.0.0"),
    )
    node2 = OperationMetadata(
        op_name="node2",
        depends_on=["node1"],
        shape="square",
        color="blue",
        operation_number=OperationNumber("1.0.1"),
    )
    node3 = OperationMetadata(
        op_name="node3",
        depends_on=["node1", "node2"],
        shape="triangle",
        color="green",
        operation_number=OperationNumber("1.1.0"),
    )
    node4 = OperationMetadata(
        op_name="node4",
        depends_on=["node2"],
        shape="triangle",
        color="green",
        operation_number=OperationNumber("1.1.1"),
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
        OperationMetadata(
            op_name="a1",
            depends_on=[],
            shape="circle",
            color="red",
            operation_number=OperationNumber("1.0.0"),
        ),
        OperationMetadata(
            op_name="a2",
            depends_on=["a1"],
            shape="square",
            color="blue",
            operation_number=OperationNumber("1.0.0"),
        ),
    ]
    domain_b_nodes = [
        OperationMetadata(
            op_name="b1",
            depends_on=[],
            shape="triangle",
            color="green",
            operation_number=OperationNumber("1.0.0"),
        ),
        OperationMetadata(
            op_name="b2",
            depends_on=["b1", "a1"],
            shape="diamond",
            color="orange",
            operation_number=OperationNumber("1.0.0"),
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
