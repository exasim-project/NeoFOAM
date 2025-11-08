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
import numpy as np
from pathlib import Path

from typing import Callable
from foamadapter.framework.step import Step
from dataclasses import dataclass, field

PLOT_DAG = True


    

@dataclass
class Step2:
    """A concrete step class that wraps a function with metadata."""

    func: Callable
    step_number: StepNumber
    step_name: str
    domain: str | None = None
    depends_on: list[str] | None = None
    shape: str = "box"
    color: str = "lightblue"
    level: int = 0
    sub_steps: list["Step2"] = field(default_factory=list)

    def node_data(self) -> NodeData:
        return NodeData(
            name=self.name,
            depends_on=self.dependency_names,
            shape=self.shape,
            step_number=self.step_number,
            color=self.color,
        )

    @property
    def name(self):
        return f"{self.domain}.{self.step_name}" if self.domain else self.step_name
    
    @property
    def dependency_names(self):
        if self.depends_on is None:
            return []
        if self.domain:
            return [f"{self.domain}.{dep}" for dep in self.depends_on]
        return self.depends_on

    def __call__(self, *args, **kwargs):
        if self.sub_steps:
            a = 0
            while self.func():
                for step in self.sub_steps:
                    a += step.func(*args, **kwargs)
            return a
        else:
            return self.func(*args, **kwargs)

    def run(self, ctx):
        if self.sub_steps:
            while self.func():
                for step in self.sub_steps:
                    step.run(ctx)
        else:
            return self.func.run(ctx)

@dataclass
class StepRunner:
    step: Step2
    def run(self, ctx):
        
        return self.step()

def function():
    return 1

class MaxIterations:
    def __init__(self):
        self.count = 0
        self.max_iterations = 3

    def __call__(self) -> bool:
        self.count += 1
        return self.count <= self.max_iterations

def test_nested_dag():
    import matplotlib
    colors = plt.cm.Greens(np.linspace(0.7, 0.1, 5))
    color_map = [matplotlib.colors.to_hex(c) for c in colors]

    step1 = Step2(
        func=function,
        step_number=StepNumber("1.0.0"),
        step_name="node1",
        depends_on=[],
        domain="domainA",
        color=color_map[0],
    )
    step2 = Step2(
        func=function,
        step_number=StepNumber("2.0.0"),
        step_name="node2",
        depends_on=["node1"],
        domain="domainA",
        color=color_map[0],
    )
    step3 = Step2(
        func=function,
        step_number=StepNumber("3.0.0"),
        step_name="node3",
        depends_on=["node1"],
        domain="domainA",
        color=color_map[0],
    )
    step4 = Step2(
        func=function,
        step_number=StepNumber("4.0.0"),
        step_name="node4",
        depends_on=["node2"],
        domain="domainA",
        color=color_map[0],
    )
    step5 = Step2(
        func=MaxIterations(),
        step_number=StepNumber("5.0.0"),
        step_name="node5",
        depends_on=["node3"],
        domain="domainA",
        color=color_map[0],
        sub_steps=
        [
            Step2(
                func=function,
                step_number=StepNumber("5.1.0"),
                step_name="node5_1",
                depends_on=["node3", "node5"],
                domain="domainA",
                color=color_map[1],
            ),
            Step2(
                func=function,
                step_number=StepNumber("5.2.0"),
                step_name="node5_2",
                depends_on=["node3", "node5", "node5_1"],
                domain="domainA",
                color=color_map[1],
            ),
        ],
    )
    steps = [step1, step2, step3, step4, step5]
    nodes = [step.node_data() for step in steps]
    order = compute_nodes_order(nodes)
    steps_ordered = compute_steps_order(steps, nodes)

    a = 0
    for step in steps_ordered:
        print(step.name)
        a += step()
    assert a == 4 + 3 * 2  # 4 from steps 1-4 and 3*2 from step5 substeps

    if PLOT_DAG:
        dag = build_dag(nodes)
        parent_dir = Path(__file__).parent
        path = str(parent_dir / "test_nested_dag.html")
        digraph_to_pyvis_html(dag, html_path=path)


