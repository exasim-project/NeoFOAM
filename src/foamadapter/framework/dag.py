from functools import total_ordering
from dataclasses import dataclass
import networkx as nx


@total_ordering
class StepNumber:
    def __init__(self, version):
        if isinstance(version, str):
            self.parts = [int(p) for p in version.split('.')]
        elif isinstance(version, (list, tuple)):
            self.parts = list(map(int, version))
        elif isinstance(version, int):
            self.parts = [version]
        else:
            raise TypeError("StepNumber must be initialized with a string, int, or list/tuple of integers")

    def _as_tuple(self, other):
        if not isinstance(other, StepNumber):
            other = StepNumber(other)
        max_len = max(len(self.parts), len(other.parts))
        a = tuple(self.parts + [0] * (max_len - len(self.parts)))
        b = tuple(other.parts + [0] * (max_len - len(other.parts)))
        return a, b

    def __eq__(self, other):
        a, b = self._as_tuple(other)
        return a == b

    def __lt__(self, other):
        a, b = self._as_tuple(other)
        return a < b

@dataclass
class NodeData:
    name: str
    depends_on: list[str]
    shape: str
    step_number: StepNumber
    color: str = None

    @property
    def dependencies(self):
        # Return the list of dependencies for this node
        # check if depends_on has node names or NodeData objects
        dependencies = self.depends_on
        return dependencies
    

def build_dag(nodes: list[NodeData]) -> nx.DiGraph:
    """
    Build a DAG from a list of NodeData objects.
    """
    G = nx.DiGraph()
    for node in nodes:
        G.add_node(node.name, meta=node, shape=node.shape, color=node.color, step_number=node.step_number)
        for dep in node.depends_on:
            G.add_edge(dep, node.name)
    return G

def build_global_dag(domains: dict[str, list[NodeData]]) -> nx.DiGraph:
    """
    Build a global DAG from multiple domain models, supporting interdomain dependencies.
    Each node is named as 'domain.step'.
    """
    G = nx.DiGraph()
    for domain_name, nodes in domains.items():
        sub_graph = build_dag(nodes)
        G = nx.compose(G, sub_graph)
    return G

def compute_steps_order(nodes: list[NodeData]) -> list[str]:
    """
    Compute a valid topological order of steps in the DAG.
    """
    dag = build_dag(nodes)
    return list(nx.lexicographical_topological_sort(dag, key=lambda n: dag.nodes[n]["step_number"]))