from dataclasses import dataclass
import networkx as nx

@dataclass
class NodeData:
    name: str
    depends_on: list[str]
    shape: str

    @property
    def dependencies(self):
        # Return the list of dependencies for this node
        # check if depends_on has node names or NodeData objects
        dependencies = self.depends_on
        return dependencies

def build_global_dag(domains: dict[str, list[NodeData]]) -> nx.DiGraph:
    """
    Build a global DAG from multiple domain models, supporting interdomain dependencies.
    Each node is named as 'domain.step'.
    """
    G = nx.DiGraph()
    for domain_name, nodes in domains.items():
        for node in nodes:
            G.add_node(node.name, meta=node, shape=node.shape)
            for dep in node.depends_on:
                G.add_edge(dep, node.name)
    return G


def build_step_dag(model):
    steps = getattr(model.__class__, "_steps", [])
    G = nx.DiGraph()
    # Add nodes
    for step in steps:
        G.add_node(step.name, meta=step)
    # Add edges by depends_on
    for step in steps:
        for dep in step.depends_on:
            G.add_edge(dep, step.name)
    # If no depends_on, use order for linear chain
    if not any(s.depends_on for s in steps):
        steps_sorted = sorted(steps, key=lambda s: s.order)
        for i in range(1, len(steps_sorted)):
            G.add_edge(steps_sorted[i-1].name, steps_sorted[i].name)
    return G