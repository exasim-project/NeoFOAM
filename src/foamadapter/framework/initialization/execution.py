"""Free functions for initialization execution."""

from typing import Any
import networkx as nx

from .lazy_init import LazyInit
from ..context import Context


def topological_sort(lazy_inits: list[LazyInit]) -> list[LazyInit]:
    """Sort lazy initializers by dependencies using DAG."""
    name_to_init = {li.name: li for li in lazy_inits}

    # Validate dependencies exist
    for li in lazy_inits:
        for dep in li.depends_on:
            if dep not in name_to_init:
                raise ValueError(
                    f"LazyInit '{li.name}' depends on '{dep}', "
                    f"but '{dep}' was not found"
                )

    # Build DAG
    G = nx.DiGraph()
    for li in lazy_inits:
        G.add_node(li.name)
        for dep in li.depends_on:
            G.add_edge(dep, li.name)

    # Topological sort
    try:
        sorted_names = list(nx.lexicographical_topological_sort(G))
    except (nx.NetworkXError, nx.NetworkXUnfeasible) as e:
        cycle = nx.find_cycle(G)
        cycle_names = [edge[0] for edge in cycle]
        raise ValueError(f"Circular dependency: {' -> '.join(cycle_names)}") from e

    return [name_to_init[name] for name in sorted_names]


def execute_lazy_inits(lazy_inits: list[LazyInit]) -> dict[str, Any]:
    """Execute lazy initializers in dependency order."""
    sorted_inits = topological_sort(lazy_inits)
    objects: dict[str, Any] = {}

    for lazy_init in sorted_inits:
        obj = lazy_init.execute(context=objects)
        objects[lazy_init.name] = obj

    return objects


def build_context_from_objects(objects: dict[str, Any]) -> Context:
    """Build Context from initialized objects."""
    fields = {}
    models = {}
    mesh = None
    runtime = None

    for name, obj in objects.items():
        if name.startswith("fields."):
            field_name = name.replace("fields.", "")
            fields[field_name] = obj
        elif name.startswith("operators."):
            operator_name = name.replace("operators.", "")
            models[operator_name] = obj
        elif name.startswith("models."):
            model_name = name.replace("models.", "")
            models[model_name] = obj
        elif name == "mesh":
            mesh = obj
        elif name == "runtime":
            runtime = obj
        else:
            models[name] = obj

    return Context(fields=fields, models=models, mesh=mesh, runTime=runtime)


def execute_initialization(lazy_inits: list[LazyInit]) -> Context:
    """Execute lazy initializers and build Context."""
    objects = execute_lazy_inits(lazy_inits)
    return build_context_from_objects(objects)


def validate_lazy_init_graph(lazy_inits: list[LazyInit]) -> list[tuple[str, str]]:
    """Validate lazy init graph for missing deps and cycles."""
    errors = []
    names = {li.name for li in lazy_inits}

    # Check for missing dependencies
    for li in lazy_inits:
        for dep in li.depends_on:
            if dep not in names:
                errors.append((li.name, f"Missing dependency: {dep}"))

    # Check for cycles
    try:
        topological_sort(lazy_inits)
    except ValueError as e:
        errors.append(("graph", str(e)))

    return errors
