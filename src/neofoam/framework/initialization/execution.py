# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""Free functions for initialization execution."""

import logging
from typing import Any
import networkx as nx

from .init_step import InitStep
from ..context import Context

logger = logging.getLogger(__name__)


def topological_sort(lazy_inits: list[InitStep]) -> list[InitStep]:
    """Sort lazy initializers by dependencies using DAG."""
    seen_names: set[str] = set()
    for li in lazy_inits:
        if li.name in seen_names:
            raise ValueError(f"Duplicate InitStep name detected: '{li.name}'")
        seen_names.add(li.name)

    name_to_init = {li.name: li for li in lazy_inits}

    # Validate dependencies exist
    for li in lazy_inits:
        for dep in li.depends_on:
            if dep not in name_to_init:
                raise ValueError(
                    f"InitStep '{li.name}' depends on '{dep}', "
                    f"but '{dep}' was not found"
                )

    # Build DAG
    graph = nx.DiGraph()
    for li in lazy_inits:
        graph.add_node(li.name)
        for dep in li.depends_on:
            graph.add_edge(dep, li.name)

    # Topological sort
    try:
        sorted_names = list(nx.lexicographical_topological_sort(graph))
    except (nx.NetworkXError, nx.NetworkXUnfeasible) as e:
        cycle = nx.find_cycle(graph)
        cycle_names = [edge[0] for edge in cycle]
        raise ValueError(f"Circular dependency: {' -> '.join(cycle_names)}") from e

    return [name_to_init[name] for name in sorted_names]


def execute_lazy_inits(lazy_inits: list[InitStep]) -> dict[str, Any]:
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
            logger.warning(
                "InitStep '%s' has no recognised prefix — routing to models", name
            )
            models[name] = obj

    return Context(fields=fields, models=models, mesh=mesh, runTime=runtime)


def execute_initialization(lazy_inits: list[InitStep]) -> Context:
    """Execute lazy initializers and build Context."""
    objects = execute_lazy_inits(lazy_inits)
    return build_context_from_objects(objects)


def validate_lazy_init_graph(lazy_inits: list[InitStep]) -> list[tuple[str, str]]:
    """Validate lazy init graph for missing deps and cycles."""
    errors: list[tuple[str, str]] = []
    names = {li.name for li in lazy_inits}

    # Check for missing dependencies
    for li in lazy_inits:
        for dep in li.depends_on:
            if dep not in names:
                errors.append((li.name, f"Missing dependency: {dep}"))

    if errors:
        return errors

    # Check for cycles (only if all deps exist, otherwise topological_sort
    # would raise for missing deps again producing duplicates)
    try:
        topological_sort(lazy_inits)
    except ValueError as e:
        errors.append(("graph", str(e)))

    return errors
