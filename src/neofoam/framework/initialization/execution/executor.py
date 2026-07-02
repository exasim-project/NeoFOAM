# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Execute init steps in dependency order."""

import os
from typing import Any

import networkx as nx  # type: ignore[import-untyped]

from ...graph import write_digraph
from ..init_step import InitStep, InitStepExecutionError
from .init_result import InitResult
from .ordering import _topological_sort

#: When set to a path, :func:`execute_lazy_inits` dumps the resolved init DAG
#: there before running any step (so a native abort mid-init still leaves it).
DUMP_INIT_DAG_ENV = "NEOFOAM_DUMP_INIT_DAG"


def _init_digraph(steps: list[InitStep]) -> nx.DiGraph:
    """Build a DiGraph from ``steps`` for rendering (edges: dependency → step).

    Each step becomes a node carrying its ``category``, ``write`` flag, and
    declared ``depends_on`` (so the text renderer lists dependencies in
    declaration order). Dependencies that are not themselves declared steps —
    external context keys — appear as attribute-less nodes via their edges.
    """
    graph = nx.DiGraph()
    for step in steps:
        graph.add_node(
            step.name,
            category=step.category,
            write=step.write,
            depends_on=list(step.depends_on),
        )
        for dep in step.depends_on:
            graph.add_edge(dep, step.name)
    return graph


def execute_step(step: InitStep, context: dict[str, Any]) -> Any:
    """Run one :class:`InitStep` against ``context`` and return its value.

    Wraps unexpected exceptions in :class:`InitStepExecutionError` to attach
    step name + declared dependencies. ``ValueError`` / ``TypeError`` raised
    by the initializer pass through unchanged.
    """
    if step.initializer is None:
        raise ValueError(f"InitStep '{step.name}' has no initializer function")

    try:
        return step.initializer(context)
    except (ValueError, TypeError):
        raise
    except Exception as exc:
        raise InitStepExecutionError(step.name, step.depends_on, exc) from exc


def execute_lazy_inits(
    lazy_inits: list[InitStep], *, assume_validated: bool = False
) -> list[InitResult]:
    """Execute ``lazy_inits`` in dependency order and collect their results."""
    sorted_inits = _topological_sort(lazy_inits, validate_graph=not assume_validated)

    dump_path = os.environ.get(DUMP_INIT_DAG_ENV)
    if dump_path:
        # ``sorted_inits`` is already the execution order — dump it verbatim,
        # before the first step runs, so a native abort still leaves the file.
        write_digraph(
            _init_digraph(sorted_inits),
            dump_path,
            order=[step.name for step in sorted_inits],
            title=f"NeoFOAM initialization DAG — {len(sorted_inits)} steps "
            "(execution order)",
        )

    context: dict[str, Any] = {}
    results: list[InitResult] = []

    for lazy_init in sorted_inits:
        obj = execute_step(lazy_init, context)
        context[lazy_init.name] = obj
        results.append(
            InitResult(
                name=lazy_init.name,
                category=lazy_init.category,
                value=obj,
                write=lazy_init.write,
            )
        )

    return results
