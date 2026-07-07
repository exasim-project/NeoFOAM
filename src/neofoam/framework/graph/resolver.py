# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""DAG resolver for StepBuilder hierarchies and model operation insertion."""

from __future__ import annotations

import dataclasses
from collections import defaultdict
from typing import Any

import networkx as nx  # type: ignore[import-untyped]

from neofoam.framework.operations import (
    IterativeOp,
    Operation,
    OperationCollection,
    StepBuilder,
)

from .sorter import NetworkxTopologicalSorter, TopologicalSorter


class CyclicDependencyError(Exception):
    """Raised when a cyclic dependency is detected in the operation graph."""


class MissingDependencyError(Exception):
    """Raised when an operation depends on a non-existent operation."""


def _walk(
    ops: list[Operation],
    parent_scope: str,
    depth: int,
    tagged: list[tuple[str, Operation]],
    scope_depth: dict[str, int],
) -> None:
    """Recursively tag operations with their scope and record loop-scope depths."""
    for op in ops:
        tagged.append((parent_scope, op))
        if isinstance(op.func, IterativeOp):
            loop_scope = op.operation_name or "loop"
            scope_depth[loop_scope] = depth + 1
            _walk(op.sub_operations, loop_scope, depth + 1, tagged, scope_depth)


def _collect_tagged_ops(
    builder: StepBuilder,
    model_ops: OperationCollection,
) -> list[tuple[str, Operation]]:
    """Flatten the builder tree and model ops into ``[(scope_name, Operation)]``.

    Loop (``IterativeOp``) operations are tagged with their *parent* scope so
    that the rebuild step can find them.  Sequential operations inside a loop
    are tagged with the loop's own scope name.

    Model operations are placed into the scope that best matches their
    dependency targets via :func:`infer_target_scope`, using actual nesting
    depth to pick the innermost scope.
    """
    tagged: list[tuple[str, Operation]] = []
    scope_depth: dict[str, int] = {"root": 0}

    _walk(builder.operations.ops, "root", 0, tagged, scope_depth)

    # Build op-name → scope lookup for scope inference
    op_to_scope: dict[str, str] = {}
    for scope, op in tagged:
        if op.operation_name:
            op_to_scope[op.operation_name] = scope

    # All scope names that exist (needed for fallback)
    all_scopes: set[str] = {scope for scope, _ in tagged}

    for op in model_ops:
        target = _infer_target_scope(op, op_to_scope, all_scopes, scope_depth)
        tagged.append((target, op))
        # Update lookup so chained model ops can find each other
        if op.operation_name:
            op_to_scope[op.operation_name] = target

    return tagged


def _deepest_scope(
    scopes: list[str],
    scope_depth: dict[str, int] | None,
) -> str:
    """Pick the deepest scope; break ties alphabetically."""
    if scope_depth:
        return max(scopes, key=lambda s: (scope_depth.get(s, 0), s))
    return sorted(scopes)[0]


def _infer_target_scope(
    op: Operation,
    op_to_scope: dict[str, str],
    all_scopes: set[str],
    scope_depth: dict[str, int] | None = None,
) -> str:
    """Determine which scope a model operation belongs to.

    Strategy:
      1. Collect scopes of all ``depends_on`` / ``before`` targets.
      2. Prefer the deepest (innermost) non-root scope among them.
      3. If no deps, fall back to the deepest available loop scope.

    When *scope_depth* is provided (as computed by
    :func:`collect_tagged_ops`), nesting depth drives the selection.
    Without it, alphabetical ordering is used as a fallback.
    """
    dep_scopes: set[str] = set()
    for dep in op.depends_on or []:
        if dep in op_to_scope:
            dep_scopes.add(op_to_scope[dep])
    for constraint in op.before or []:
        if constraint in op_to_scope:
            dep_scopes.add(op_to_scope[constraint])

    if dep_scopes:
        non_root = [s for s in dep_scopes if s != "root"]
        return _deepest_scope(non_root, scope_depth) if non_root else "root"

    # Default: deepest available loop scope
    loop_scopes = [s for s in all_scopes if s != "root"]
    return _deepest_scope(loop_scopes, scope_depth) if loop_scopes else "root"


def _build_global_graph(
    tagged: list[tuple[str, Operation]],
) -> tuple[nx.DiGraph, dict[str, tuple[str, Operation]]]:
    """Build a single ``nx.DiGraph`` from all tagged operations.

    Only non-loop (sequential/conditional) operations with names become graph
    nodes.  Loop operations are structural containers handled by
    :func:`rebuild_builder`.

    Returns the graph and an ``op_map`` mapping node names to
    ``(scope, Operation)`` pairs.
    """
    graph = nx.DiGraph()
    op_map: dict[str, tuple[str, Operation]] = {}

    for scope, op in tagged:
        if op.operation_name and not isinstance(op.func, IterativeOp):
            graph.add_node(op.operation_name, scope=scope)
            op_map[op.operation_name] = (scope, op)

    all_op_names = set(op_map.keys())

    for name, (scope, op) in op_map.items():
        for dep in op.depends_on or []:
            if dep not in all_op_names:
                raise MissingDependencyError(
                    f"Operation '{name}' in scope '{scope}' "
                    f"depends on '{dep}' which does not exist"
                )
            graph.add_edge(dep, name)

        for before_target in op.before or []:
            if before_target not in all_op_names:
                raise MissingDependencyError(
                    f"Operation '{name}' in scope '{scope}' "
                    f"has before target '{before_target}' which does not exist"
                )
            graph.add_edge(name, before_target)

    return graph, op_map


def _sort_global(
    graph: nx.DiGraph,
    op_map: dict[str, tuple[str, Operation]],
    tagged: list[tuple[str, Operation]],
    sorter: TopologicalSorter,
) -> dict[str, list[Operation]]:
    """Topologically sort the global graph, grouping results back by scope.

    Ordering precedence among the sequential ops of a scope:

    1. ``depends_on`` / ``before`` edges (hard topological constraint);
    2. ``OperationNumber`` (numbered ops first, then by value);
    3. **builder insertion order** — the op's position in the pre-order
       ``tagged`` list — so explicitly stepped ops that carry no number keep the
       order they were stepped in. (Model ops, appended after the builder walk,
       sort last.)

    Loop (``IterativeOp``) operations are not graph nodes — they are structural
    containers. They are merged back into their parent scope *by builder order*
    (not forced to the front), so a loop sits exactly where it was stepped
    relative to the surrounding ops; ops may therefore precede **and** follow a
    loop in the same scope. Loops carry no edges, so interleaving them by builder
    index is always dependency-safe.
    """
    # Builder insertion order: position in the pre-order ``tagged`` list.
    order: dict[int, int] = {id(op): i for i, (_, op) in enumerate(tagged)}

    def _sort_key(node_name: str) -> tuple[Any, ...]:
        _, op = op_map[node_name]
        return (
            op.operation_number is None,  # numbered ops first
            op.operation_number,  # then by number value
            order.get(id(op), len(order)),  # finally builder order
        )

    try:
        sorted_names = sorter.sort(graph, key=_sort_key)
    except nx.NetworkXUnfeasible:
        raise CyclicDependencyError("Cyclic dependency detected in operation graph")

    # Sequential ops per scope, in topological order.
    seq_by_scope: dict[str, list[Operation]] = defaultdict(list)
    for name in sorted_names:
        scope, op = op_map[name]
        seq_by_scope[scope].append(op)

    # Loop ops per scope (structural containers, not graph nodes).
    loops_by_scope: dict[str, list[Operation]] = defaultdict(list)
    for scope, op in tagged:
        if isinstance(op.func, IterativeOp):
            loops_by_scope[scope].append(op)

    # Merge loops into each scope's sequential ops by builder order. The
    # sequential ops keep their topological order; each loop is inserted ahead of
    # the first sequential op stepped after it.
    by_scope: dict[str, list[Operation]] = {}
    for scope in set(seq_by_scope) | set(loops_by_scope):
        seq = seq_by_scope[scope]
        loops = sorted(loops_by_scope[scope], key=lambda op: order[id(op)])
        merged: list[Operation] = []
        next_loop = 0
        for seq_op in seq:
            seq_idx = order.get(id(seq_op), len(order))
            while next_loop < len(loops) and order[id(loops[next_loop])] < seq_idx:
                merged.append(loops[next_loop])
                next_loop += 1
            merged.append(seq_op)
        merged.extend(loops[next_loop:])
        by_scope[scope] = merged

    return by_scope


def _rebuild_builder(
    original: StepBuilder,
    sorted_scopes: dict[str, list[Operation]],
) -> StepBuilder:
    """Reconstruct a ``StepBuilder`` from sorted scope data.

    Walks the original builder tree to preserve nesting structure but
    replaces each scope's operations with the sorted versions.
    """
    new_builder = StepBuilder()

    def _rebuild(
        original_ops: list[Operation],
        target_builder: StepBuilder,
    ) -> None:
        for op in original_ops:
            if isinstance(op.func, IterativeOp):
                loop_name = op.operation_name or "loop"
                sorted_ops = sorted_scopes.get(loop_name, [])

                new_op = dataclasses.replace(op, sub_operations=[])

                with target_builder.loop(new_op) as loop_builder:
                    for sub_op in sorted_ops:
                        if isinstance(sub_op.func, IterativeOp):
                            _rebuild([sub_op], loop_builder)
                        else:
                            loop_builder.step(sub_op)

    has_loops = any(isinstance(op.func, IterativeOp) for op in original.operations.ops)

    if has_loops:
        _rebuild(original.operations.ops, new_builder)
    elif "root" in sorted_scopes:
        for op in sorted_scopes["root"]:
            new_builder.step(op)

    return new_builder


class DAGResolver:
    """Merge and order operations by dependency across loop scopes.

    Thin orchestrator that delegates to pure functions::

        collect_tagged_ops → build_global_graph → sort_global → rebuild_builder

    The topological sorter is injected (DIP seam); default is
    :class:`NetworkxTopologicalSorter`.
    """

    def __init__(self, sorter: TopologicalSorter | None = None) -> None:
        self._sorter: TopologicalSorter = sorter or NetworkxTopologicalSorter()

    def resolve(
        self,
        builder: StepBuilder,
        additional_ops: OperationCollection,
    ) -> StepBuilder:
        tagged = _collect_tagged_ops(builder, additional_ops)
        graph, op_map = _build_global_graph(tagged)
        sorted_scopes = _sort_global(graph, op_map, tagged, sorter=self._sorter)
        return _rebuild_builder(builder, sorted_scopes)
