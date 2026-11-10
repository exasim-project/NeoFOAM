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
    Operations,
    StepBuilder,
)

from .sorter import NetworkxTopologicalSorter, TopologicalSorter
from .wrapping import check_passes, wrap


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
                    f"Operation '{name}' in scope '{scope}' depends on '{dep}' which does not exist"
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
    stepped: int | None = None,
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

    Only the first ``stepped`` entries of ``tagged`` (the builder's; default all)
    place a loop: it goes after the last op of its scope stepped before it, or
    ahead of the first one stepped after it if that sorts earlier. Model ops have
    no place of their own beside a loop.
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

    # Merge loops into each scope's sequential ops, which keep their topological order.
    n_stepped = len(tagged) if stepped is None else stepped
    by_scope: dict[str, list[Operation]] = {}
    for scope in set(seq_by_scope) | set(loops_by_scope):
        seq = seq_by_scope[scope]
        builder_idx = [order.get(id(op), len(order)) for op in seq]
        at: list[tuple[int, Operation]] = []
        for loop in sorted(loops_by_scope[scope], key=lambda op: order[id(op)]):
            loop_idx = order[id(loop)]
            after = [i for i, idx in enumerate(builder_idx) if idx < loop_idx]
            ahead = [i for i, idx in enumerate(builder_idx) if loop_idx < idx < n_stepped]
            at.append((min(after[-1] + 1 if after else 0, ahead[0] if ahead else len(seq)), loop))
        merged: list[Operation] = []
        next_loop = 0
        for i, seq_op in enumerate(seq):
            while next_loop < len(at) and at[next_loop][0] <= i:
                merged.append(at[next_loop][1])
                next_loop += 1
            merged.append(seq_op)
        merged.extend(loop for _, loop in at[next_loop:])
        by_scope[scope] = merged

    return by_scope


def _rebuild_builder(sorted_scopes: dict[str, list[Operation]]) -> StepBuilder:
    """A ``StepBuilder`` from ``root``, each loop filled with its sorted scope, recursively."""

    def rebuild(scope: str) -> list[Operation]:
        return [
            dataclasses.replace(op, sub_operations=rebuild(op.operation_name or "loop"))
            if isinstance(op.func, IterativeOp)
            else op
            for op in sorted_scopes.get(scope, [])
        ]

    return StepBuilder(rebuild("root"))


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
        """The operations of ``builder`` and ``additional_ops``, ordered in their scopes.

        Additional wrapping loops (``metadata.wraps``) are placed afterwards, in
        the order given (:func:`~.wrapping.wrap`, :func:`~.wrapping.check_passes`).
        """
        wrapping = [op for op in additional_ops if op.metadata.wraps is not None]
        rest = Operations([op for op in additional_ops if op.metadata.wraps is None])
        tagged = _collect_tagged_ops(builder, rest)
        graph, op_map = _build_global_graph(tagged)
        sorted_scopes = _sort_global(
            graph, op_map, tagged, sorter=self._sorter, stepped=len(tagged) - len(rest)
        )
        ops = _rebuild_builder(sorted_scopes).operations.ops
        for loop in wrapping:
            ops = wrap(ops, loop)
        check_passes(ops)
        return StepBuilder(ops)
