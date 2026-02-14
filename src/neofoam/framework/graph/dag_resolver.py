# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""DAG resolver for StepBuilder hierarchies and model operation insertion."""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

from neofoam.framework.operations import (
    IterativeOp,
    Operation,
    OperationCollection,
    StepBuilder,
)
from neofoam.framework.types import OperationNumber


class CyclicDependencyError(Exception):
    """Raised when a cyclic dependency is detected in the operation graph."""


class MissingDependencyError(Exception):
    """Raised when an operation depends on a non-existent operation."""


class DAGResolver:
    """Merge and order operations by dependency across loop scopes."""

    def resolve(
        self,
        builder: StepBuilder,
        additional_ops: OperationCollection,
    ) -> StepBuilder:
        scopes = self._extract_scopes(builder)

        for op in additional_ops:
            self._insert_operation(op, scopes)

        all_ops: dict[str, str] = {}
        for scope_name, ops in scopes.items():
            for op in ops:
                if op.operation_name:
                    all_ops[op.operation_name] = scope_name

        for scope_name in scopes:
            scopes[scope_name] = self._topological_sort(
                scopes[scope_name], scope_name, all_ops
            )

        return self._rebuild_builder(builder, scopes)

    def _extract_scopes(self, builder: StepBuilder) -> dict[str, list[Operation]]:
        scopes: dict[str, list[Operation]] = defaultdict(list)

        def extract_recursive(ops: list[Operation], parent_scope: str) -> None:
            for op in ops:
                scopes[parent_scope].append(op)
                if isinstance(op.func, IterativeOp):
                    loop_name = op.operation_name or "loop"
                    extract_recursive(op.sub_operations, loop_name)

        extract_recursive(builder.operations.ops, "root")
        return scopes

    def _insert_operation(
        self,
        op: Operation,
        scopes: dict[str, list[Operation]],
    ) -> None:
        target_scope = None

        if op.depends_on or op.before:
            op_to_scope: dict[str, str] = {}
            for scope_name, scope_ops in scopes.items():
                for scope_op in scope_ops:
                    if scope_op.operation_name:
                        op_to_scope[scope_op.operation_name] = scope_name

            dep_scopes = set()
            for dep in op.depends_on or []:
                if dep in op_to_scope:
                    dep_scopes.add(op_to_scope[dep])
            for constraint in op.before or []:
                if constraint in op_to_scope:
                    dep_scopes.add(op_to_scope[constraint])

            if dep_scopes:
                if "inner_loop" in dep_scopes:
                    target_scope = "inner_loop"
                elif len(dep_scopes) == 1:
                    target_scope = dep_scopes.pop()
                else:
                    non_root_scopes = [s for s in dep_scopes if s != "root"]
                    target_scope = (
                        sorted(non_root_scopes)[0] if non_root_scopes else "root"
                    )

        if target_scope is None:
            available_loops = [s for s in scopes.keys() if s != "root"]
            if available_loops:
                target_scope = (
                    "inner_loop"
                    if "inner_loop" in available_loops
                    else available_loops[0]
                )
            else:
                target_scope = "root"

        if target_scope not in scopes:
            target_scope = "root"

        scopes[target_scope].append(op)

    def _topological_sort(
        self,
        operations: list[Operation],
        scope_name: str,
        all_ops: Optional[dict[str, str]] = None,
    ) -> list[Operation]:
        if not operations:
            return []

        loop_ops = [op for op in operations if isinstance(op.func, IterativeOp)]
        regular_ops = [op for op in operations if not isinstance(op.func, IterativeOp)]

        if not regular_ops:
            return loop_ops

        op_map = {op.operation_name: op for op in regular_ops if op.operation_name}

        graph: dict[str, list[str]] = {}
        in_degree: dict[str, int] = {}
        for op in regular_ops:
            if not op.operation_name:
                continue
            graph[op.operation_name] = []
            in_degree[op.operation_name] = 0

        for op in regular_ops:
            if not op.operation_name:
                continue

            op_name = op.operation_name
            deps = op.depends_on or []
            before = op.before or []

            for dep in deps:
                if all_ops and dep not in all_ops:
                    raise MissingDependencyError(
                        f"Operation '{op_name}' in scope '{scope_name}' depends on '{dep}' which does not exist"
                    )
                if dep in op_map:
                    graph[dep].append(op_name)
                    in_degree[op_name] += 1

            for before_op in before:
                if all_ops and before_op not in all_ops:
                    raise MissingDependencyError(
                        f"Operation '{op_name}' in scope '{scope_name}' has before target '{before_op}' which does not exist"
                    )
                if before_op in op_map:
                    graph[op_name].append(before_op)
                    in_degree[before_op] += 1

        import heapq

        queue: list[tuple[tuple[int, OperationNumber], str]] = []
        for name, degree in in_degree.items():
            if degree == 0:
                op = op_map[name]
                if op.operation_number is not None:
                    priority = (0, op.operation_number)
                else:
                    priority = (1, OperationNumber([0]))
                heapq.heappush(queue, (priority, name))

        sorted_names: list[str] = []
        while queue:
            _, current = heapq.heappop(queue)
            sorted_names.append(current)

            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    op = op_map[neighbor]
                    if op.operation_number is not None:
                        priority = (0, op.operation_number)
                    else:
                        priority = (1, OperationNumber([0]))
                    heapq.heappush(queue, (priority, neighbor))

        if len(sorted_names) != len(op_map):
            raise CyclicDependencyError(
                f"Cyclic dependency detected in scope '{scope_name}'"
            )

        sorted_regular = [op_map[name] for name in sorted_names]
        result: list[Operation] = []
        regular_idx = 0

        for orig_op in operations:
            if isinstance(orig_op.func, IterativeOp):
                result.append(orig_op)
            elif regular_idx < len(sorted_regular):
                result.append(sorted_regular[regular_idx])
                regular_idx += 1

        while regular_idx < len(sorted_regular):
            result.append(sorted_regular[regular_idx])
            regular_idx += 1

        return result

    def _rebuild_builder(
        self,
        original_builder: StepBuilder,
        sorted_scopes: dict[str, list[Operation]],
    ) -> StepBuilder:
        new_builder = StepBuilder()

        def rebuild_recursive(
            original_ops: list[Operation],
            target_builder: StepBuilder,
            parent_scope: str,
        ) -> None:
            _ = parent_scope
            for op in original_ops:
                if isinstance(op.func, IterativeOp):
                    loop_name = op.operation_name or "loop"
                    sorted_ops = sorted_scopes.get(loop_name, [])

                    new_op = Operation(
                        func=op.func,
                        operation_name=op.operation_name,
                        operation_number=op.operation_number,
                        domain_name=op.domain_name,
                        depends_on=op.depends_on,
                        before=op.before,
                        shape=op.shape,
                        color=op.color,
                        level=op.level,
                        sub_operations=[],
                    )

                    with target_builder.loop(new_op) as loop_builder:
                        for sub_op in sorted_ops:
                            if isinstance(sub_op.func, IterativeOp):
                                rebuild_recursive([sub_op], loop_builder, loop_name)
                            else:
                                loop_builder.step(sub_op)

        has_loops = any(
            isinstance(op.func, IterativeOp) for op in original_builder.operations.ops
        )

        if has_loops:
            rebuild_recursive(original_builder.operations.ops, new_builder, "root")
        else:
            if "root" in sorted_scopes:
                for op in sorted_scopes["root"]:
                    new_builder.step(op)

        return new_builder
