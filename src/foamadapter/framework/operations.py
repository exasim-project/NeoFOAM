# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors
from __future__ import annotations

import inspect
from collections import defaultdict
from dataclasses import dataclass, field, is_dataclass
from typing import Annotated, Any, Callable, Iterator, Union, get_args, get_origin

from foamadapter.framework.context import Context, FieldUpdates

from .types import OperationMetadata, OpType, OperationNumber
from pybFoam import Info


class CyclicDependencyError(Exception):
    """Raised when a cyclic dependency is detected in the operation graph."""

    pass


class MissingDependencyError(Exception):
    """Raised when an operation depends on a non-existent operation."""

    pass


def _get_value(ctx: Context, name: str, annotation: Any) -> dict[str, Any]:
    call_args = {}
    ctx_var = ctx.fields

    is_annotated = get_origin(annotation) is Annotated
    var_name = "fields"
    if is_annotated:
        var_name = get_args(annotation)[1]
        ctx_var = getattr(ctx, var_name)

    if name in ctx_var:
        call_args[name] = ctx_var[name]
    else:
        raise KeyError(
            f"Required parameter '{name}' was not found in {var_name} and has no default value."
        )

    return call_args


def get_function_parameters(func: Callable[..., Any]) -> dict[str, type]:
    sig = inspect.signature(func)
    param_types = {}
    for name, param in sig.parameters.items():
        param_types[name] = param.annotation
    return param_types


def get_call_arguments(func_args: dict[str, type], context: Context) -> dict[str, Any]:
    call_args = {}
    for name, annotation in func_args.items():
        if is_dataclass(annotation):
            dc_sig = inspect.signature(annotation)
            dc_args = {}
            for dc_name, dc_param in dc_sig.parameters.items():
                dc_annotation = dc_sig.parameters[dc_name].annotation
                dc_args.update(_get_value(context, dc_name, dc_annotation))
            call_args[name] = annotation(**dc_args)
        elif annotation == Context:
            call_args[name] = context
        elif get_origin(annotation) is Annotated:
            # Handle Annotated types (e.g., Model[SomeType] to get from ctx.models)
            call_args.update(_get_value(context, name, annotation))
        else:
            call_args[name] = context.fields[name]
    return call_args


def _get_operation_type(func: Callable[..., Any]) -> OpType:
    if hasattr(func, "_metadata"):
        op_type = func._metadata.op_type
        if not isinstance(op_type, OpType):
            raise ValueError("Function metadata does not have valid op_type")
        return op_type
    raise ValueError("Function is not decorated with metadata")


def context_adapter(func: Callable[..., Any]) -> Callable[[Context], Any]:
    func_paras = get_function_parameters(func)
    op_type = _get_operation_type(func)
    # TODO error handling if not FieldUpdates is returned

    def wrapped_function(context: Context) -> Any:
        call_args = get_call_arguments(func_paras, context)
        results = func(**call_args)

        if op_type == OpType.OPERATION and isinstance(results, FieldUpdates):
            context.fields.update(results)
            return None

        return results

    return wrapped_function


class ConditionalOp:
    def __init__(self, func: Callable[[Context], bool]):
        self.func = func

    def __call__(self, ctx: Context) -> bool:
        return self.func(ctx)


class IterativeOp:
    def __init__(self, func: Callable[[Context], bool]):
        self.func = func

    def __call__(self, ctx: Context) -> bool:
        return self.func(ctx)


class SequentialOp:
    @staticmethod
    def from_method(method: Callable[..., Any]) -> SequentialOp:
        func = context_adapter(method)
        return SequentialOp(func=func)

    def __init__(self, func: Callable[[Context], Any]) -> None:
        self.func = func

    def __call__(self, ctx: Context) -> None:
        self.func(ctx)


@dataclass
class Operation:
    """A concrete operation class that wraps a function with metadata."""

    func: Union[ConditionalOp, IterativeOp, SequentialOp]
    operation_number: OperationNumber | None = None
    operation_name: str | None = None
    domain_name: str | None = None
    depends_on: list[str] | None = None
    before: list[str] | None = None  # Operations this should come before
    # TODO move visualization metadata to a separate class
    shape: str = "box"
    color: str = "lightblue"
    level: int = 0
    sub_operations: list["Operation"] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.depends_on is None:
            self.depends_on = []
        if self.before is None:
            self.before = []

    @staticmethod
    def create_SeqOp(callable_method: Callable[..., Any], **kwargs: Any) -> Operation:
        if (
            not hasattr(callable_method, "_metadata")
            or not callable_method._metadata.is_operation
        ):
            raise ValueError("callable_method cannot be None")
        seq_op = SequentialOp.from_method(callable_method)
        metadata = callable_method._metadata
        if "operation_name" not in kwargs and metadata.name is not None:
            kwargs["operation_name"] = metadata.name
        if "operation_number" not in kwargs and metadata.operation_number is not None:
            kwargs["operation_number"] = metadata.operation_number
        if "depends_on" not in kwargs and metadata.depends_on is not None:
            kwargs["depends_on"] = metadata.depends_on
        return Operation(func=seq_op, **kwargs)

    @staticmethod
    def create_IterOp(callable_method: Callable[..., bool], **kwargs: Any) -> Operation:
        if (
            not hasattr(callable_method, "_metadata")
            or not callable_method._metadata.is_condition
        ):
            raise ValueError("callable_method cannot be None")
        iter_op = IterativeOp(context_adapter(callable_method))
        metadata = callable_method._metadata
        if "operation_name" not in kwargs and metadata.name is not None:
            kwargs["operation_name"] = metadata.name
        if "operation_number" not in kwargs and metadata.operation_number is not None:
            kwargs["operation_number"] = metadata.operation_number
        if "depends_on" not in kwargs and metadata.depends_on is not None:
            kwargs["depends_on"] = metadata.depends_on
        return Operation(func=iter_op, **kwargs)

    @property
    def operation_type(self) -> str:
        if isinstance(self.func, ConditionalOp):
            return "conditional"
        elif isinstance(self.func, IterativeOp):
            return "iterative"
        elif isinstance(self.func, SequentialOp):
            return "sequential"
        else:
            raise ValueError("Unknown operation type")

    def operation_metadata(self) -> OperationMetadata:
        return OperationMetadata(
            op_name=self.operation_name or "unknown",
            depends_on=self.depends_on,
            shape=self.shape,
            operation_number=self.operation_number,
            color=self.color,
            domain_name=self.domain_name,
        )

    @property
    def name(self) -> str | None:
        return (
            f"{self.domain_name}.{self.operation_name}"
            if self.domain_name
            else self.operation_name
        )

    @property
    def dependency_names(self) -> list[str]:
        if self.depends_on is None:
            return []
        if self.domain_name:
            return [f"{self.domain_name}.{dep}" for dep in self.depends_on]
        return self.depends_on

    def run(self, ctx: Context) -> Any:
        op_type = self.operation_type
        if op_type == "conditional":
            return self.func(ctx)
        elif op_type == "iterative":
            while self.func(ctx):
                for op in self.sub_operations:
                    op.run(ctx)
        elif op_type == "sequential":
            self.func(ctx)
        else:
            raise ValueError("Unknown operation type")


class OperationCollection:
    def __init__(self, operations: list[Operation] | None = None) -> None:
        self.ops: list[Operation] = operations if operations is not None else []

    @classmethod
    def discover(cls, obj: Any) -> "OperationCollection":
        """Auto-discover all decorated operation methods from an object.

        Args:
            obj: Object with @Model.operation or @Solver.operation decorated methods

        Returns:
            OperationCollection with all discovered operations

        Example:
            class MyModel(Model):
                @Model.operation(operation_number="0.3")
                def solve_energy(self, ...): pass

            # Auto-discover operations
            ops = OperationCollection.discover(my_model)
        """
        from .decorator import decorated_member_functions

        funcs = decorated_member_functions(obj)
        ops = [Operation.create_SeqOp(func) for func in funcs]
        return cls(ops)

    def add(
        self, operation: Union[Operation, OperationCollection]
    ) -> OperationCollection:
        if isinstance(operation, OperationCollection):
            self.ops.extend(operation.ops)
        else:
            self.ops.append(operation)
        return self

    def add_suboperation(
        self, operation: Operation, index: int | str = -1
    ) -> OperationCollection:
        if isinstance(index, str):
            for i, op in enumerate(self.ops):
                if op.operation_name == index:
                    self.ops[i].sub_operations.append(operation)
                    return self
            raise KeyError(f"Operation with operation_name '{index}' not found.")
        self.ops[index].sub_operations.append(operation)
        return self

    def __getitem__(self, index: int | str) -> Operation:
        if isinstance(index, str):
            for op in self.ops:
                if op.operation_name == index:
                    return op
            raise KeyError(f"Operation with operation_name '{index}' not found.")
        return self.ops[index]

    def __contains__(self, key: int | str) -> bool:
        """Check if operation exists by name (str) or index (int)."""
        if isinstance(key, str):
            return any(op.operation_name == key for op in self.ops)
        return 0 <= key < len(self.ops)

    def __len__(self) -> int:
        return len(self.ops)

    def __iter__(self) -> Iterator[Operation]:
        return iter(self.ops)

    def total_operations(self) -> int:
        def count_ops(ops: list[Operation]) -> int:
            total = 0
            for op in ops:
                total += 1
                if op.sub_operations:
                    total += count_ops(op.sub_operations)
            return total

        return count_ops(self.ops)


class Operations:
    def __init__(
        self, operations: list[Operation] | OperationCollection | None = None
    ) -> None:
        if isinstance(operations, OperationCollection):
            self.ops = operations.ops
        else:
            self.ops = operations if operations is not None else []

    def add(self, operation: Operation) -> Operations:
        self.ops.append(operation)
        return self

    def add_suboperation(self, operation: Operation) -> Operations:
        self.ops[-1].sub_operations.append(operation)
        return self

    def __getitem__(self, index: int | str) -> Operation:
        if isinstance(index, str):
            for op in self.ops:
                if op.operation_name == index:
                    return op
            raise KeyError(f"Operation with operation_name '{index}' not found.")
        return self.ops[index]

    def __len__(self) -> int:
        return len(self.ops)

    def __iter__(self) -> Iterator[Operation]:
        return iter(self.ops)

    def run(self, ctx: Context) -> None:
        for operation in self.ops:
            operation.run(ctx)


class StepBuilder:
    def __init__(self, operations: list[Operation] | None = None) -> None:
        self.operations = (
            Operations(operations) if operations is not None else Operations()
        )

    def __enter__(self) -> StepBuilder:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        pass

    def step(self, operation: Operation) -> StepBuilder:
        """Add an operation to the current scope."""
        self.operations.add(operation)
        return self

    def loop(self, operation: Operation, name: str | None = None) -> StepBuilder:
        """Create nested loop context.

        Args:
            operation: The loop operation
            name: Optional name override. If None, uses operation.operation_name
        """
        # If name is provided, override operation_name
        if name is not None:
            operation.operation_name = name

        self.operations.add(operation)
        return StepBuilder(operations=self.operations[-1].sub_operations)


class DAGResolver:
    """
    Resolves operation execution order using topological sort.

    Takes a StepBuilder (with structure) and OperationCollection (model ops)
    and merges them respecting dependencies.
    """

    def __init__(self) -> None:
        pass

    def _print_operations(
        self, builder: StepBuilder, title: str = "Operations"
    ) -> None:
        """
        Print operations in nested format with indentation for loops.

        Args:
            builder: StepBuilder to print
            title: Title for the output
        """
        Info(f"\n{title}:")

        def print_recursive(ops: list[Operation], indent: int = 0) -> None:
            prefix = "  " * indent
            for op in ops:
                if isinstance(op.func, IterativeOp):
                    # Loop operation - print in order where it appears
                    loop_name = op.operation_name or "loop"
                    Info(f"{prefix}Loop: {loop_name}")
                    if op.sub_operations:
                        print_recursive(op.sub_operations, indent + 1)
                else:
                    # Regular operation - print in order where it appears
                    op_name = op.operation_name or "unnamed"
                    deps = f" (depends_on: {op.depends_on})" if op.depends_on else ""
                    # Display operation_number
                    if op.operation_number is not None:
                        op_num = f" [#{str(op.operation_number)}]"
                    else:
                        op_num = ""
                    Info(f"{prefix}- {op_name}{op_num}{deps}")

        print_recursive(builder.operations.ops)
        Info("")  # Empty line for readability

    def resolve(
        self,
        builder: StepBuilder,
        additional_ops: OperationCollection,
    ) -> StepBuilder:
        """
        Returns a new StepBuilder with all operations in resolved order.

        Args:
            builder: StepBuilder with solver/algorithm structure
            additional_ops: Model operations to insert based on dependencies

        Returns:
            New StepBuilder with resolved operation order

        Raises:
            CyclicDependencyError: If circular dependencies detected
            MissingDependencyError: If dependency target doesn't exist
        """
        # Extract scopes from builder
        scopes = self._extract_scopes(builder)

        # Insert additional operations into appropriate scopes
        for op in additional_ops:
            self._insert_operation(op, scopes)

        # Build global operation map for dependency validation
        all_ops = {}
        for scope_name, ops in scopes.items():
            for op in ops:
                if op.operation_name:
                    all_ops[op.operation_name] = scope_name

        # Topologically sort operations within each scope
        for scope_name in scopes:
            scopes[scope_name] = self._topological_sort(
                scopes[scope_name], scope_name, all_ops
            )

        # Rebuild StepBuilder with sorted operations
        resolved = self._rebuild_builder(builder, scopes)

        # Print the resolved operations structure
        self._print_operations(resolved, "DAG Resolved Operations")

        return resolved

    def _extract_scopes(self, builder: StepBuilder) -> dict[str, list[Operation]]:
        """
        Extract all operations from StepBuilder organized by scope.

        For each scope, includes both regular operations AND loop operations
        that are direct children of that scope.

        Returns:
            Dictionary mapping scope name to list of operations in that scope.
        """
        scopes: dict[str, list[Operation]] = defaultdict(list)

        def extract_recursive(ops: list[Operation], parent_scope: str) -> None:
            for op in ops:
                # Add this operation to its parent scope
                scopes[parent_scope].append(op)

                if isinstance(op.func, IterativeOp):
                    # This is a loop - recursively extract its sub-operations
                    loop_name = op.operation_name or "loop"
                    extract_recursive(op.sub_operations, loop_name)

        extract_recursive(builder.operations.ops, "root")
        return scopes

    def _insert_operation(
        self,
        op: Operation,
        scopes: dict[str, list[Operation]],
    ) -> None:
        """
        Insert an operation into the appropriate scope based on dependencies.

        Infers scope from:
        1. Dependencies (depends_on) - place in same scope as dependencies
        2. Constraints (before) - place in same scope as constraint targets
        3. Default to innermost non-root loop if no dependencies

        Args:
            op: Operation to insert
            scopes: Dictionary of scopes to insert into
        """
        target_scope = None

        if op.depends_on or op.before:
            # Infer scope from dependencies and constraints
            # Build map of operation name -> scope
            op_to_scope = {}
            for scope_name, scope_ops in scopes.items():
                for scope_op in scope_ops:
                    if scope_op.operation_name:
                        op_to_scope[scope_op.operation_name] = scope_name

            # Find scopes of all dependencies and constraints
            dep_scopes = set()
            for dep in op.depends_on or []:
                if dep in op_to_scope:
                    dep_scopes.add(op_to_scope[dep])
            for constraint in op.before or []:
                if constraint in op_to_scope:
                    dep_scopes.add(op_to_scope[constraint])

            if dep_scopes:
                # Use the innermost scope (prefer nested loops over root)
                if "inner_loop" in dep_scopes:
                    target_scope = "inner_loop"
                elif len(dep_scopes) == 1:
                    target_scope = dep_scopes.pop()
                else:
                    # Multiple scopes - choose the most nested one
                    non_root_scopes = [s for s in dep_scopes if s != "root"]
                    if non_root_scopes:
                        target_scope = sorted(non_root_scopes)[0]
                    else:
                        target_scope = "root"

        if target_scope is None:
            # No dependencies - default to innermost loop or root
            available_loops = [s for s in scopes.keys() if s != "root"]
            if available_loops:
                # Prefer "inner_loop" if it exists, otherwise first available loop
                target_scope = (
                    "inner_loop"
                    if "inner_loop" in available_loops
                    else available_loops[0]
                )
            else:
                target_scope = "root"

        if target_scope not in scopes:
            # If scope doesn't exist, fall back to root
            target_scope = "root"

        scopes[target_scope].append(op)

    def _topological_sort(
        self,
        operations: list[Operation],
        scope_name: str,
        all_ops: dict[str, str] | None = None,
    ) -> list[Operation]:
        """
        Topologically sort operations based on dependencies.

        Loop operations are kept in their original relative position.
        Only regular (non-loop) operations are topologically sorted.

        Args:
            operations: List of operations to sort
            scope_name: Name of the scope (for error messages)
            all_ops: Global map of operation names to their scopes (for validation)

        Returns:
            Sorted list of operations

        Raises:
            CyclicDependencyError: If cycle detected
            MissingDependencyError: If dependency doesn't exist
        """
        if not operations:
            return []

        # Separate loop operations from regular operations
        loop_ops = [op for op in operations if isinstance(op.func, IterativeOp)]
        regular_ops = [op for op in operations if not isinstance(op.func, IterativeOp)]

        if not regular_ops:
            return loop_ops

        # Build operation lookup for regular ops only
        op_map = {op.operation_name: op for op in regular_ops if op.operation_name}

        # Build dependency graph
        graph: dict[str, list[str]] = {}
        in_degree: dict[str, int] = {}

        for op in regular_ops:
            if not op.operation_name:
                continue

            op_name = op.operation_name
            graph[op_name] = []
            in_degree[op_name] = 0

        # Add edges
        for op in regular_ops:
            if not op.operation_name:
                continue

            op_name = op.operation_name
            deps = op.depends_on or []
            before = op.before or []

            # Handle depends_on: this operation depends on these
            for dep in deps:
                # Check if dependency exists globally (for error reporting)
                if all_ops and dep not in all_ops:
                    raise MissingDependencyError(
                        f"Operation '{op_name}' in scope '{scope_name}' depends on "
                        f"'{dep}' which doesn't exist in any scope."
                    )

                # Only add edge if dependency is in the same scope
                # (cross-scope dependencies are valid and handled by hierarchy)
                if dep in op_map:
                    graph[dep].append(op_name)
                    in_degree[op_name] += 1

            # Handle before: this operation comes before these
            for before_op in before:
                # Check if target exists
                if all_ops and before_op not in all_ops:
                    raise MissingDependencyError(
                        f"Operation '{op_name}' in scope '{scope_name}' specifies before "
                        f"'{before_op}' which doesn't exist in any scope."
                    )

                # Only add edge if target is in the same scope
                if before_op in op_map:
                    # This creates edge: op_name -> before_op
                    graph[op_name].append(before_op)
                    in_degree[before_op] += 1

        # Kahn's algorithm for topological sort with priority queue for tie-breaking
        import heapq

        # Use priority queue with operation_number for deterministic ordering
        queue: list[tuple[tuple[int, int], str]] = []
        for name, degree in in_degree.items():
            if degree == 0:
                op = op_map[name]
                # Use tuple: (0, op_num) for numbered ops, (1, dummy) for unnumbered (sorts last)
                if op.operation_number:
                    priority = (0, op.operation_number)  # Has number - higher priority
                else:
                    priority = (1, OperationNumber([0]))  # No number - lower priority
                heapq.heappush(queue, (priority, name))  # type: ignore[misc]

        sorted_names: list[str] = []

        while queue:
            _, current = heapq.heappop(queue)
            sorted_names.append(current)

            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    op = op_map[neighbor]
                    if op.operation_number:
                        priority = (0, op.operation_number)
                    else:
                        priority = (1, OperationNumber([0]))
                    heapq.heappush(queue, (priority, neighbor))  # type: ignore[misc]

        # Check for cycles
        if len(sorted_names) != len(op_map):
            raise CyclicDependencyError(
                f"Cyclic dependency detected in scope '{scope_name}'. "
                f"Sorted {len(sorted_names)} operations but expected {len(op_map)}."
            )

        # Merge sorted regular operations with loop operations in correct order
        # Loop operations should maintain their relative position from the original order
        sorted_regular = [op_map[name] for name in sorted_names]
        result = []
        regular_idx = 0

        # Interleave loop operations at their original positions
        for orig_op in operations:
            if isinstance(orig_op.func, IterativeOp):
                # This is a loop - add it at this position
                result.append(orig_op)
            elif regular_idx < len(sorted_regular):
                # Add next sorted regular operation
                result.append(sorted_regular[regular_idx])
                regular_idx += 1

        # Add any remaining regular operations
        while regular_idx < len(sorted_regular):
            result.append(sorted_regular[regular_idx])
            regular_idx += 1

        return result

    def _rebuild_builder(
        self,
        original_builder: StepBuilder,
        sorted_scopes: dict[str, list[Operation]],
    ) -> StepBuilder:
        """
        Rebuild StepBuilder with sorted operations.

        Args:
            original_builder: Original builder with structure
            sorted_scopes: Dictionary of sorted operations by scope

        Returns:
            New StepBuilder with sorted operations
        """
        new_builder = StepBuilder()

        def rebuild_recursive(
            original_ops: list[Operation],
            target_builder: StepBuilder,
            parent_scope: str,
        ) -> None:
            for op in original_ops:
                if isinstance(op.func, IterativeOp):
                    # This is a loop - rebuild it with sorted sub-operations
                    loop_name = op.operation_name or "loop"
                    sorted_ops = sorted_scopes.get(loop_name, [])

                    # Create new operation with empty sub-operations
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

                    # Add to target builder and populate with sorted operations
                    with target_builder.loop(new_op) as loop_builder:
                        # Add operations in the order they appear (preserves sorted order)
                        for sub_op in sorted_ops:
                            if isinstance(sub_op.func, IterativeOp):
                                # Nested loop - recursively rebuild
                                rebuild_recursive([sub_op], loop_builder, loop_name)
                            else:
                                # Regular operation - add directly
                                loop_builder.step(sub_op)

        # Check if we have any loop structures
        has_loops = any(
            isinstance(op.func, IterativeOp) for op in original_builder.operations.ops
        )

        if has_loops:
            # Rebuild from original structure
            rebuild_recursive(original_builder.operations.ops, new_builder, "root")
        else:
            # Flat structure - just add sorted root operations
            if "root" in sorted_scopes:
                for op in sorted_scopes["root"]:
                    new_builder.step(op)

        return new_builder
