# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

"""Operation primitives, adapters, and the Operations container."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Union

from neofoam import telemetry
from neofoam.framework.context import Context

from .types import OperationMetadata, OperationNumber


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
    def __init__(self, func: Callable[[Context], Any]) -> None:
        self.func = func

    def __call__(self, ctx: Context) -> None:
        self.func(ctx)


@dataclass
class Operation:
    """A concrete operation class that wraps a function with metadata.

    Stores an :class:`OperationMetadata` instance as the single source of
    truth for name, numbering, dependencies, and visualisation hints.
    """

    func: Union[ConditionalOp, IterativeOp, SequentialOp]
    metadata: OperationMetadata = field(default_factory=OperationMetadata)
    level: int = 0
    sub_operations: list["Operation"] = field(default_factory=list)

    # --- Convenience accessors delegating to metadata ---

    @property
    def operation_name(self) -> str | None:
        return self.metadata.op_name

    @property
    def operation_number(self) -> OperationNumber | None:
        return self.metadata.operation_number

    @property
    def domain_name(self) -> str | None:
        return self.metadata.domain_name

    @property
    def depends_on(self) -> list[str]:
        return self.metadata.depends_on if self.metadata.depends_on is not None else []

    @property
    def before(self) -> list[str]:
        return self.metadata.before if self.metadata.before is not None else []

    @property
    def shape(self) -> str:
        return self.metadata.shape

    @property
    def color(self) -> str | None:
        return self.metadata.color

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

    @property
    def name(self) -> str | None:
        return self.metadata.name

    @property
    def dependency_names(self) -> list[str]:
        return self.metadata.dependencies

    def run(self, ctx: Context) -> Any:
        if not telemetry.is_active():
            return self._execute(ctx)
        with telemetry.span(
            self.metadata.op_name or self.metadata.name or "operation",
            operation_type=self.operation_type,
            domain=self.metadata.domain_name,
        ):
            return self._execute(ctx)

    def _execute(self, ctx: Context) -> Any:
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


class Operations:
    """Unified container for Operation objects.

    Supports flexible add (single, list, or another Operations),
    suboperation management, iteration, indexing, and batch execution.
    """

    def __init__(self, operations: list[Operation] | None = None) -> None:
        if isinstance(operations, list):
            self.ops: list[Operation] = operations
        else:
            self.ops = operations if operations is not None else []

    def add(
        self, operation: Union[Operation, "Operations", list[Operation]]
    ) -> "Operations":
        if isinstance(operation, Operations):
            self.ops.extend(operation.ops)
        elif isinstance(operation, list):
            self.ops.extend(operation)
        else:
            self.ops.append(operation)
        return self

    def add_suboperation(
        self, operation: Operation, index: int | str = -1
    ) -> "Operations":
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

    def __len__(self) -> int:
        return len(self.ops)

    def __iter__(self) -> Iterator[Operation]:
        return iter(self.ops)

    def run(self, ctx: Context) -> None:
        for operation in self.ops:
            operation.run(ctx)

    def total_operations(self) -> int:
        def count_ops(ops: list[Operation]) -> int:
            total = 0
            for op in ops:
                total += 1
                if op.sub_operations:
                    total += count_ops(op.sub_operations)
            return total

        return count_ops(self.ops)


# Backward-compatible alias — use Operations directly in new code.
OperationCollection = Operations


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
        self.operations.add(operation)
        return self

    def loop(self, operation: Operation) -> StepBuilder:
        self.operations.add(operation)
        return StepBuilder(operations=self.operations[-1].sub_operations)
