# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

"""Operation primitives, adapters, and the Operations container."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Union

from neofoam import telemetry
from neofoam.framework import execution_observer
from neofoam.framework.context import Context, LoopEntry

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
        """Run on ``ctx``, unless its ``first``/``final`` mark skips this pass."""
        if (self.metadata.first or self.metadata.final or ctx.run_every_pass) and self._skipped(
            ctx
        ):
            return None
        # loops report their predicate evaluations instead (_loop_test)
        observer = None if self.operation_type == "iterative" else execution_observer.current()
        if observer is not None:
            observer.before_operation(self, ctx)
        if not telemetry.is_active():
            result = self._execute(ctx)
        else:
            with telemetry.span(
                self.metadata.op_name or self.metadata.name or "operation",
                operation_type=self.operation_type,
                domain=self.metadata.domain_name,
            ):
                result = self._execute(ctx)
        if observer is not None:
            observer.after_operation(self, ctx)
        return result

    def _loop_test(self, ctx: Context, iteration: int) -> bool:
        observer = execution_observer.current()
        if observer is None:
            return bool(self.func(ctx))
        observer.before_loop_test(self, ctx, iteration)
        result = bool(self.func(ctx))
        observer.after_loop_test(self, ctx, iteration, result)
        return result

    def _skipped(self, ctx: Context) -> bool:
        """Whether the pass of the wrapping loops around the operation skips it."""
        every_pass = self.name in ctx.run_every_pass
        if every_pass and not self.metadata.final:
            raise ValueError(_not_final(self))
        wrapping = [entry for entry in ctx.loops if entry.wrapping]
        if self.metadata.first:
            return not all(entry.first for entry in wrapping)
        if self.metadata.final and not every_pass:
            return not all(entry.final for entry in wrapping)
        return False

    def _execute(self, ctx: Context) -> Any:
        op_type = self.operation_type
        if op_type == "conditional":
            return self.func(ctx)
        elif op_type == "iterative":
            self._loop(ctx)
        elif op_type == "sequential":
            self.func(ctx)
        else:
            raise ValueError("Unknown operation type")

    def _loop(self, ctx: Context) -> None:
        """Run the loop's passes, its entry on ``ctx.loops`` while it runs."""
        wrapping = self.metadata.wraps is not None
        loops = ctx.loops
        index = len(loops)
        loops.append(LoopEntry(self.name, 0, wrapping))
        try:
            iteration = 0
            # the observer counts the passes from 1
            while self._loop_test(ctx, iteration + 1):
                for op in self.sub_operations:
                    op.run(ctx)
                iteration += 1
                loops[index] = LoopEntry(self.name, iteration, wrapping)
        finally:
            del loops[index:]


def _not_final(op: Operation) -> str:
    kind = "first" if op.metadata.first else "not marked final"
    return (
        f"run_every_pass names {op.name}, which is {kind}; only an operation marked final "
        "can run in every pass of its wrapping loops"
    )


def _check_run_every_pass(ops: list[Operation], names: set[str]) -> None:
    """Raise ``ValueError`` unless each of ``names`` is a ``final`` operation of ``ops``."""
    found: dict[str, Operation] = {}

    def walk(scope: list[Operation]) -> None:
        for op in scope:
            if op.name is not None:
                found[op.name] = op
            walk(op.sub_operations)

    walk(ops)
    for name in sorted(names):
        if name not in found:
            final = sorted(other for other, op in found.items() if op.metadata.final)
            raise ValueError(
                f"run_every_pass names {name}, which is no operation of the graph; "
                f"its operations marked final: {final}"
            )
        if not found[name].metadata.final:
            raise ValueError(_not_final(found[name]))


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

    def add(self, operation: Union[Operation, "Operations", list[Operation]]) -> "Operations":
        if isinstance(operation, Operations):
            self.ops.extend(operation.ops)
        elif isinstance(operation, list):
            self.ops.extend(operation)
        else:
            self.ops.append(operation)
        return self

    def add_suboperation(self, operation: Operation, index: int | str = -1) -> "Operations":
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
        """Run the operations on ``ctx``; ``ValueError`` if ``ctx.run_every_pass`` is invalid."""
        if ctx.run_every_pass:
            _check_run_every_pass(self.ops, ctx.run_every_pass)
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
        self.operations = Operations(operations) if operations is not None else Operations()

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
