# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors
from __future__ import annotations

import inspect
from dataclasses import dataclass, field, is_dataclass
from typing import Annotated, Any, Callable, Iterator, Union, get_args, get_origin

from neofoam.framework.context import Context, FieldUpdates

from .types import OperationMetadata, OpType, OperationNumber


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
    call_args: dict[str, Any] = {}
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
    before: list[str] | None = None
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
        self.operations.add(operation)
        return self

    def loop(self, operation: Operation) -> StepBuilder:
        self.operations.add(operation)
        return StepBuilder(operations=self.operations[-1].sub_operations)
