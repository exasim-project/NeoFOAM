import inspect
from dataclasses import dataclass, field, is_dataclass
from typing import Annotated, Any, Callable, Union, get_args, get_origin

from foamadapter.framework.context import Context, FieldUpdates

from .types import OperationMetadata, OpType, StepNumber


def _get_value(ctx: Context, name, annotation: Any) -> dict[str, Any]:
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


def get_function_parameters(func: Callable) -> dict[str, type]:
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
        else:
            call_args[name] = context.fields[name]
    return call_args


def _get_operation_type(func: Callable) -> OpType:
    if hasattr(func, "_metadata"):
        return func._metadata.op_type
    raise ValueError("Function is not decorated with metadata")


def context_adapter(func: Callable) -> Callable[[Context], None]:
    func_paras = get_function_parameters(func)
    op_type = _get_operation_type(func)
    # TODO error handling if not FieldUpdates is returned

    def wrapped_function(context: Context):
        call_args = get_call_arguments(func_paras, context)
        results = func(**call_args)

        if op_type == OpType.STEP and isinstance(results, FieldUpdates):
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
    def from_method(method: Callable) -> "SequentialOp":
        func = context_adapter(method)
        return SequentialOp(func=func)

    def __init__(self, func: Callable[[Context], None]):
        self.func = func

    def __call__(self, ctx: Context) -> None:
        self.func(ctx)


@dataclass
class Operation:
    """A concrete step class that wraps a function with metadata."""

    func: Union[ConditionalOp, IterativeOp, SequentialOp]
    step_number: StepNumber = None
    step_name: str = None
    domain_name: str | None = None
    depends_on: list[str] | None = None
    shape: str = "box"
    color: str = "lightblue"
    level: int = 0
    sub_steps: list["Operation"] = field(default_factory=list)

    def __post_init__(self):
        if self.depends_on is None:
            self.depends_on = []

    @staticmethod
    def create_SeqOp(callable_method: Callable[[Context], None], **kwargs) -> "Operation":
        if not hasattr(callable_method, "_metadata") or not callable_method._metadata.is_step:
            raise ValueError("callable_method cannot be None")
        seq_op = SequentialOp.from_method(callable_method)
        if "step_name" not in kwargs and callable_method._metadata.name is not None:
            kwargs["step_name"] = callable_method._metadata.name
        if "step_number" not in kwargs and callable_method._metadata.step_number is not None:
            kwargs["step_number"] = callable_method._metadata.step_number
        if "depends_on" not in kwargs and callable_method._metadata.depends_on is not None:
            kwargs["depends_on"] = callable_method._metadata.depends_on
        return Operation(func=seq_op, **kwargs)

    @property
    def operation_type(self):
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
            op_name=self.step_name,
            depends_on=self.depends_on,
            shape=self.shape,
            step_number=self.step_number,
            color=self.color,
            domain_name=self.domain_name,
        )

    @property
    def name(self):
        return f"{self.domain_name}.{self.step_name}" if self.domain_name else self.step_name

    @property
    def dependency_names(self):
        if self.depends_on is None:
            return []
        if self.domain_name:
            return [f"{self.domain_name}.{dep}" for dep in self.depends_on]
        return self.depends_on

    def run(self, ctx):
        op_type = self.operation_type
        if op_type == "conditional":
            return self.func(ctx)
        elif op_type == "iterative":
            while self.func(ctx):
                for step in self.sub_steps:
                    step.run(ctx)
        elif op_type == "sequential":
            self.func(ctx)
        else:
            raise ValueError("Unknown operation type")


class OperationCollection:
    def __init__(self, operations: list[Operation] = None):
        if operations is not None:
            self.ops = operations
        else:
            self.ops: list[Operation] = []

    def add(self, operation: Union[Operation, "OperationCollection"]) -> "OperationCollection":
        if isinstance(operation, OperationCollection):
            self.ops.extend(operation.ops)
        else:
            self.ops.append(operation)
        return self

    def add_suboperation(
        self, operation: Operation, index: Union[int, str] = -1
    ) -> "OperationCollection":
        self.ops[index].sub_steps.append(operation)
        return self

    def __getitem__(self, index: Union[int, str]) -> Operation:
        if isinstance(index, str):
            for op in self.ops:
                if op.step_name == index:
                    return op
            raise KeyError(f"Operation with step_name '{index}' not found.")
        return self.ops[index]

    def __len__(self):
        return len(self.ops)

    def total_operations(self) -> int:
        def count_ops(ops: list[Operation]) -> int:
            total = 0
            for op in ops:
                total += 1
                if op.sub_steps:
                    total += count_ops(op.sub_steps)
            return total

        return count_ops(self.ops)


class Operations:
    def __init__(self, operations: list[Operation] = None):
        if operations is not None:
            self.ops = operations
        else:
            self.ops: list[Operation] = []

    def add(self, operation: Operation) -> "Operations":
        self.ops.append(operation)
        return self

    def add_suboperation(self, operation: Operation) -> "Operations":
        self.ops[-1].sub_steps.append(operation)
        return self

    def __getitem__(self, index: Union[int, str]) -> Operation:
        if isinstance(index, str):
            for op in self.ops:
                if op.step_name == index:
                    return op
            raise KeyError(f"Operation with step_name '{index}' not found.")
        return self.ops[index]

    def __len__(self):
        return len(self.ops)

    def run(self, ctx):
        for operation in self.ops:
            operation.run(ctx)


class StepBuilder:
    def __init__(self, operations: list[Operation] = None):
        if operations is not None:
            self.operations: Operations = Operations(operations)
        else:
            self.operations: Operations = Operations()

    def __enter__(self) -> "StepBuilder":
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def step(self, operation: Operation) -> "StepBuilder":
        self.operations.add(operation)
        return self

    def loop(self, operation: Operation) -> "StepBuilder":
        self.operations.add(operation)
        return StepBuilder(operations=self.operations[-1].sub_steps)
