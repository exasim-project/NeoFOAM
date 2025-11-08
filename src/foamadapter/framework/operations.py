from dataclasses import dataclass, field
import functools
import inspect
from dataclasses import is_dataclass
from typing import Union, Callable
from foamadapter.framework.dag import StepNumber, NodeData
from foamadapter.framework.context import Context
from foamadapter.framework.step import _get_value

# def step(*, step_number: int, depends_on=None):
#     """
#     Decorator to mark a method as a step.
#     Usage:
#         @Solver.step(step_number=1)
#         def foo(...): ...
#         @Solver.step(step_number=2, depends_on=["step_one"])
#         def bar(...): ...
#     """

#     def decorator(wrapped_func):
#         sig = inspect.signature(wrapped_func)
#         param_names = sig.parameters

#         # @functools.wraps(wrapped_func)
#         def context_wrapper(self, context: Context):
#             # Build the arguments for the function call
#             print(f"Calling {wrapped_func.__name__} with self={self}")
#             call_args = {}

#             for name in param_names:
#                 if name == "self":
#                     call_args[name] = self
#                     continue

#                 param = sig.parameters[name]
#                 annotation = param.annotation
#                 if is_dataclass(annotation):
#                     dc_sig = inspect.signature(annotation)

#                     dc_args = {}

#                     for dc_name, dc_param in dc_sig.parameters.items():
#                         dc_annotation = dc_sig.parameters[dc_name].annotation
#                         dc_args.update(_get_value(context, dc_name, dc_annotation))

#                     call_args[name] = annotation(**dc_args)
#                 elif annotation == Context:
#                     call_args[name] = context
#                     continue
#                 else:
#                     call_args.update(_get_value(context, name, annotation))

#             # Call the original function with the unpacked arguments
#             print(f"--- Running step: {wrapped_func.__name__} ---")
#             results = wrapped_func(**call_args)

            
#             if isinstance(results, FieldUpdates):
#                 print(f"--- Updating fields with: {results} ---")
#                 context.fields.update(results)
#             else:
#                 print(
#                     f"--- Warning: Step '{wrapped_func.__name__}' returned an unprocessed object: {type(results)} ---"
#                 )
#             return results

#         wrapped_func.run = context_wrapper

#         wrapped_func._is_step = True
#         wrapped_func._step_number = step_number
#         wrapped_func._depends_on = depends_on or []

#         return wrapped_func

#     return decorator

class ContextAdapter:

    def __init__(self, context: Context):
        pass


def context_adapter(func: Callable, instance ) -> Callable[[Context], None]:

    sig = inspect.signature(func)
    param_names = sig.parameters

    call_args = {}

    for name in param_names:
        if name == "self":
            call_args[name] = instance
            continue

        param = sig.parameters[name]
        annotation = param.annotation
        if is_dataclass(annotation):
            dc_sig = inspect.signature(annotation)

            dc_args = {}

            for dc_name, dc_param in dc_sig.parameters.items():
                dc_annotation = dc_sig.parameters[dc_name].annotation
                dc_args.update(_get_value(context, dc_name, dc_annotation))

            call_args[name] = annotation(**dc_args)
        elif annotation == Context:
            call_args[name] = context
            continue
        else:
            call_args.update(_get_value(context, name, annotation))
    

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
    def from_method(method, instance):

        func = functools.partial(method.run, instance)
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
    domain: str | None = None
    depends_on: list[str] | None = None
    shape: str = "box"
    color: str = "lightblue"
    level: int = 0
    sub_steps: list["Operation"] = field(default_factory=list)

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

    def node_data(self) -> NodeData:
        return NodeData(
            name=self.name,
            depends_on=self.dependency_names,
            shape=self.shape,
            step_number=self.step_number,
            color=self.color,
        )
   


    @property
    def name(self):
        return f"{self.domain}.{self.step_name}" if self.domain else self.step_name

    @property
    def dependency_names(self):
        if self.depends_on is None:
            return []
        if self.domain:
            return [f"{self.domain}.{dep}" for dep in self.depends_on]
        return self.depends_on

    def run(self, ctx):
        if self.operation_type == "conditional":
            return self.func(ctx)
        elif self.operation_type == "iterative":
            while self.func(ctx):
                for step in self.sub_steps:
                    step.run(ctx)
        elif self.operation_type == "sequential":
            self.func(ctx)

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
    
    
    def add_suboperation(self, operation: Operation, index: Union[int,str] = -1) -> "OperationCollection":
        self.ops[index].sub_steps.append(operation)
        return self
    
    def __getitem__(self, index: Union[int,str]) -> Operation:
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
    
    def __getitem__(self, index: Union[int,str]) -> Operation:
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
