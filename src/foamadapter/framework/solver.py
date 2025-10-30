import inspect
import functools
from pydantic import BaseModel
from typing import ClassVar, Protocol, runtime_checkable, Callable
from foamadapter.framework.dag import NodeData

from typing import Annotated, get_origin, get_args,Protocol, Any

from .context import Context, FieldUpdates
from dataclasses import is_dataclass, dataclass


def _get_value(ctx: Context, name, annotation: any):
    call_args = {}
    ctx_var = ctx.fields

    is_annotated = (get_origin(annotation) is Annotated)
    var_name = "fields"
    if is_annotated:
        var_name = get_args(annotation)[1]
        ctx_var = getattr(ctx, var_name)

    if name in ctx_var:
        call_args[name] = ctx_var[name]
    else:
        raise KeyError(
            f"Required parameter '{name}' "
            f"was not found in {var_name} and has no default value."
        )
    
    return call_args

@dataclass
class Step:
    """A concrete step class that wraps a function with metadata."""
    func: Callable
    step_number: int
    step_name: str
    depends_on: list[str] | None = None
    
    def __call__(self, *args, **kwargs):
        return self.func(self,*args, **kwargs)
    
    def run(self, ctx: Context):
        return self.func.run(ctx)
    




def _update_dependencies(steps: list[Step]) -> list[Step]:
    """
    For each step, if depends_on is None, set it to depend on the previous step (by order).
    """
    for i, step in enumerate(steps):
        if i > 0:
            if step.depends_on is None:
                # Set dependency to previous step
                step.depends_on = [steps[i-1].step_name]
            else:
                # Add previous step as dependency if not already present
                if steps[i-1].step_name not in step.depends_on:
                    step.depends_on.append(steps[i-1].step_name)
    return steps

def _step(*, step_number: int, depends_on=None):
    """
    Decorator to mark a method as a step.
    Usage:
        @Solver.step(step_number=1)
        def foo(...): ...
        @Solver.step(step_number=2, depends_on=["step_one"])
        def bar(...): ...
    """
    # Get the function's signature and parameter names ONCE
    def decorator(wrapped_func):
        sig = inspect.signature(wrapped_func)
        param_names = sig.parameters

        @functools.wraps(wrapped_func)
        def context_wrapper(context: Context):
            # Build the arguments for the function call
            call_args = {}
            
            for name in param_names:
                param = sig.parameters[name]
                annotation = param.annotation
                if is_dataclass(annotation):
                    dc_sig = inspect.signature(annotation)

                    dc_args = {}

                    for dc_name, dc_param in dc_sig.parameters.items():
                        dc_annotation = dc_sig.parameters[dc_name].annotation
                        dc_args.update(_get_value(context,dc_name,dc_annotation))
                        
                    call_args[name] = annotation(**dc_args)
                else:
                    call_args.update(_get_value(context,name,annotation))
            

            # Call the original function with the unpacked arguments
            print(f"--- Running step: {wrapped_func.__name__} ---")
            results = wrapped_func(**call_args) 

            # 3. Process the results
            
            # Normalize results to a tuple to handle single or multiple returns
            if not isinstance(results, tuple):
                results = (results,) # Make it a single-item tuple

            for item in results:
                if isinstance(item, FieldUpdates):
                    print(f"--- Updating fields with: {item} ---")
                    context.fields.update(item)            
                else:
                    print(f"--- Warning: Step '{wrapped_func.__name__}' returned an unprocessed object: {type(item)} ---")
        
        wrapped_func.run = context_wrapper

        wrapped_func._is_step = True
        wrapped_func._step_number = step_number
        wrapped_func._depends_on = depends_on or []

        return wrapped_func
    
    return decorator


def Solver(cls):
    step_functions: list[Step] = []
    
    # Collect all step functions
    for attr_name in dir(cls):
        attr = getattr(cls, attr_name)
        if getattr(attr, "_is_step", False):
            step_functions.append(
                Step(
                    func=attr,
                    step_number=attr._step_number,
                    depends_on=attr._depends_on,
                    step_name=attr.__name__,
                )
            )
    # Sort steps by step_number
    step_functions = sorted(step_functions, key=lambda s: s.step_number)
    
    # Automatically set dependencies if not provided
    step_functions = _update_dependencies(step_functions)
    
    # Store the step functions
    cls._steps = step_functions
    cls.number_steps = classmethod(lambda c: len(c._steps))
    
    return cls

Solver.step = staticmethod(_step)

@runtime_checkable
class SolverInterface(Protocol):
    _steps: ClassVar[list[Step]]

    @classmethod
    def number_steps(cls) -> int:
        ...

    def dependencies(self) -> list[NodeData]:
        ...

    def main_loop(self, ctx: Context):
        ...

