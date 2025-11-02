import inspect
from typing import Annotated, get_origin, get_args, Any, Callable
from .context import Context, FieldUpdates
from dataclasses import is_dataclass, dataclass


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
            f"Required parameter '{name}' "
            f"was not found in {var_name} and has no default value."
        )

    return call_args


@dataclass
class Step:
    """A concrete step class that wraps a function with metadata."""

    func: Callable
    cls: type
    step_number: int
    step_name: str
    domain: str | None = None
    depends_on: list[str] | None = None

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

    def __call__(self, *args, **kwargs):
        return self.func(self.cls, *args, **kwargs)

    def run(self, ctx: Context):
        return self.func.run(self.cls, ctx)


def update_dependencies(steps: list[Step]) -> list[Step]:
    """
    For each step, if depends_on is None, set it to depend on the previous step (by order).
    """
    for i, step in enumerate(steps):
        if i > 0:
            if step.depends_on is None:
                # Set dependency to previous step
                step.depends_on = [steps[i - 1].step_name]
            else:
                # Add previous step as dependency if not already present
                if steps[i - 1].step_name not in step.depends_on:
                    step.depends_on.append(steps[i - 1].step_name)
    return steps


def step(*, step_number: int, depends_on=None):
    """
    Decorator to mark a method as a step.
    Usage:
        @Solver.step(step_number=1)
        def foo(...): ...
        @Solver.step(step_number=2, depends_on=["step_one"])
        def bar(...): ...
    """

    def decorator(wrapped_func):
        sig = inspect.signature(wrapped_func)
        param_names = sig.parameters

        # @functools.wraps(wrapped_func)
        def context_wrapper(self, context: Context):
            # Build the arguments for the function call
            print(f"Calling {wrapped_func.__name__} with self={self}")
            call_args = {}

            for name in param_names:
                if name == "self":
                    call_args[name] = self
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

            # Call the original function with the unpacked arguments
            print(f"--- Running step: {wrapped_func.__name__} ---")
            results = wrapped_func(**call_args)

            
            if isinstance(results, FieldUpdates):
                print(f"--- Updating fields with: {results} ---")
                context.fields.update(results)
            else:
                print(
                    f"--- Warning: Step '{wrapped_func.__name__}' returned an unprocessed object: {type(results)} ---"
                )
            return results

        wrapped_func.run = context_wrapper

        wrapped_func._is_step = True
        wrapped_func._step_number = step_number
        wrapped_func._depends_on = depends_on or []

        return wrapped_func

    return decorator
