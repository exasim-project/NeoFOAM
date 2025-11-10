import functools
import inspect
from typing import Any

from .types import OperationMetadata, OpType, StepNumber


def _is_decorated_method(method: callable) -> bool:
    return callable(method) and hasattr(method, "_metadata")


def decorated_member_functions(instance: Any) -> list[callable]:
    decorated_functions = []
    for name, method in vars(instance.__class__).items():
        if _is_decorated_method(method):
            # Get the bound method
            # otherwise we would be appending unbound methods aka without self
            bound_method = getattr(instance, name)
            decorated_functions.append(bound_method)
    return decorated_functions


def step(
    _func=None, step_number: StepNumber | int | None = None, depends_on: list[str] | None = None
):
    def _step_decorator(_func):
        """Decorator to mark a function as a step in the workflow."""

        @functools.wraps(_func)
        def wrapper(*args, **kwargs):
            return _func(*args, **kwargs)

        wrapper._metadata = OperationMetadata(
            op_type=OpType.STEP,
            op_name=_func.__name__,
            step_number=step_number,
            depends_on=depends_on,
        )
        return wrapper

    if _func is None:
        return _step_decorator
    else:
        return _step_decorator(_func)


def condition(
    _func=None, step_number: StepNumber | int | None = None, depends_on: list[str] | None = None
):
    def _condition_decorator(_func):
        # check if return value is a Condition instance
        return_annotation = inspect.signature(_func).return_annotation
        # Require return annotation to be present and be Condition
        if return_annotation is inspect.Signature.empty:
            raise TypeError(
                f"Function {_func.__name__} must have a return type annotation of 'bool'"
            )

        if return_annotation.__name__ != bool.__name__:
            raise TypeError(
                f"Return type of {_func.__name__} must be bool not a {return_annotation}"
            )

        @functools.wraps(_func)
        def wrapper(*args, **kwargs):
            return _func(*args, **kwargs)

        wrapper._metadata = OperationMetadata(
            op_type=OpType.CONDITION,
            op_name=_func.__name__,
            step_number=step_number,
            depends_on=depends_on,
        )
        return wrapper

    if _func is None:
        return _condition_decorator
    else:
        return _condition_decorator(_func)
