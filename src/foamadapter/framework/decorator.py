# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors
from __future__ import annotations

import functools
import inspect
from typing import Any, Callable, TypeVar

from .types import OperationMetadata, OpType, StepNumber


F = TypeVar("F", bound=Callable[..., Any])


def _is_decorated_method(method: Callable[..., Any]) -> bool:
    return callable(method) and hasattr(method, "_metadata")


def decorated_member_functions(instance: Any) -> list[Callable[..., Any]]:
    decorated_functions = []
    for name, method in vars(instance.__class__).items():
        if _is_decorated_method(method):
            # Get the bound method
            # otherwise we would be appending unbound methods aka without self
            bound_method = getattr(instance, name)
            decorated_functions.append(bound_method)
    return decorated_functions


def step(
    _func: F | None = None,
    step_number: StepNumber | int | None = None,
    depends_on: list[str] | None = None,
) -> F | Callable[[F], F]:
    def _step_decorator(_func: F) -> F:
        """Decorator to mark a function as a step in the workflow."""

        @functools.wraps(_func)
        def wrapper(*args: object, **kwargs: object) -> Any:
            return _func(*args, **kwargs)

        step_num = (
            StepNumber(step_number) if isinstance(step_number, int) else step_number
        )
        wrapper._metadata = OperationMetadata(  # type: ignore[attr-defined]
            op_type=OpType.STEP,
            op_name=_func.__name__,
            step_number=step_num,
            depends_on=depends_on,
        )
        return wrapper  # type: ignore[return-value]

    if _func is None:
        return _step_decorator
    else:
        return _step_decorator(_func)


def condition(
    _func: F | None = None,
    step_number: StepNumber | int | None = None,
    depends_on: list[str] | None = None,
) -> F | Callable[[F], F]:
    def _condition_decorator(_func: F) -> F:
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
        def wrapper(*args: object, **kwargs: object) -> Any:
            return _func(*args, **kwargs)

        step_num = (
            StepNumber(step_number) if isinstance(step_number, int) else step_number
        )
        wrapper._metadata = OperationMetadata(  # type: ignore[attr-defined]
            op_type=OpType.CONDITION,
            op_name=_func.__name__,
            step_number=step_num,
            depends_on=depends_on,
        )
        return wrapper  # type: ignore[return-value]

    if _func is None:
        return _condition_decorator
    else:
        return _condition_decorator(_func)
