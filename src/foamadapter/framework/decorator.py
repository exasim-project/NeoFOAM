# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors
from __future__ import annotations

import functools
import inspect
from typing import Any, Callable, TypeVar, overload, Optional, Union

from .types import OperationMetadata, OpType, OperationNumber


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


@overload
def operation(_func: F) -> F:
    """Decorator form: @operation (without parentheses)."""
    ...


@overload
def operation(
    _func: None = None,
    *,
    operation_number: Union[Union[OperationNumber, int], str, None] = None,
    depends_on: Optional[list[str]] = None,
) -> Callable[[F], F]:
    """Decorator factory form: @operation(...) (with parentheses)."""
    ...


def operation(
    _func: Optional[F] = None,
    *,
    operation_number: Union[Union[OperationNumber, int], str, None] = None,
    depends_on: Optional[list[str]] = None,
) -> Union[F, Callable][[F], F]:
    """decorator Factory to mark a decorator to mark a function as an operation in the workflow."""

    def _operation_decorator(func: F) -> F:
        """Decorator to mark a function as an operation in the workflow."""

        @functools.wraps(func)
        def wrapper(*args: object, **kwargs: object) -> Any:
            return func(*args, **kwargs)

        op_num: Optional[OperationNumber] = None
        if isinstance(operation_number, (int, str)):
            op_num = OperationNumber(operation_number)
        elif operation_number is not None:
            op_num = operation_number
        wrapper._metadata = OperationMetadata(  # type: ignore[attr-defined]
            op_type=OpType.OPERATION,
            op_name=func.__name__,
            operation_number=op_num,
            depends_on=depends_on,
        )
        return wrapper  # type: ignore[return-value]

    if _func is None:
        return _operation_decorator
    else:
        return _operation_decorator(_func)


@overload
def condition(_func: F) -> F:
    """Decorator form: @condition (without parentheses)."""
    ...


@overload
def condition(
    _func: None = None,
    *,
    operation_number: Union[Union[OperationNumber, int], str, None] = None,
    depends_on: Optional[list[str]] = None,
) -> Callable[[F], F]:
    """Decorator factory form: @condition(...) (with parentheses)."""
    ...


def condition(
    _func: Optional[F] = None,
    *,
    operation_number: Union[Union[OperationNumber, int], str, None] = None,
    depends_on: Optional[list[str]] = None,
) -> Union[F, Callable][[F], F]:
    """decorator Factory to mark a decorator to mark a function as a condition in the workflow."""

    def _condition_decorator(func: F) -> F:
        """Decorator to mark a function as a condition in the workflow."""
        # check if return value is a Condition instance
        return_annotation = inspect.signature(func).return_annotation
        # Require return annotation to be present and be Condition
        if return_annotation is inspect.Signature.empty:
            raise TypeError(
                f"Function {func.__name__} must have a return type annotation of 'bool'"
            )

        if return_annotation.__name__ != bool.__name__:
            raise TypeError(
                f"Return type of {func.__name__} must be bool not a {return_annotation}"
            )

        @functools.wraps(func)
        def wrapper(*args: object, **kwargs: object) -> Any:
            return func(*args, **kwargs)

        op_num: Optional[OperationNumber] = None
        if isinstance(operation_number, (int, str)):
            op_num = OperationNumber(operation_number)
        elif operation_number is not None:
            op_num = operation_number
        wrapper._metadata = OperationMetadata(  # type: ignore[attr-defined]
            op_type=OpType.CONDITION,
            op_name=func.__name__,
            operation_number=op_num,
            depends_on=depends_on,
        )
        return wrapper  # type: ignore[return-value]

    if _func is None:
        return _condition_decorator
    else:
        return _condition_decorator(_func)
