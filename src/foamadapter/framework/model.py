# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

from __future__ import annotations

from typing import Any, Callable, Protocol, TypeVar, cast, runtime_checkable

from foamadapter.framework.context import Context
from foamadapter.framework.operations import OperationCollection

from .decorator import operation as operation_func
from .decorator import condition as condition_func

F = TypeVar("F", bound=Callable[..., Any])
C = TypeVar("C", bound=type)


class ModelNamespace(Protocol):
    """Protocol defining the Model decorator namespace with operation decorator."""

    def __call__(self, cls: C) -> C:
        """Decorate a class as a Model."""
        ...

    # Operation decorator
    operation: Callable[..., Any]
    condition: Callable[..., bool]


def _Model(cls: C) -> C:
    """
    A class decorator to mark a class as a Model in the framework.
    Models contain a set of operations that are meant to extend a solver's functionality.

    Models should implement explicit load(), resolve(), and build() methods
    instead of using decorators.
    """
    return cls


# Tell mypy that Model is a namespace with attributes
Model = cast(ModelNamespace, _Model)

# Attach operation decorators
Model.operation = operation_func
Model.condition = condition_func


@runtime_checkable
class ModelInterface(Protocol):
    def execution_graph(self) -> OperationCollection: ...

    def operations(self) -> OperationCollection: ...

    def run(self, ctx: Context) -> None: ...
