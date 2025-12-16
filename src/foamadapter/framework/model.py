# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

from __future__ import annotations

from typing import Any, Callable, Protocol, TypeVar, cast, runtime_checkable

from foamadapter.framework.context import Context
from foamadapter.framework.operations import OperationCollection

from .decorator import operation as operation_func
from .initialization import load, resolve_dependencies, build
from .initialization.helpers import field as field_func
from .initialization.helpers import lazy as lazy_func
from .initialization.helpers import model as model_func
from .initialization.helpers import operator as operator_func

F = TypeVar("F", bound=Callable[..., Any])
C = TypeVar("C", bound=type)


class ModelNamespace(Protocol):
    """Protocol defining the Model decorator namespace with all helper attributes."""

    def __call__(self, cls: C) -> C:
        """Decorate a class as a Model."""
        ...

    # Stage decorators
    load: Callable[[F], F]
    resolve_dependencies: Callable[[F], F]
    build: Callable[[F], F]

    # Operation decorator
    operation: Callable[..., Any]

    # Helper functions
    field: Callable[..., Any]
    operator: Callable[..., Any]
    lazy: Callable[..., Any]
    model: Callable[..., Any]


def _Model(cls: C) -> C:
    """
    A class decorator to mark a class as a Model in the framework.
    Models contain a set of operations that are meant to extend a solver's functionality.
    """
    return cls


# Tell mypy that Model is a namespace with attributes
Model = cast(ModelNamespace, _Model)

# Attach helper functions and decorators
Model.operation = operation_func
Model.load = load
Model.resolve_dependencies = resolve_dependencies
Model.build = build
Model.field = field_func
Model.operator = operator_func
Model.lazy = lazy_func
Model.model = model_func


@runtime_checkable
class ModelInterface(Protocol):
    def operations(self) -> OperationCollection: ...

    def run(self, ctx: Context) -> None: ...
