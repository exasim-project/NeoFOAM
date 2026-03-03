# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
BaseSpec — shared base for ModelSpec and SolverSpec.

Contains the common decorator storage (config, operation) and the
operation → Operation assembly loop (_build_operations_for).
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from neofoam.framework.operation_wrapper import (
    DependencyResolver,
    wrap_operation,
)
from neofoam.framework.operations import Operation, SequentialOp
from neofoam.framework.types import OperationDef, OperationMetadata, OperationNumber


class BaseSpec:
    """Shared base for ModelSpec and SolverSpec.

    Provides:
    - ``config()`` decorator to register a config class
    - ``operation()`` decorator to register operations
    - ``_build_operations_for(runtime, suffix)`` to assemble Operation objects
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._config_class: Optional[type] = None
        self._operations: list[OperationDef] = []
        self._dependency_resolver = DependencyResolver()

    def config(self, cls: type) -> type:
        """Register the config class."""
        self._config_class = cls
        return cls

    def operation(
        self,
        operation_number: Optional[str] = None,
        depends_on: Optional[list[str]] = None,
        before: Optional[list[str]] = None,
        name: Optional[str] = None,
    ) -> Callable[..., Any]:
        """Decorator to register an operation."""

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self._operations.append(
                OperationDef(
                    func=func,
                    operation_number=operation_number,
                    depends_on=depends_on,
                    before=before,
                    name=name or func.__name__,
                )
            )
            return func

        return decorator

    def _build_operations_for(self, runtime: Any, suffix: str = "") -> list[Operation]:
        """Build Operation list with *runtime* as the ``self`` binding.

        Returns a fresh list per call so multiple runtimes never
        share wrapper state.
        """
        ops: list[Operation] = []
        for op_def in self._operations:
            wrapped = wrap_operation(op_def.func, runtime, self._dependency_resolver)

            op_name = op_def.name + suffix
            op = Operation(
                func=SequentialOp(wrapped),
                metadata=OperationMetadata(
                    op_name=op_name,
                    operation_number=(
                        OperationNumber(op_def.operation_number)
                        if op_def.operation_number
                        else None
                    ),
                    depends_on=op_def.depends_on or [],
                    before=op_def.before or [],
                ),
            )
            ops.append(op)
        return ops
