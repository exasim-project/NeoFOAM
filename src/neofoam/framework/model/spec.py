# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
ModelSpec — immutable model definition registered once at module import.

Provides the same decorator API as the old ModelInstance but stores only
callables; all mutable state lives on ModelRuntime created per instantiate().
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Literal, Optional

from pydantic import BaseModel

from neofoam.framework.dependency_resolver import (
    DependencyResolver,
    wrap_with_dependency_resolution,
)
from neofoam.framework.operations import Operation, Operations, SequentialOp
from neofoam.framework.types import OperationMetadata, OperationNumber

from .runtime import ModelRuntime


class ModelSpec:
    """
    Immutable model definition.  Read-only after module import.

    Decorator methods store callables only; execution is delegated to
    ModelRuntime so that multiple runtimes can coexist independently.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.enabled = True

        self._load_func: Optional[Callable[..., Any]] = None
        self._resolve_func: Optional[Callable[..., Any]] = None
        self._build_func: Optional[Callable[..., Any]] = None
        self._detect_func: Optional[Callable[..., Any]] = None

        self._operations: list[tuple[Any, dict[str, Any]]] = []
        self._operation_collection_func: Optional[Callable[..., Any]] = None

        self._dependency_resolver = DependencyResolver()

    # ------------------------------------------------------------------
    # Stage decorators — store only, no side-effects
    # ------------------------------------------------------------------

    def load(self, func: Callable[[Path, str], Any]) -> Callable[[Path, str], Any]:
        """
        Register the LOAD function.

        Signature: ``def load(case_dir: Path, instance_id: str) -> SomeConfig``
        """
        self._load_func = func
        return func

    def resolve(self, func: Callable[[Any, Any], Any]) -> Callable[..., Any]:
        """
        Register the RESOLVE function.

        Signature: ``def resolve(config: MyConfig, ctx: ConfigContext) -> MyConfig``
        """
        self._resolve_func = func
        return func

    def build(self, func: Callable[..., list[Any]]) -> Callable[..., list[Any]]:
        """
        Register the BUILD function.

        Signature: ``def build(config: MyConfig) -> list[InitStep]``
        """
        self._build_func = func
        return func

    def detect(self, func: Callable[[], bool]) -> Callable[[], bool]:
        """Register the DETECT predicate."""
        self._detect_func = func
        return func

    def run_detect(self) -> bool:
        """Return True if the detect predicate passes (default: True)."""
        return self._detect_func() if self._detect_func is not None else True

    # ------------------------------------------------------------------
    # Operation decorators
    # ------------------------------------------------------------------

    def operation(
        self,
        operation_number: Optional[str] = None,
        depends_on: Optional[list[str]] = None,
        before: Optional[list[str]] = None,
        name: Optional[str] = None,
    ) -> Callable[..., Any]:
        """Decorator to register a model operation."""

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self._operations.append(
                (
                    func,  # store original; wrapping happens in _build_operations_for
                    {
                        "operation_number": operation_number,
                        "depends_on": depends_on,
                        "before": before,
                        "name": name or func.__name__,
                    },
                )
            )
            return func

        return decorator

    def operation_collection(
        self, func: Callable[..., Operations]
    ) -> Callable[..., Operations]:
        """Decorator for conditional operation dispatch."""
        self._operation_collection_func = func
        return func

    # ------------------------------------------------------------------
    # Instantiation
    # ------------------------------------------------------------------

    def instantiate(self, case_dir: Path, instance_id: str) -> ModelRuntime:
        """
        Create a fresh ModelRuntime for one instance of this spec.

        Calls the @load function and wraps the result in a ModelRuntime.
        """
        if self._load_func is None:
            raise ValueError(
                f"ModelSpec '{self.name}' has no @load stage. "
                "Every model must register a load function with @<spec>.load."
            )
        config = self._load_func(case_dir, instance_id)

        rt = ModelRuntime(
            spec=self,
            name=f"{self.name}_{instance_id}",
            config=config,
        )
        return rt

    # ------------------------------------------------------------------
    # Operation building (called by ModelRuntime.operations)
    # ------------------------------------------------------------------

    def _build_operations_for(self, runtime: ModelRuntime) -> list[Operation]:
        """
        Build Operation objects with *runtime* as the ``self`` binding.

        Returns a fresh list per call so multiple runtimes never share
        wrapper state.
        """
        from .config_injection import (
            _discover_configs_from_signature,
            _create_runtime_config_wrapper,
        )

        if self._operation_collection_func is not None:
            result = self._operation_collection_func(runtime)
            if isinstance(result, Operations):
                return list(result)
            return result  # type: ignore[no-any-return]

        ops: list[Operation] = []
        for func, metadata in self._operations:
            discovered = _discover_configs_from_signature(func)
            if discovered:
                wrapped = _create_runtime_config_wrapper(func, discovered, runtime)
            else:
                wrapped = wrap_with_dependency_resolution(
                    func, runtime, self._dependency_resolver
                )

            op = Operation(
                func=SequentialOp(wrapped),
                metadata=OperationMetadata(
                    op_name=metadata["name"],
                    operation_number=(
                        OperationNumber(metadata["operation_number"])
                        if metadata["operation_number"]
                        else None
                    ),
                    depends_on=metadata["depends_on"] or [],
                    before=metadata["before"] or [],
                ),
            )
            ops.append(op)
        return ops

    # ------------------------------------------------------------------
    # Plugin registration
    # ------------------------------------------------------------------

    def register_with(self, plugin_interface: type) -> "ModelSpec":
        """
        Register this ModelSpec with a PluginSystem interface.

        Creates a dynamic wrapper class whose ``get_model_instance``
        returns this spec, matching the pattern used by ModelInstance.
        """
        wrapper_class = type(
            self.name,
            (BaseModel,),
            {
                "__module__": plugin_interface.__module__,
                "__annotations__": {"model_type": Literal[self.name]},
                "model_type": self.name,
                "get_model_instance": lambda self_wrapper: self,
            },
        )

        if not hasattr(plugin_interface, "register"):
            raise TypeError(
                f"Plugin interface '{plugin_interface}' does not have a 'register' method"
            )
        plugin_interface.register(wrapper_class)
        return self

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_snake_case(name: str) -> str:
        s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
        return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def Model(name: str) -> ModelSpec:
    """Factory: create a named ModelSpec."""
    return ModelSpec(name)
