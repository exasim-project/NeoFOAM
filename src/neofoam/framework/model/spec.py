# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
ModelSpec — immutable model definition registered once at module import.

Provides the same decorator API as the old ModelInstance but stores only
callables; all mutable state lives on ModelRuntime created per instantiate().
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Union

from pydantic import BaseModel

from neofoam.framework.base_spec import BaseSpec
from neofoam.framework.operations import Operation, Operations

from .runtime import ModelRuntime


def _validate_param_count(
    func: Callable[..., Any],
    expected: int,
    decorator: str,
) -> None:
    import inspect

    actual = len(inspect.signature(func).parameters)
    if actual != expected:
        raise TypeError(
            f"{decorator} function '{func.__name__}' has {actual} parameter(s); "
            f"expected exactly {expected}."
        )


def _validate_min_param_count(
    func: Callable[..., Any],
    minimum: int,
    decorator: str,
) -> None:
    import inspect

    actual = len(inspect.signature(func).parameters)
    if actual < minimum:
        raise TypeError(
            f"{decorator} function '{func.__name__}' has {actual} parameter(s); "
            f"expected at least {minimum}."
        )


@dataclass
class DetectResult:
    """Result of a model detection check."""

    detected: bool
    instance_ids: list[str] = field(default_factory=list)


class ModelSpec(BaseSpec):
    """
    Immutable model definition.  Read-only after module import.

    Decorator methods store callables only; execution is delegated to
    ModelRuntime so that multiple runtimes can coexist independently.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.enabled = True

        self._load_func: Optional[Callable[..., Any]] = None
        self._resolve_func: Optional[Callable[..., Any]] = None
        self._resolve_config_params: list[dict[str, Any]] = []
        self._build_func: Optional[Callable[..., Any]] = None
        self._build_config_params: list[dict[str, Any]] = []
        self._detect_func: Optional[Callable[..., Any]] = None

        self._operation_collection_func: Optional[Callable[..., Any]] = None

    # ------------------------------------------------------------------
    # Stage decorators — store only, no side-effects
    # ------------------------------------------------------------------

    def load(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """
        Register the LOAD function (optional override for custom logic).

        Signature: ``def load(case_dir: Path, entry: dict) -> SomeConfig``
        """
        _validate_param_count(func, expected=2, decorator="@load")
        self._load_func = func
        return func

    def resolve(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """
        Register the RESOLVE function.

        Signature: ``def resolve(self: ModelRuntime, ctx: ConfigContext, cfg: MyConfig, ...) -> Config``
        """
        from neofoam.framework.operation_wrapper import discover_configs_from_signature

        _validate_min_param_count(func, minimum=2, decorator="@resolve")
        self._resolve_func = func
        self._resolve_config_params = discover_configs_from_signature(func)
        return func

    def build(self, func: Callable[..., list[Any]]) -> Callable[..., list[Any]]:
        """
        Register the BUILD function.

        Signature: ``def build(self: ModelRuntime, cfg: MyConfig, ...) -> list[InitStep]``
        """
        from neofoam.framework.operation_wrapper import discover_configs_from_signature

        _validate_min_param_count(func, minimum=1, decorator="@build")
        self._build_func = func
        self._build_config_params = discover_configs_from_signature(func)
        return func

    def detect(
        self, func: Callable[..., Union[bool, list[str]]]
    ) -> Callable[..., Union[bool, list[str]]]:
        """Register the DETECT predicate."""
        _validate_param_count(func, expected=1, decorator="@detect")
        self._detect_func = func
        return func

    def run_detect(self, case_dir: Path) -> DetectResult:
        """Run the detect predicate and return a DetectResult."""
        if self._detect_func is None:
            return DetectResult(detected=True)

        result = self._detect_func(case_dir)

        if isinstance(result, list):
            return DetectResult(detected=len(result) > 0, instance_ids=result)
        return DetectResult(detected=bool(result))

    # ------------------------------------------------------------------
    # Operation decorators (operation() inherited from BaseSpec)
    # ------------------------------------------------------------------

    def operation_collection(
        self, func: Callable[..., Operations]
    ) -> Callable[..., Operations]:
        """Decorator for conditional operation dispatch."""
        self._operation_collection_func = func
        return func

    # ------------------------------------------------------------------
    # Instantiation
    # ------------------------------------------------------------------

    def instantiate(
        self,
        case_dir: Path,
        entry: Optional[dict[str, Any]] = None,
    ) -> ModelRuntime:
        """
        Create a fresh ModelRuntime for one instance of this spec.

        Loading priority:
        1. If @load registered → call it (with case_dir + entry or just case_dir)
        2. Else if @config registered and entry provided → auto-construct
        3. Else → error
        """
        if isinstance(entry, str):
            raise TypeError(
                "instantiate() no longer accepts instance_id as a string. "
                "Pass entry=dict or use the manifest loader."
            )

        runtime_name = entry["name"] if entry and "name" in entry else self.name

        if self._load_func is not None:
            config = self._load_func(case_dir, entry)
        elif self._config_class is not None and entry is not None:
            fields = {k: v for k, v in entry.items() if k not in ("type", "name")}
            config = self._config_class.model_construct(**fields)  # type: ignore[attr-defined]
        else:
            raise ValueError(
                f"ModelSpec '{self.name}': cannot instantiate. "
                "Register @load or @config with an entry dict."
            )

        return ModelRuntime(
            spec=self,
            name=runtime_name,
            config=config,
        )

    # ------------------------------------------------------------------
    # Operation building (called by ModelRuntime.operations)
    # ------------------------------------------------------------------

    def _build_operations_for(self, runtime: ModelRuntime) -> list[Operation]:  # type: ignore[override]
        """
        Build Operation objects with *runtime* as the ``self`` binding.

        Handles operation_collection dispatch and multi-instance suffix,
        then delegates to BaseSpec._build_operations_for for the core loop.
        """
        if self._operation_collection_func is not None:
            result = self._operation_collection_func(runtime)
            if isinstance(result, Operations):
                return list(result)
            return result  # type: ignore[no-any-return]

        # Suffix operation names when runtime.name differs from spec name
        # (indicates multi-instance model from manifest).
        suffix = (
            f"_{runtime.name}" if runtime.name and runtime.name != self.name else ""
        )

        return super()._build_operations_for(runtime, suffix=suffix)

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
