# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
Config injection helpers shared by ModelSpec and SolverSpec operations.

Operates on plain functions and any runtime object that exposes a
``config`` attribute — no reference to a specific Spec class.
"""

from __future__ import annotations

import inspect
from functools import wraps
from typing import Any, Callable, TYPE_CHECKING

from neofoam.io import BaseConfig

if TYPE_CHECKING:
    from neofoam.framework.context import Context


def _discover_configs_from_signature(func: Callable[..., Any]) -> list[dict[str, Any]]:
    """Return [{param_name, config_type}] for BaseConfig parameters in *func*."""
    sig = inspect.signature(func)
    discovered: list[dict[str, Any]] = []

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "ctx"):
            continue
        annotation = param.annotation
        if annotation is inspect.Parameter.empty:
            continue
        if getattr(annotation, "__origin__", None) is not None:
            continue
        try:
            if isinstance(annotation, type) and issubclass(annotation, BaseConfig):
                discovered.append({"param_name": param_name, "config_type": annotation})
        except TypeError:
            continue

    return discovered


def _find_config_by_type(runtime_config: Any, config_type: type) -> Any:
    """
    Find a config instance of *config_type* in *runtime_config*.

    Handles two cases:
    - Direct match: ``runtime_config`` is already an instance of ``config_type``.
    - Namespace: ``runtime_config`` is a SimpleNamespace whose attributes
      include an instance of ``config_type``.
    """
    if isinstance(runtime_config, config_type):
        return runtime_config
    # Search SimpleNamespace attributes
    if hasattr(runtime_config, "__dict__"):
        for val in vars(runtime_config).values():
            if isinstance(val, config_type):
                return val
    raise ValueError(
        f"No config of type {config_type.__name__} found in runtime.config "
        f"(type: {type(runtime_config).__name__})"
    )


def _create_runtime_config_wrapper(
    func: Callable[..., Any],
    discovered: list[dict[str, Any]],
    runtime: Any,
) -> Callable[["Context"], Any]:
    """
    Wrap an operation function to inject config from ``runtime.config``.

    Binding:
        self  -> runtime
        config params -> resolved from runtime.config by type
        field params  -> ctx.fields[param_name]
    """
    sig = inspect.signature(func)
    expects_self = "self" in sig.parameters
    expects_ctx = "ctx" in sig.parameters

    @wraps(func)
    def wrapper(ctx: "Context") -> Any:
        call_kwargs: dict[str, Any] = {}

        if expects_ctx:
            call_kwargs["ctx"] = ctx
        if expects_self:
            call_kwargs["self"] = runtime

        for cfg_meta in discovered:
            param_name = cfg_meta["param_name"]
            config_type = cfg_meta["config_type"]
            call_kwargs[param_name] = _find_config_by_type(runtime.config, config_type)

        for pname, param in sig.parameters.items():
            if pname in ("self", "ctx") or pname in call_kwargs:
                continue
            if (
                param.annotation in (float, int, str, bool)
                or param.annotation is inspect.Parameter.empty
            ):
                if hasattr(ctx, "fields") and pname in ctx.fields:
                    call_kwargs[pname] = ctx.fields[pname]

        result = func(**call_kwargs)

        from neofoam.framework.context import FieldUpdates

        if isinstance(result, FieldUpdates):
            ctx.fields.update(result)
            return None

        return result

    setattr(wrapper, "_already_wrapped", True)
    return wrapper
