# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Contribution parameter resolution, shared by every hook dispatch.

A model-owned interface (``@<model>.interface``) *is* an extension
:class:`~neofoam.framework.model.extension.Hook` on a model-private extension —
see :meth:`ModelSpec.interface <neofoam.framework.model.spec.ModelSpec.interface>`.
This module holds the one piece both spellings share beyond the ``Hook`` itself:
resolving a contribution's parameters against its own model runtime plus the
live ``Context``.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, Callable, TypeVar

if TYPE_CHECKING:
    from .runtime import ModelRuntime

T = TypeVar("T")


def _resolve_contribution_kwargs(
    interface_name: str,
    func: Callable[..., T],
    runtime: ModelRuntime,
    ctx: Any,
    call_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve *func*'s parameters the same way ``@model.operation`` does.

    *call_kwargs* (a hook's call-time arguments, named against the hook
    declaration) pre-resolve any parameter of the same name. Config-typed
    params (``BaseConfig`` subclasses) are pulled from the *contributing*
    runtime's own ``config`` by type via ``config_injection``; the remainder is
    resolved by the shared ``DependencyResolver`` (``ctx.fields`` by name,
    ``Depends`` markers). A parameter that resolves to neither raises a
    ``ValueError`` naming the hook, the contribution, and the parameter.
    """
    # Lazy imports break the cycle interface -> config_injection/dependency_resolver
    # -> (resolver) -> model.interface.
    from neofoam.framework.config_injection import (  # noqa: PLC0415
        _discover_configs_from_signature,
        _find_config_by_type,
    )
    from neofoam.framework.dependency_resolver import (  # noqa: PLC0415
        DependencyResolver,
    )

    preresolved: dict[str, Any] = {}
    if call_kwargs:
        func_params = inspect.signature(func).parameters
        preresolved.update({k: v for k, v in call_kwargs.items() if k in func_params})
    for meta in _discover_configs_from_signature(func):
        pname = meta["param_name"]
        try:
            preresolved[pname] = _find_config_by_type(runtime.config, meta["config_type"])
        except ValueError as exc:
            raise ValueError(
                f"interface '{interface_name}': contribution '{func.__name__}' "
                f"(model '{runtime.spec.name}') could not resolve config "
                f"parameter '{pname}': {exc}"
            ) from exc

    resolved = DependencyResolver().resolve_arguments(func, ctx=ctx, **preresolved)

    sig = inspect.signature(func)
    kwargs: dict[str, Any] = {}
    for pname in sig.parameters:
        if pname in ("self", "cls"):
            kwargs[pname] = None  # operation-method form: skip the receiver
            continue
        if pname not in resolved:
            available = sorted(ctx.fields) if ctx is not None else []
            raise ValueError(
                f"interface '{interface_name}': contribution '{func.__name__}' "
                f"(model '{runtime.spec.name}') declares parameter '{pname}' but no "
                f"provider supplies it. Available context fields: {available}"
            )
        kwargs[pname] = resolved[pname]
    return kwargs
