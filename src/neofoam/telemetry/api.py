# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The telemetry shim: importable and near-zero-cost without OpenTelemetry.

Framework code wraps execution in :func:`span`; operation authors open
nested spans (or use :func:`instrument`) inside their operations for finer
detail. Everything is a no-op until :func:`configure` activates tracing,
which requires the optional ``neofoam[telemetry]`` extra.
"""

from __future__ import annotations

import importlib
from contextlib import AbstractContextManager, nullcontext
from functools import wraps
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Optional, TypeVar, cast

from .settings import MpiInfo, TelemetrySettings

F = TypeVar("F", bound=Callable[..., Any])


class TelemetryNotInstalledError(RuntimeError):
    """Telemetry was enabled but the optional OpenTelemetry extra is missing."""


# The live ActiveTelemetry instance (from ._sdk) — None while inactive.
_active: Optional[Any] = None


def is_active() -> bool:
    """Whether tracing is currently configured (the fast-path guard)."""
    return _active is not None


def _load_sdk() -> ModuleType:
    try:
        return importlib.import_module("neofoam.telemetry._sdk")
    except ImportError as exc:
        raise TelemetryNotInstalledError(
            "Telemetry is enabled but the OpenTelemetry packages are not "
            "installed. Install the optional extra: pip install neofoam[telemetry]"
        ) from exc


def configure(
    settings: Optional[TelemetrySettings] = None,
    *,
    case_dir: str | Path,
    mpi: Optional[MpiInfo] = None,
) -> None:
    """Activate tracing for a run rooted at ``case_dir``.

    ``mpi=None`` resolves the rank from ``pybFoam.Pstream`` lazily at the
    first span export — MPI is typically initialized *after* configure
    (by the solver's ``argList``). Disabled settings deactivate. Calling
    while active flushes the previous run and replaces it.
    """
    global _active
    resolved = settings if settings is not None else TelemetrySettings()
    if not resolved.enabled:
        shutdown()
        return
    sdk = _load_sdk()
    shutdown()
    _active = sdk.start(resolved, Path(case_dir), mpi)


def shutdown() -> None:
    """Flush spans, write the summary, and deactivate. Idempotent."""
    global _active
    if _active is not None:
        _active.shutdown()
        _active = None


def span(name: str, **attributes: Any) -> AbstractContextManager[Any]:
    """A tracing span context manager; a shared no-op while inactive.

    ``None``-valued attributes are dropped so callers can pass optional
    metadata unconditionally.
    """
    if _active is None:
        return nullcontext()
    attrs = {key: value for key, value in attributes.items() if value is not None}
    return cast(
        AbstractContextManager[Any],
        _active.tracer.start_as_current_span(name, attributes=attrs),
    )


def instrument(name: Optional[str] = None, **attributes: Any) -> Callable[[F], F]:
    """Decorate a callable so each call runs inside a span.

    The span is named ``name`` (default: the function's qualified name) and
    nests under whatever span is current at call time — e.g. the framework's
    operation span.
    """

    def decorator(func: F) -> F:
        span_name = name if name is not None else func.__qualname__

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if _active is None:
                return func(*args, **kwargs)
            with span(span_name, **attributes):
                return func(*args, **kwargs)

        return cast(F, wrapper)

    return decorator
