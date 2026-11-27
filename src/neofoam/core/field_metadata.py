# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Field metadata the case wizard reads from pydantic schemas."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import Field


def advanced(
    default: Any = None,
    *,
    group: Optional[str] = None,
    default_hint: Any = None,
    **field_kwargs: Any,
) -> Any:
    """A pydantic field the wizard shows under "All entries"; adds schema metadata only.

    Use it for options most cases leave unset, on a model field or wrapped around a key
    or type in ``fvSchemes.add`` / ``fvSolution.add_controls``. ``default_hint`` is a
    placeholder and is never written to the file. Stateless and thread-safe.

    Example::

        krylov_dim: Optional[int] = advanced(None, default_hint=100)
    """
    extra: dict[str, Any] = {"nfLevel": "advanced"}
    if group is not None:
        extra["nfGroup"] = group
    if default_hint is not None:
        extra["nfDefaultHint"] = default_hint
    return Field(default, json_schema_extra=extra, **field_kwargs)
