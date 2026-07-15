# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Variant-payload validation for a parameter sweep.

Each sweep dimension validates its variants differently:

* a *regular* dimension → each variant payload against the dimension's config class;
* the reserved ``mesh`` dimension → config-name-keyed inner payloads, each against its
  own class;
* the reserved ``cad`` dimension → a plain ``{alias: number}`` map, numeric shape only.

That "what a valid variant looks like" rule lives in **one** place, :func:`_variant_error`.
Everything else is a thin wrapper over it: the non-throwing :func:`variant_errors` (used
to badge the canvas live) collects its messages, and the throwing :func:`validate_dimensions`
/ :func:`validate_mesh_dimension` / :func:`validate_cad_dimension` raise the first one.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ValidationError

from neofoam.tooling.workflow.rules import CAD_DIM, MESH_DIM


def _short_validation_error(exc: Exception) -> str:
    """A one-line, user-facing message from a pydantic ``ValidationError``.

    Takes the first error's location + message (e.g. ``nu: Input should be
    greater than 0``); falls back to the exception text for non-pydantic errors.
    """
    if isinstance(exc, ValidationError):
        detail = exc.errors()
        if detail:
            first = detail[0]
            loc = ".".join(str(p) for p in first.get("loc", ())) or "value"
            return f"{loc}: {first.get('msg', 'invalid')}"
    return str(exc).splitlines()[0] if str(exc) else exc.__class__.__name__


def _variant_error(
    dim: str,
    payload: Any,
    classes: Mapping[str, type[BaseModel]],
) -> str | None:
    """Validate one variant payload; return a short message or ``None`` if valid.

    The single shape-checker for every dimension kind — the mesh/cad/regular rules
    live here and nowhere else. The throwing validators and :func:`variant_errors`
    both delegate to it.
    """
    if dim == MESH_DIM:
        if not isinstance(payload, Mapping):
            return "must map config names to payloads"
        for config_name, inner in payload.items():
            cls = classes.get(str(config_name))
            if cls is None:
                return f"unknown config '{config_name}'"
            try:
                cls.model_validate(dict(inner))
            except Exception as exc:
                return f"{config_name}: {_short_validation_error(exc)}"
        return None
    if dim == CAD_DIM:
        if not isinstance(payload, Mapping):
            return "must map parameter aliases to numbers"
        for alias, value in payload.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return f"{alias}: must be a number, got {type(value).__name__}"
        return None
    cls = classes.get(dim)
    if cls is None:
        return f"unknown dimension '{dim}'"
    try:
        cls.model_validate(dict(payload))
    except Exception as exc:
        return _short_validation_error(exc)
    return None


def _raise_variant_errors(
    dimensions: Mapping[str, Mapping[str, Any]],
    classes: Mapping[str, type[BaseModel]],
) -> None:
    """Raise ``ValueError`` on the first invalid variant — the throwing form.

    A thin ``raise`` wrapper over :func:`_variant_error`, naming the offending
    ``dim.variant`` so callers get the same "which variant, and why" report.
    """
    for dim, variants in dimensions.items():
        for name, payload in variants.items():
            msg = _variant_error(dim, payload, classes)
            if msg is not None:
                raise ValueError(f"variant '{dim}.{name}' failed validation: {msg}")


def validate_dimensions(
    dimensions: Mapping[str, Mapping[str, Mapping[str, Any]]],
    classes: Mapping[str, type[BaseModel]],
) -> None:
    """Validate every variant payload against its dimension's config class.

    Payloads are validated, not round-tripped: several configs materialize their
    content in a wrap serializer (``model_dump()`` of the defaults is ``{}``), so
    dumping the validated instance would lose the form values. The runner re-validates
    (and thereby coerces) payloads at apply time.

    Raises:
        ValueError: On an unknown dimension or a failing variant, naming ``dim.variant``.
    """
    _raise_variant_errors(dimensions, classes)


def validate_mesh_dimension(
    variants: Mapping[str, Mapping[str, Any]],
    classes: Mapping[str, type[BaseModel]],
) -> None:
    """Validate the ``mesh`` dimension's config-name-keyed variant payloads.

    Each mesh variant holds ``{config_name: payload}`` (e.g.
    ``{"block_mesh_dict_config": {...}}``); every inner payload is validated against
    its config class. Empty variants are allowed (the base dicts run unchanged).

    Raises:
        ValueError: Naming the offending ``mesh.variant`` (and inner config).
    """
    _raise_variant_errors({MESH_DIM: variants}, classes)


def validate_cad_dimension(variants: Mapping[str, Any]) -> None:
    """Validate the ``cad`` dimension's numeric parameter-map variants.

    A CAD variant is a plain ``{alias: number}`` map (the parametric model's driven
    dimensions); there is no config class to validate against, only the numeric shape.

    Raises:
        ValueError: Naming the offending ``cad.variant`` when it is not a mapping or
            carries a non-numeric value.
    """
    _raise_variant_errors({CAD_DIM: variants}, {})


def variant_errors(
    dimensions: Mapping[str, Mapping[str, Mapping[str, Any]]],
    classes: Mapping[str, type[BaseModel]],
) -> dict[str, dict[str, str]]:
    """Per-variant validation errors, ``{dim: {variant: message}}`` — never raises.

    The non-throwing counterpart of the ``validate_*`` functions (all four share
    :func:`_variant_error`): only failing variants appear, so it is safe to badge the
    canvas live. Covers mesh and cad dimensions the same way the throwing validators do.
    """
    errors: dict[str, dict[str, str]] = {}
    for dim, variants in dimensions.items():
        for name, payload in variants.items():
            msg = _variant_error(dim, payload, classes)
            if msg is not None:
                errors.setdefault(dim, {})[name] = msg
    return errors
