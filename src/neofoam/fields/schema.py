# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Synthesise a per-field :class:`BaseConfig` from a :class:`FieldDecl`.

A field declaration carries everything the on-disk schema needs:

* ``dimensions`` — written verbatim into the ``dimensions`` entry.
* ``value_type`` — drives the ``FoamFile.class`` (volScalarField vs
  volVectorField) and the ``internalField`` literal shape.
* ``allowed_bcs`` — the discriminated-union arms for the
  ``boundaryField`` map.

:func:`schema_for` returns a :class:`BaseConfig` subclass bound to
``0/<decl.name>`` via the existing OpenFOAM IO strategy. The schema:

* preserves the ``FoamFile`` sub-dict so OpenFOAM still recognises the
  file's class on the next read;
* exposes ``dimensions`` as ``list[int]`` in Python while serialising
  it as OpenFOAM's bracket form (``[0 1 -1 0 0 0 0]``);
* keeps ``internalField`` as the literal string OpenFOAM reads/writes
  (``"uniform 0"`` / ``"uniform (0 0 0)"``); helpers in
  :mod:`neofoam.fields.value_types` convert to/from Python values.

Results are cached per ``FieldDecl`` identity so
``configurations(solver)`` enumeration is cheap and stable across
calls — repeated calls with the same decl object return the *same*
synthesised class.
"""

from __future__ import annotations

import re
from typing import Any, Optional, cast

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator

from neofoam.fields.bc import build_bc_union
from neofoam.fields.decl import FieldDecl
from neofoam.fields.value_types import (
    FieldValue,
    Scalar,
    Tensor,
    Vector,
    zero_uniform,
)
from neofoam.io.base import BaseConfig
from neofoam.io.decorator import IOStrategy, OF


# OpenFOAM class names per value-type marker.
_FOAM_CLASS: dict[type, str] = {
    Scalar: "volScalarField",
    Vector: "volVectorField",
    Tensor: "volTensorField",
}

_DIM_BRACKET = re.compile(r"^\s*\[\s*(.*?)\s*\]\s*$")


def _parse_dimensions(value: Any) -> list[int]:
    """Accept either an OpenFOAM bracket string or a list/tuple of ints."""
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    if isinstance(value, str):
        m = _DIM_BRACKET.match(value)
        if m is None:
            raise ValueError(f"dimensions: cannot parse {value!r}")
        return [int(part) for part in m.group(1).split()]
    raise TypeError(f"dimensions: unsupported type {type(value).__name__}")


def _format_dimensions(value: list[int]) -> str:
    """Render a dimension list as OpenFOAM's bracketed token."""
    return "[" + " ".join(str(int(v)) for v in value) + "]"


class _FoamFileHeader(BaseModel):
    """The ``FoamFile`` sub-dict OpenFOAM uses to identify a field file.

    Kept minimal — ``version`` / ``format`` default to OpenFOAM's ASCII
    norm and ``class`` / ``object`` are pinned by the field declaration.
    ``ConfigDict(extra="allow")`` keeps any additional keys present on
    disk (e.g. ``location``).
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    version: str = "2.0"
    format: str = "ascii"
    field_class: str = Field(alias="class")
    object: str


def schema_for(decl: FieldDecl) -> type[BaseConfig]:
    """Return a :class:`BaseConfig` subclass bound to ``0/<decl.name>``.

    The class is cached on ``decl._schema_cache`` keyed by ``""`` (a
    single entry per decl); repeat calls return the same class object,
    which keeps :func:`configurations(solver)` enumeration referentially
    stable.

    Args:
        decl: The field declaration to synthesise a schema for.

    Returns:
        A new :class:`BaseConfig` subclass. ``cls.io_config.file`` is
        ``"0/<name>"`` so the existing :class:`OpenFOAMStrategy`
        loader / writer round-trip drives it straight to disk.

    Raises:
        ValueError: If ``decl.allowed_bcs`` is empty (the schema has
            no way to validate ``boundaryField`` patches).
    """
    cached = decl._schema_cache.get("")
    if cached is not None:
        return cast("type[BaseConfig]", cached)

    if not decl.allowed_bcs:
        raise ValueError(
            f"schema_for({decl.name!r}): allowed_bcs is empty. "
            "Declare at least one BC arm (or include GenericBC for the open set)."
        )

    foam_class = _FOAM_CLASS.get(decl.value_type)
    if foam_class is None:
        raise TypeError(
            f"schema_for({decl.name!r}): unsupported value_type "
            f"{decl.value_type!r}; expected Scalar / Vector / Tensor."
        )

    bc_union = build_bc_union(decl.allowed_bcs)
    default_header = _FoamFileHeader(
        # populate_by_name → the Python attribute is ``field_class``.
        field_class=foam_class,
        object=decl.name,
    )
    default_internal: Optional[str] = (
        decl.initial_value
        if isinstance(decl.initial_value, str)
        else (zero_uniform(decl.value_type) if decl.initial_value is None else None)
    )
    if decl.initial_value is not None and not isinstance(decl.initial_value, str):
        # Bake the Python literal into a uniform string at registration time.
        from neofoam.fields.value_types import default_uniform

        default_internal = default_uniform(decl.value_type, decl.initial_value)

    # Build the schema by subclassing BaseConfig and stamping the fields via
    # ``__annotations__`` + class-attribute defaults; this gives us a normal
    # pydantic class (so model_dump / validators / serializers all attach
    # cleanly) without the create_model() ergonomics of having to spell out
    # every field as a (type, FieldInfo) tuple.
    # Drop the underscore between the field name and ``FieldConfig`` —
    # ``_snake_case`` inserts its own separator at every case boundary,
    # so ``U_FieldConfig`` would produce ``u__field_config`` (double
    # underscore from the existing ``_``). ``UFieldConfig`` gives the
    # clean ``u_field_config`` the agent layer + CaseSpec keying expect.
    cls_name = f"{decl.name}FieldConfig"

    # Use a closure for the dimension validator so we can keep the parser /
    # formatter as module-level helpers (testable in isolation).
    @field_validator("dimensions", mode="before")
    def _validate_dimensions(value: Any) -> list[int]:
        return _parse_dimensions(value)

    @field_serializer("dimensions", when_used="always")
    def _serialize_dimensions(self: Any, value: list[int]) -> str:  # noqa: ARG001
        return _format_dimensions(value)

    namespace: dict[str, Any] = {
        "__annotations__": {
            "FoamFile": _FoamFileHeader,
            "dimensions": list[int],
            # internalField is typed for *this* field's element kind (scalar /
            # vector), so a Vector field accepts ``[0,0,0]`` and a Scalar ``0``.
            # (runtime metaprogramming: decl.value_type is a value, not a static
            # type, so mypy can't follow the subscription)
            "internalField": FieldValue[decl.value_type],  # type: ignore[name-defined]
            # runtime metaprogramming: bc_union is a value (a built Union), not a
            # static type, so mypy can't use it as a subscript here.
            "boundaryField": dict[str, bc_union],  # type: ignore[valid-type]
        },
        "FoamFile": default_header,
        "internalField": default_internal,
        "boundaryField": Field(default_factory=dict),
        "_validate_dimensions": _validate_dimensions,
        "_serialize_dimensions": _serialize_dimensions,
        "model_config": ConfigDict(populate_by_name=True),
        # Pinning dimensions as a class-level default lets validation
        # succeed even when a YAML / dict input omits the key. The
        # declaration is the source of truth.
        "dimensions": list(decl.dimensions),
    }
    new_cls = type(cls_name, (BaseConfig,), namespace)
    # Apply IO strategy *after* class creation (the decorator mutates
    # the class object in place).
    new_cls = IOStrategy(OF(f"0/{decl.name}"))(new_cls)
    decl._schema_cache[""] = new_cls
    return cast("type[BaseConfig]", new_cls)
