# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Value-type markers for declared fields + uniform-literal helpers.

A field declaration carries a ``value_type`` — :class:`Scalar` or
:class:`Vector` — that drives:

* The Python form of ``internalField`` (a ``float`` for scalar, a
  three-tuple of floats for vector).
* The OpenFOAM uniform-literal form (``"uniform 0"`` /
  ``"uniform (0 0 0)"``) that the OpenFOAM strategy reads and writes.

These markers are *not* pydantic field types. They parameterise the
synthesised schema (Step 4) so a single declaration line — ``value_type=
Vector`` — pins both the Python and on-disk shapes consistently. The
discriminated-union BC arms in :mod:`neofoam.fields.bc` reuse the same
``parse_uniform`` / ``default_uniform`` helpers when their ``value``
crosses the disk boundary.
"""

from __future__ import annotations

import re
from typing import Annotated, Any, Final, Generic, TypeVar, Union

from pydantic import BaseModel, SerializationInfo, WrapSerializer


class Scalar:
    """Marker: the field holds one ``float`` per cell.

    ``internalField`` is ``float`` in Python; on disk it is
    ``"uniform <number>"``.
    """

    py_type: Final = float
    n_components: Final = 1


class Vector:
    """Marker: the field holds a 3-vector per cell.

    ``internalField`` is ``tuple[float, float, float]`` in Python; on
    disk it is ``"uniform (<x> <y> <z>)"``.
    """

    py_type: Final = tuple
    n_components: Final = 3


# Tensor placeholder reserved for a later expansion (volTensorField,
# volSymmTensorField); not used by Phase 1's in-tree fields.
class Tensor:
    """Marker reserved for tensor fields. Not exercised in Phase 1."""

    py_type: Final = tuple
    n_components: Final = 9


_VECTOR_LITERAL = re.compile(
    r"^\s*uniform\s*\(\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+"
    r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+"
    r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\)\s*$"
)
_SCALAR_LITERAL = re.compile(r"^\s*uniform\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$")


def default_uniform(value_type: type, value: Any) -> str:
    """Serialise a Python ``value`` into an OpenFOAM uniform literal.

    Accepts either the bare Python form (``0`` / ``[0, 0, 0]``) or an
    already-formatted literal — the latter is returned untouched so
    callers can keep the on-disk text intact when no normalisation is
    needed.

    Args:
        value_type: :class:`Scalar` or :class:`Vector`.
        value: Either ``float`` / ``tuple`` / ``list``, or a string
            already shaped as ``"uniform …"``.

    Returns:
        An OpenFOAM uniform-literal string.

    Raises:
        TypeError: For an unsupported value type / value shape.
    """
    if isinstance(value, str):
        return value
    if value_type is Scalar:
        return f"uniform {float(value)}"
    if value_type is Vector:
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise TypeError(
                f"default_uniform(Vector, ...) expects a 3-sequence; got {value!r}"
            )
        x, y, z = (float(v) for v in value)
        return f"uniform ({x} {y} {z})"
    raise TypeError(f"default_uniform: unsupported value_type {value_type!r}")


def parse_uniform(value_type: type, literal: str) -> Any:
    """Parse an OpenFOAM uniform literal into the Python ``value_type`` form.

    Args:
        value_type: :class:`Scalar` or :class:`Vector`.
        literal: The on-disk string (``"uniform 0"`` /
            ``"uniform (0 0 0)"``).

    Returns:
        ``float`` for :class:`Scalar`, ``tuple[float, float, float]``
        for :class:`Vector`.

    Raises:
        ValueError: If ``literal`` does not parse for the given type.
    """
    if value_type is Scalar:
        m = _SCALAR_LITERAL.match(literal)
        if not m:
            raise ValueError(f"parse_uniform(Scalar, …): cannot parse {literal!r}")
        return float(m.group(1))
    if value_type is Vector:
        m = _VECTOR_LITERAL.match(literal)
        if not m:
            raise ValueError(f"parse_uniform(Vector, …): cannot parse {literal!r}")
        return (float(m.group(1)), float(m.group(2)), float(m.group(3)))
    raise TypeError(f"parse_uniform: unsupported value_type {value_type!r}")


def zero_uniform(value_type: type) -> str:
    """The all-zeros uniform literal for ``value_type``.

    Convenience for the scaffolding path (default ``internalField`` when
    a field declaration does not pin one).
    """
    if value_type is Scalar:
        return "uniform 0"
    if value_type is Vector:
        return "uniform (0 0 0)"
    raise TypeError(f"zero_uniform: unsupported value_type {value_type!r}")


# ---------------------------------------------------------------------------
# FieldValue: one type for every field-value slot (BC value / inletValue,
# internalField, GenericBC extras). It carries the *meaning* of the value
# (uniform scalar/vector, or non-uniform per-element list, or an already
# on-disk literal string); how it is *encoded* is decided per writer by the
# format in the serialization context — so JSON/YAML keep raw structured data
# and only OpenFOAM emits ``uniform`` / ``nonuniform List<…>`` text. This
# replaces the ad-hoc ``normalize_field_bcs`` write-boundary pass.
# ---------------------------------------------------------------------------


class NonUniform(BaseModel):
    """Per-element field data: a list of scalars or a list of 3-vectors.

    Tagged explicitly (rather than inferred from shape) so a bare ``[1, 0, 0]``
    stays an unambiguous *uniform vector* while ``NonUniform(nonuniform=[...])``
    is the *non-uniform* field.
    """

    nonuniform: list[Union[float, list[float]]]


def to_uniform_literal(value: Any) -> Any:
    """Render a uniform field value as an OpenFOAM literal.

    ``float`` → ``"uniform <n>"``, 3-sequence → ``"uniform (x y z)"``; strings
    (already on-disk literals) and other types pass through unchanged.
    """
    if isinstance(value, bool) or isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return f"uniform {float(value)}"
    if isinstance(value, (list, tuple)):
        return "uniform (" + " ".join(str(float(x)) for x in value) + ")"
    return value


def to_nonuniform_literal(items: list[Any]) -> str:
    """Render per-element data as an OpenFOAM ``nonuniform List<…>`` literal."""
    if items and isinstance(items[0], (list, tuple)):
        kind = "vector"
        body = " ".join("(" + " ".join(str(float(x)) for x in v) + ")" for v in items)
    else:
        kind = "scalar"
        body = " ".join(str(float(x)) for x in items)
    return f"nonuniform List<{kind}> {len(items)} ({body})"


def _serialize_field_value(value: Any, handler: Any, info: SerializationInfo) -> Any:
    """Per-writer mapping for :data:`FieldValue`.

    Only ``format == "openfoam"`` in the serialization context produces OpenFOAM
    text; every other writer (JSON, YAML, plain ``model_dump`` for forms / tests)
    gets the raw structured value via ``handler``.
    """
    if (info.context or {}).get("format") != "openfoam":
        return handler(value)
    if isinstance(value, NonUniform):
        return to_nonuniform_literal(value.nonuniform)
    return to_uniform_literal(value)


_ElementT = TypeVar("_ElementT")


class FieldValue(Generic[_ElementT]):
    """Generic field-value type — parameterise with the element marker.

    ``FieldValue[Scalar]`` accepts ``float`` (uniform) / ``"uniform 0"`` /
    :class:`NonUniform`; ``FieldValue[Vector]`` accepts ``list[float]`` (uniform)
    / literal / :class:`NonUniform`; ``FieldValue[Any]`` accepts either (used by
    BC arms shared across scalar and vector fields). All variants share the
    per-writer serializer: raw structured data for forms / JSON / YAML, OpenFOAM
    ``uniform`` / ``nonuniform`` text under ``context={"format": "openfoam"}``.

    It is not instantiated; subscription returns the corresponding
    ``Annotated[Union[...], WrapSerializer]`` for use as a Pydantic field type.
    """

    def __class_getitem__(cls, element: Any) -> Any:
        if element is Scalar:
            members: Any = Union[float, str, NonUniform]
        elif element is Vector:
            members = Union[list[float], str, NonUniform]
        else:  # Any / Tensor / unknown → accept both scalar and vector forms
            members = Union[float, list[float], str, NonUniform]
        return Annotated[members, WrapSerializer(_serialize_field_value)]
