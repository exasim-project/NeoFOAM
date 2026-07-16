# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""OpenFOAM dictionary reading and writing strategy via pybFoam.

Read path: convert the pybFoam dictionary to a plain ``dict`` of strings and
nested dicts, then coerce primitive scalars (``int``, ``float``, ``bool``)
to the types the Pydantic model declares via :data:`READ_DISPATCH` — so
callers using ``BaseConfig.load(validate=False)`` see typed values
(matching the JSON/YAML strategies, which get them for free from their
native formats). Failing coercions (``count not_an_integer;`` for a field
typed as ``int``) raise ``ValueError`` eagerly with file context, ahead of
Pydantic validation. Discriminated-union schemes (``DivScheme = Annotated[
Union[...], BeforeValidator(...)]``) parse themselves from the raw string,
and ``ConfigDict(extra="allow")`` extras pass through unchanged (no schema
to guide their type).

Write path: ``model_dump(by_alias=True)`` produces a ``dict`` tree which one
walker pushes into the pybFoam dictionary, dispatching scalars through
:data:`WRITE_DISPATCH` (``bool`` → ``yes``/``no``, OpenFOAM's Switch
encoding).
"""

from __future__ import annotations

from pathlib import Path
from typing import (
    Annotated,
    Any,
    Callable,
    Literal,
    Optional,
    Union,
    get_args,
    get_origin,
)

from pydantic import BaseModel

_BOOL_TRUE = frozenset({"yes", "true", "on", "1"})
_BOOL_FALSE = frozenset({"no", "false", "off", "0"})


def _read_bool(raw: str) -> bool:
    """Parse an OpenFOAM Switch token into a Python ``bool``."""
    low = raw.strip().lower()
    if low in _BOOL_TRUE:
        return True
    if low in _BOOL_FALSE:
        return False
    raise ValueError(f"Cannot parse '{raw}' as bool")


READ_DISPATCH: dict[type, Callable[[str], Any]] = {
    str: lambda v: v,
    int: int,
    float: float,
    bool: _read_bool,
}

WRITE_DISPATCH: dict[type, Callable[[Any, str, Any], None]] = {
    str: lambda d, key, v: d.set(key, v),
    int: lambda d, key, v: d.set(key, v),
    float: lambda d, key, v: d.set(key, v),
    bool: lambda d, key, v: d.set(key, "yes" if v else "no"),
}


def foam_header(path: Path) -> dict[str, str]:
    """A minimal ``FoamFile`` header for a plain OpenFOAM dictionary file.

    Field configs synthesise their own header (``class`` = ``volVectorField``
    etc., see :mod:`neofoam.fields.schema`); generic dict configs
    (``controlDict``, ``fvSchemes``, ``transportProperties``, …) carry no
    header field, so the writer injects this one — ``class dictionary`` with
    ``object`` taken from the file name — making a from-scratch case readable
    by OpenFOAM without copying a template.
    """
    return {
        "version": "2.0",
        "format": "ascii",
        "class": "dictionary",
        "object": path.stem,
    }


def _to_python(foam_dict: Any) -> dict[str, Any]:
    """Convert a pybFoam dictionary into a plain ``dict`` of strings/sub-dicts."""
    out: dict[str, Any] = {}
    for entry in foam_dict.toc():
        key = str(entry)
        if foam_dict.isDict(key):
            out[key] = _to_python(foam_dict.subDict(key))
        else:
            out[key] = str(foam_dict.get[str](key))
    return out


def _unwrap_type(tp: Any) -> Any:
    """Peel ``Annotated``, ``Optional``, ``Union``, and ``Literal`` wrappers."""
    if get_origin(tp) is Annotated:
        tp = get_args(tp)[0]
    if get_origin(tp) in (Optional, Union):
        args = [a for a in get_args(tp) if a is not type(None)]
        # A union that accepts ``str`` can hold the raw OpenFOAM text (e.g.
        # ``FieldValue`` = ``float | list | str | NonUniform``, whose on-disk form
        # is ``"uniform (0 0 0)"``); leave it uncoerced and let Pydantic validate.
        if str in args:
            return str
        if args:
            tp = args[0]
    if get_origin(tp) is Literal:
        return str
    return tp


def _read_fields(model_cls: type[BaseModel], data: dict[str, Any]) -> dict[str, Any]:
    """Walk the model schema; coerce string scalars via :data:`READ_DISPATCH`.

    OpenFOAM is a textual format — every leaf comes back as a string. The
    dispatch table maps a declared primitive type to its parser; a missing
    entry leaves the value untouched (e.g. ``dict[str, Any]`` fields, raw
    sub-dicts for downstream `BeforeValidator` parsing). Nested
    ``BaseModel`` fields recurse; ``extra="allow"`` extras pass through
    unchanged.
    """
    out: dict[str, Any] = {}
    for name, field_info in model_cls.model_fields.items():
        key = str(field_info.validation_alias or field_info.alias or name)
        if key not in data:
            continue
        value = data[key]
        typ = _unwrap_type(field_info.annotation)
        if (
            isinstance(value, dict)
            and isinstance(typ, type)
            and issubclass(typ, BaseModel)
        ):
            out[key] = _read_fields(typ, value)
            continue
        reader = READ_DISPATCH.get(typ) if isinstance(value, str) else None
        out[key] = reader(value) if reader is not None else value
    # Extras under ``ConfigDict(extra="allow")`` pass through untouched.
    for k, v in data.items():
        if k not in out:
            out[k] = v
    return out


class OpenFOAMStrategy:
    """Format marker for OpenFOAM dictionary files.

    Reads and writes go through :class:`neofoam.io.DictFile`; this class survives
    as the format tag stored in ``IOMetadata`` (carrying the optional
    ``subdict_path``) and hosts :meth:`_write`, the ``dict`` → pybFoam-dictionary
    walker that ``DictFile`` reuses.
    """

    def __init__(self, subdict_path: Optional[str] = None):
        self.subdict_path = subdict_path

    def _write(self, foam_dict: Any, data: dict[str, Any]) -> None:
        """Push a Pydantic-dumped ``dict`` into a pybFoam dictionary.

        Dispatch is on the runtime type of each value: ``dict`` becomes a
        sub-dict (recurse); scalars route through :data:`WRITE_DISPATCH`,
        which encodes ``bool`` as OpenFOAM's Switch (``yes``/``no``) and
        forwards the rest to ``set`` for pybFoam to stringify. Values whose
        type isn't in the table raise ``TypeError`` — explicit beats silent
        coercion. ``None`` values are dropped (unset Optional fields).

        ``model_dump`` iterates declared fields in declaration order and
        appends ``extra="allow"`` extras at the end, so on-disk ordering
        matches the Python model.
        """
        for key, value in data.items():
            if value is None:
                continue
            if isinstance(value, dict):
                # ``subDictOrAdd`` creates the sub-dict if missing but returns
                # a *detached* reference; re-fetch via ``subDict`` to get a
                # live handle whose mutations propagate to the parent.
                foam_dict.subDictOrAdd(key)
                sub = foam_dict.subDict(key)
                sub.clear()
                self._write(sub, value)
                continue
            writer = WRITE_DISPATCH.get(type(value))
            if writer is None:
                raise TypeError(
                    f"Unsupported value type for OpenFOAM dictionary write "
                    f"at key '{key}': {type(value).__name__}"
                )
            writer(foam_dict, key, value)
