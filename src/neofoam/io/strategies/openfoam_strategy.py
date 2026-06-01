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

import pybFoam as pyf
from pydantic import BaseModel

from neofoam.io.strategies.subdict import SubdictMixin

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


class OpenFOAMStrategy(SubdictMixin):
    """Read/write OpenFOAM dictionaries with optional subdict support.

    Uses ``pybFoam.dictionary`` as the backend. Reads return a Python ``dict``
    with primitive scalars coerced to the model's declared types; writes walk
    the dumped Pydantic dict and push it back through the pybFoam API.
    """

    def __init__(self, subdict_path: Optional[str] = None):
        super().__init__(subdict_path)

    def read(
        self,
        model_cls: type[BaseModel],
        path: Path,
        encoding: str = "utf-8",
    ) -> dict[str, Any]:
        """Read an OpenFOAM dictionary file into a coerced ``dict``.

        Args:
            model_cls: Pydantic model whose field types drive primitive
                coercion (string → int/float/bool). Required so that
                ``validate=False`` callers receive typed values.
            path: Path to the OpenFOAM dictionary file.
            encoding: Unused (kept for strategy protocol compatibility).

        Returns:
            Mapping of field-name → parsed value (full file or subdict).

        Raises:
            FileNotFoundError: If *path* does not exist.
            KeyError: If the configured subdict path cannot be resolved.
        """
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        root_dict = pyf.dictionary.read(str(path))
        data = self._extract_subdict(_to_python(root_dict))
        return _read_fields(model_cls, data)

    def write(
        self,
        instance: BaseModel,
        path: Path,
        encoding: str = "utf-8",
    ) -> None:
        """Write a Pydantic model instance to an OpenFOAM dictionary file.

        Args:
            instance: The model instance to persist.
            path: Path to the OpenFOAM dictionary file.
            encoding: Unused (kept for strategy protocol compatibility).

        Raises:
            FileNotFoundError: If subdict write is requested for a missing file
                (OpenFOAM dictionaries need their ``FoamFile`` header, so we
                refuse to synthesise one — unlike JSON/YAML).
            KeyError: If the configured subdict path does not exist in the file.
        """
        # ``by_alias=True`` so synthesised sections (``fv_configs._rebuild_sections``
        # registers OpenFOAM keys like ``"div(phi,U)"`` as the ``Field.alias`` on a
        # sanitised attribute) emit data keyed by the alias the writer looks up.
        data = instance.model_dump(mode="python", exclude_none=False, by_alias=True)
        path.parent.mkdir(parents=True, exist_ok=True)

        if self.subdict_path:
            if not path.exists():
                raise FileNotFoundError(
                    f"Configuration file not found: {path} "
                    f"(required for subdict write '{self.subdict_path}')"
                )
            root_dict = pyf.dictionary.read(str(path))
            target = self._resolve_subdict(root_dict)
            target.clear()
            self._write(target, data)
            root_dict.write(str(path))
            return

        root_dict = (
            pyf.dictionary.read(str(path)) if path.exists() else pyf.dictionary()
        )
        root_dict.clear()
        self._write(root_dict, data)
        root_dict.write(str(path))

    def _resolve_subdict(self, root_dict: Any) -> Any:
        """Navigate to the configured subdict in a live pybFoam dictionary.

        Used only by :meth:`write`; reads go through :meth:`_to_python` first
        and reuse :meth:`SubdictMixin._extract_subdict` on a plain ``dict``.
        """
        if not self.path_parts:
            return root_dict

        current = root_dict
        traversed: list[str] = []
        for part in self.path_parts:
            if not current.found(part) or not current.isDict(part):
                full_path = ".".join(self.path_parts)
                traversed_path = ".".join(traversed) if traversed else "<root>"
                raise KeyError(
                    f"Subdict path '{full_path}' not found: "
                    f"key '{part}' does not exist (or is not a sub-dictionary) "
                    f"at '{traversed_path}'"
                )
            traversed.append(part)
            current = current.subDict(part)
        return current

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
