# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""OpenFOAM dictionary reading and writing strategy via pybFoam."""

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


_BOOL_TRUE = frozenset({"yes", "true", "on"})
_BOOL_FALSE = frozenset({"no", "false", "off"})


def _read_bool(foam_dict: Any, key: str) -> bool:
    """Read an OpenFOAM Switch value and return a Python bool."""
    raw = str(foam_dict.get[str](key)).strip().lower()
    if raw in _BOOL_TRUE:
        return True
    if raw in _BOOL_FALSE:
        return False
    raise ValueError(f"Cannot parse '{raw}' as bool")


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


READ_DISPATCH: dict[type, Callable[[Any, str], Any]] = {
    str: lambda d, key: str(d.get[str](key)),
    int: lambda d, key: int(d.get[str](key)),
    float: lambda d, key: float(d.get[str](key)),
    bool: _read_bool,
}

WRITE_DISPATCH: dict[type, Callable[[Any, str, Any], None]] = {
    str: lambda d, key, v: d.set(key, v),
    int: lambda d, key, v: d.set(key, v),
    float: lambda d, key, v: d.set(key, v),
    bool: lambda d, key, v: d.set(key, "yes" if v else "no"),
}


class OpenFOAMStrategy(SubdictMixin):
    """Read/write OpenFOAM dictionaries with optional subdict support.

    Uses ``pybFoam.dictionary`` as backend.  Field types from the Pydantic
    model drive type-aware dispatch (see ``READ_DISPATCH`` / ``WRITE_DISPATCH``).
    Nested ``BaseModel`` sub-classes are handled recursively.
    """

    def __init__(self, subdict_path: Optional[str] = None):
        super().__init__(subdict_path)

    def read(
        self,
        model_cls: type[BaseModel],
        path: Path,
        encoding: str = "utf-8",
    ) -> dict[str, Any]:
        """Read an OpenFOAM dictionary file using *model_cls* field types.

        Args:
            model_cls: Pydantic model whose fields guide typed extraction.
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
        target_dict = (
            self._resolve_subdict(root_dict) if self.subdict_path else root_dict
        )
        return self._read_fields(model_cls, target_dict)

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
            FileNotFoundError: If subdict write is requested for a missing file.
            KeyError: If the configured subdict path does not exist in the file.
            TypeError: If a value type has no entry in ``WRITE_DISPATCH``.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        data = instance.model_dump(mode="python", exclude_none=False)

        if self.subdict_path:
            if not path.exists():
                raise FileNotFoundError(
                    f"Configuration file not found: {path} "
                    f"(required for subdict write '{self.subdict_path}')"
                )

            root_dict = pyf.dictionary.read(str(path))
            target_dict = self._resolve_subdict(root_dict)
            target_dict.clear()
            self._write_fields(target_dict, data, type(instance))
            root_dict.write(str(path))
            return

        root_dict = (
            pyf.dictionary.read(str(path)) if path.exists() else pyf.dictionary()
        )
        root_dict.clear()
        self._write_fields(root_dict, data, type(instance))
        root_dict.write(str(path))

    def _resolve_subdict(self, root_dict: Any) -> Any:
        """Resolve configured subdict path against a pybFoam dictionary."""
        if not self.path_parts:
            return root_dict

        current = root_dict
        traversed: list[str] = []
        for part in self.path_parts:
            if not current.found(part):
                full_path = ".".join(self.path_parts)
                traversed_path = ".".join(traversed) if traversed else "<root>"
                raise KeyError(
                    f"Subdict path '{full_path}' not found: "
                    f"key '{part}' does not exist at '{traversed_path}'"
                )
            if not current.isDict(part):
                full_path = ".".join(self.path_parts)
                raise KeyError(
                    f"Subdict path '{full_path}' resolved to non-dictionary entry "
                    f"at '{part}'"
                )
            traversed.append(part)
            current = current.subDict(part)
        return current

    def _read_fields(
        self, model_cls: type[BaseModel], foam_dict: Any
    ) -> dict[str, Any]:
        """Recursively read pybFoam dictionary entries guided by model fields.

        For each field in *model_cls*:
        * If the key corresponds to a sub-dictionary **and** the field type is a
          ``BaseModel`` subclass, recurse.
        * Otherwise, use ``READ_DISPATCH`` for type-aware extraction; fall back
          to a plain string when the typed getter raises (e.g. the file
          contains ``count not_an_integer;`` but the model expects ``int``).
        """
        result: dict[str, Any] = {}

        for name, field_info in model_cls.model_fields.items():
            key = str(field_info.validation_alias or field_info.alias or name)
            typ = _unwrap_type(field_info.annotation)

            # Nested BaseModel → recurse into sub-dictionary
            if (
                foam_dict.found(key)
                and foam_dict.isDict(key)
                and isinstance(typ, type)
                and issubclass(typ, BaseModel)
            ):
                result[key] = self._read_fields(typ, foam_dict.subDict(key))
                continue

            # Key not present in dictionary → skip (Pydantic handles defaults / missing)
            if not foam_dict.found(key):
                continue

            # Typed dispatch
            reader = READ_DISPATCH.get(typ)
            if reader is None:
                raise TypeError(
                    f"Unsupported field type for OpenFOAM dictionary read "
                    f"at key '{key}': {typ.__name__ if isinstance(typ, type) else typ}"
                )
            result[key] = reader(foam_dict, key)

        return result

    def _write_fields(
        self,
        foam_dict: Any,
        data: dict[str, Any],
        model_cls: type[BaseModel],
    ) -> None:
        """Recursively write Python values into a pybFoam dictionary.

        Uses ``WRITE_DISPATCH`` for scalars and recurses for nested
        ``BaseModel`` sub-classes.
        """
        for name, field_info in model_cls.model_fields.items():
            key = str(field_info.validation_alias or field_info.alias or name)
            if key not in data:
                continue

            value = data[key]
            typ = _unwrap_type(field_info.annotation)

            # Nested BaseModel → recurse
            if isinstance(typ, type) and issubclass(typ, BaseModel):
                if not isinstance(value, dict):
                    raise TypeError(
                        f"Expected dict for nested model field '{key}', "
                        f"got {type(value).__name__}"
                    )
                if not foam_dict.found(key) or not foam_dict.isDict(key):
                    raise KeyError(
                        f"Cannot create nested dictionary '{key}' via pybFoam "
                        "bindings. Pre-create subdictionaries in file before "
                        "writing."
                    )
                sub = foam_dict.subDict(key)
                sub.clear()
                self._write_fields(sub, value, typ)
                continue

            # Scalar dispatch
            writer = WRITE_DISPATCH.get(type(value))
            if writer is None:
                raise TypeError(
                    f"Unsupported value type for OpenFOAM dictionary write "
                    f"at key '{key}': {type(value).__name__}"
                )

            writer(foam_dict, key, value)
