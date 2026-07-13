# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""``DictFile`` -- a format-agnostic handle mirroring the OpenFOAM dictionary API.

The format (and thus backend) is inferred from the file name: ``.json`` / ``.yaml``
/ ``.yml`` are worked on as a nested Python mapping, everything else (including
suffix-less ``controlDict`` / ``fvSolution``) as a live ``pybFoam.dictionary``.
Both backends present the identical interface -- ``get[T]``, ``set``, ``remove``,
``subDict``, ``found``, ``write`` -- so calling code is oblivious to the on-disk
format. Entries are addressed by a bare string (top level) or a tuple of keys
(nested); ``subDict`` returns a live view whose edits are persisted by the
parent's ``write``.

The pydantic bridge is deliberately *not* folded into ``get`` (that stays a typed
leaf accessor): :meth:`DictFile.fill` reads a validated config out of the file and
:meth:`DictFile.save` writes one to a fresh file. Both honour the config's declared
``subdict`` and reuse the OpenFOAM strategy's coercion so ``bool`` <-> ``yes``/``no``
and field-value encodings round-trip.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Protocol, TypeVar, Union, cast

import pybFoam as pyf
import yaml

from neofoam.io.strategies.openfoam_strategy import (
    WRITE_DISPATCH,
    OpenFOAMStrategy,
    _read_fields,
    _to_python,
    foam_header,
)

if TYPE_CHECKING:
    # Annotation-only: importing BaseConfig at runtime would close a base <-> dictfile
    # cycle (base.load/save import DictFile). The isinstance dispatch below uses
    # pydantic's BaseModel instead, and the TypeVar bound is a forward reference.
    from neofoam.io.base import BaseConfig

T = TypeVar("T")
TModel = TypeVar("TModel", bound="BaseConfig")

#: An entry address: a bare key (top level) or a tuple of keys (nested).
Key = Union[str, tuple[str, ...]]

# Reused purely for its recursive ``_write`` (dict -> pybFoam dictionary) walker.
_OF_STRATEGY = OpenFOAMStrategy()


def _fmt(path: Path) -> str:
    """The backend format implied by *path*'s suffix (OpenFOAM is the default)."""
    suffix = path.suffix.lower()
    if suffix == ".json":
        return "json"
    if suffix in (".yaml", ".yml"):
        return "yaml"
    return "openfoam"


def _load_map(path: Path, fmt: str) -> dict[str, Any]:
    raw = (
        json.loads(path.read_text())
        if fmt == "json"
        else yaml.safe_load(path.read_text())
    )
    data: dict[str, Any] = raw or {}
    return data


def _dump_map(path: Path, data: Mapping[str, Any], fmt: str) -> None:
    if fmt == "json":
        path.write_text(json.dumps(data, indent=2) + "\n")
    else:
        path.write_text(
            yaml.safe_dump(dict(data), default_flow_style=False, sort_keys=False)
        )


def _dump_model(instance: BaseConfig, fmt: str) -> dict[str, Any]:
    """Dump a config to a plain dict, OpenFOAM-encoding field values when needed."""
    if fmt == "openfoam":
        return instance.model_dump(
            mode="python",
            exclude_none=False,
            by_alias=True,
            context={"format": "openfoam"},
        )
    return instance.model_dump(mode="python", exclude_none=False, by_alias=True)


def _as_tuple(key: Key) -> tuple[str, ...]:
    return (key,) if isinstance(key, str) else tuple(key)


def _plainify(value: Any) -> Any:
    """Deep-copy nested mappings into plain ``dict`` (for the JSON/YAML backend)."""
    if isinstance(value, Mapping):
        return {k: _plainify(v) for k, v in value.items()}
    return value


# --------------------------------------------------------------------------- #
# Backends: one live node within a file, plus the root/path needed to persist. #
# --------------------------------------------------------------------------- #


class _Backend(Protocol):
    root: Any
    node: Any
    path: Path

    def found(self, key: str) -> bool: ...
    def has_dict(self, key: str) -> bool: ...
    def get(self, key: str, typ: type) -> Any: ...
    def set(self, key: str, value: object) -> None: ...
    def overwrite(self, mapping: Mapping[str, Any]) -> None: ...
    def remove(self, key: str) -> None: ...
    def child(self, key: str) -> "_Backend": ...
    def child_or_add(self, key: str) -> "_Backend": ...
    def read_fields(self, model_cls: type[BaseConfig]) -> dict[str, Any]: ...
    def write(self, path: Path) -> None: ...


class _OFBackend(_Backend):
    """Live ``pybFoam.dictionary`` node -- surgical edits, verbatim untouched."""

    def __init__(self, root: Any, node: Any, path: Path) -> None:
        self.root = root
        self.node = node
        self.path = path

    def found(self, key: str) -> bool:
        return bool(self.node.found(key))

    def has_dict(self, key: str) -> bool:
        return bool(self.node.found(key) and self.node.isDict(key))

    def get(self, key: str, typ: type) -> Any:
        if not self.node.found(key):
            raise KeyError(key)
        return self.node.get[typ](key)

    def set(self, key: str, value: object) -> None:
        if isinstance(value, Mapping):
            self.child_or_add(key).overwrite(value)
            return
        writer = WRITE_DISPATCH.get(type(value))
        if writer is None:
            raise TypeError(
                f"Unsupported value type for key '{key}': {type(value).__name__} "
                f"(supported: str, int, float, bool, or a nested mapping)"
            )
        writer(self.node, key, value)

    def overwrite(self, mapping: Mapping[str, Any]) -> None:
        self.node.clear()
        _OF_STRATEGY._write(self.node, dict(mapping))

    def remove(self, key: str) -> None:
        self.node.remove(key)  # pybFoam >= 0.5.1; returns False when absent

    def child(self, key: str) -> "_Backend":
        if not self.has_dict(key):
            raise KeyError(f"Subdict '{key}' not found")
        return _OFBackend(self.root, self.node.subDict(key), self.path)

    def child_or_add(self, key: str) -> "_Backend":
        # subDictOrAdd returns a detached handle; re-fetch via subDict for a live one.
        self.node.subDictOrAdd(key)
        return _OFBackend(self.root, self.node.subDict(key), self.path)

    def read_fields(self, model_cls: type[BaseConfig]) -> dict[str, Any]:
        return _read_fields(model_cls, _to_python(self.node))

    def write(self, path: Path) -> None:
        self.root.write(str(path))


class _MapBackend(_Backend):
    """Nested Python mapping node -- JSON/YAML, dumped whole on write."""

    def __init__(self, root: Any, node: Any, path: Path, fmt: str) -> None:
        self.root = root
        self.node = node
        self.path = path
        self.fmt = fmt

    def found(self, key: str) -> bool:
        return isinstance(self.node, dict) and key in self.node

    def has_dict(self, key: str) -> bool:
        return isinstance(self.node, dict) and isinstance(self.node.get(key), dict)

    def get(self, key: str, typ: type) -> Any:
        if key not in self.node:
            raise KeyError(key)
        return typ(self.node[key])

    def set(self, key: str, value: object) -> None:
        self.node[key] = _plainify(value)

    def overwrite(self, mapping: Mapping[str, Any]) -> None:
        self.node.clear()
        self.node.update(_plainify(mapping))

    def remove(self, key: str) -> None:
        self.node.pop(key, None)

    def child(self, key: str) -> "_Backend":
        if not self.has_dict(key):
            raise KeyError(f"Subdict '{key}' not found")
        return _MapBackend(self.root, self.node[key], self.path, self.fmt)

    def child_or_add(self, key: str) -> "_Backend":
        if not isinstance(self.node.get(key), dict):
            self.node[key] = {}
        return _MapBackend(self.root, self.node[key], self.path, self.fmt)

    def read_fields(self, model_cls: type[BaseConfig]) -> dict[str, Any]:
        return dict(self.node)  # native formats are already typed

    def write(self, path: Path) -> None:
        _dump_map(path, self.root, self.fmt)


def _open(path: Path) -> _Backend:
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    fmt = _fmt(path)
    if fmt == "openfoam":
        root = pyf.dictionary.read(str(path))
        return _OFBackend(root, root, path)
    data = _load_map(path, fmt)  # root and node are the *same* object at the top
    return _MapBackend(data, data, path, fmt)


def _write_payload(
    path: Path, data: dict[str, Any], key: tuple[str, ...], fmt: str
) -> None:
    """Persist a pre-dumped model payload to *path* under *key*, creating as needed.

    The shared engine behind :meth:`DictFile.save`, ``BaseConfig.save`` and
    :func:`neofoam.io.write_configs`. Whole-file OpenFOAM writes inject a
    ``FoamFile`` header for header-less dict configs (field configs carry their
    own); an OpenFOAM sub-dict write requires the file to already exist (its
    header cannot be synthesised). JSON/YAML sub-dict writes merge into the file,
    preserving sibling sections.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "openfoam":
        if key:
            if not path.is_file():
                raise FileNotFoundError(
                    f"Configuration file not found: {path} "
                    f"(required for subdict write {'.'.join(key)!r})"
                )
            root = pyf.dictionary.read(str(path))
            node = root
            for part in key:
                node.subDictOrAdd(part)
                node = node.subDict(part)
            node.clear()
            _OF_STRATEGY._write(node, data)
        else:
            root = (
                pyf.dictionary.read(str(path)) if path.is_file() else pyf.dictionary()
            )
            root.clear()
            if "FoamFile" not in data:
                _OF_STRATEGY._write(root, {"FoamFile": foam_header(path)})
            _OF_STRATEGY._write(root, data)
        root.write(str(path))
        return
    if key:
        existing = _load_map(path, fmt) if path.is_file() else {}
        cursor = existing
        for part in key[:-1]:
            child = cursor.get(part)
            if not isinstance(child, dict):
                child = {}
                cursor[part] = child
            cursor = child
        cursor[key[-1]] = dict(data)
        _dump_map(path, existing, fmt)
    else:
        _dump_map(path, data, fmt)


# --------------------------------------------------------------------------- #
# Facade                                                                       #
# --------------------------------------------------------------------------- #


class _GetProxy:
    """Backs ``d.get[T](key)`` -- subscript picks the type, the call the key."""

    def __init__(self, owner: "DictFile") -> None:
        self._owner = owner

    def __getitem__(self, typ: type[T]) -> Callable[[Key], T]:
        def getter(key: Key) -> T:
            return cast(T, self._owner._get(typ, key))

        return getter


class DictFile:
    """A dictionary file, edited through the OpenFOAM ``dictionary`` interface."""

    def __init__(self, path: Union[str, Path]) -> None:
        self._backend = _open(Path(path))

    @classmethod
    def _wrap(cls, backend: _Backend) -> "DictFile":
        obj = cls.__new__(cls)
        obj._backend = backend
        return obj

    @property
    def path(self) -> Path:
        return self._backend.path

    # -- read: get[T] / subDict / found --------------------------------------

    @property
    def get(self) -> _GetProxy:
        return _GetProxy(self)

    def _get(self, typ: type, key: Key) -> Any:
        keys = _as_tuple(key)
        node = self._backend
        for part in keys[:-1]:
            node = node.child(part)
        return node.get(keys[-1], typ)

    def found(self, key: Key) -> bool:
        keys = _as_tuple(key)
        node = self._backend
        for part in keys[:-1]:
            if not node.has_dict(part):
                return False
            node = node.child(part)
        return node.found(keys[-1])

    def subDict(self, key: Key) -> "DictFile":
        node = self._backend
        for part in _as_tuple(key):
            node = node.child(part)
        return DictFile._wrap(node)

    # -- write: set / remove / write -----------------------------------------

    def set(self, key: Key, value: Any) -> None:
        """Set the entry at *key* (a bare key or a tuple address) to *value*."""
        keys = _as_tuple(key)
        node = self._backend
        for part in keys[:-1]:
            node = node.child_or_add(part)
        node.set(keys[-1], value)

    def remove(self, key: Key) -> None:
        keys = _as_tuple(key)
        node = self._backend
        for part in keys[:-1]:
            if not node.has_dict(part):
                return  # a missing parent means the entry is already absent
            node = node.child(part)
        node.remove(keys[-1])

    def write(self, path: Optional[Union[str, Path]] = None) -> None:
        self._backend.write(Path(path) if path is not None else self._backend.path)

    # -- pydantic bridge -----------------------------------------------------

    def fill(self, model_cls: type[TModel], *, validate: bool = True) -> TModel:
        """Read a config out of the file (dict -> validated model).

        The sub-dict is the model's declared ``subdict`` (the model's *file* is
        ignored -- this ``DictFile`` owns the file). With ``validate=False`` the
        data is coerced but not validated (``model_construct``), mirroring
        :meth:`BaseConfig.load`.
        """
        scope = self._scope(self._model_key(model_cls))
        data = scope._backend.read_fields(model_cls)
        if validate:
            return model_cls.model_validate(data)
        return cast(TModel, model_cls.model_construct(**data))

    @classmethod
    def save(cls, instance: BaseConfig, path: Union[str, Path]) -> None:
        """Persist *instance* to *path*, creating the file/dirs as needed.

        The file format comes from the suffix and the sub-dict from the config's
        declared metadata. Unlike :meth:`set` (which edits an already-open file),
        this creates from scratch -- injecting a ``FoamFile`` header for
        header-less OpenFOAM dict configs -- and is the engine behind
        :meth:`BaseConfig.save`.
        """
        p = Path(path)
        fmt = _fmt(p)
        _write_payload(
            p,
            _dump_model(instance, fmt),
            cls._model_key(type(instance)),
            fmt,
        )

    @staticmethod
    def _model_key(model_cls: type[BaseConfig]) -> tuple[str, ...]:
        io = model_cls.io_config
        subdict = io.subdict if io is not None else None
        return tuple(subdict.split(".")) if subdict else ()

    def _scope(self, key: Key) -> "DictFile":
        keys = _as_tuple(key)
        return self.subDict(keys) if keys else self
