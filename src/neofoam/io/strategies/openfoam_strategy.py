# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""OpenFOAM dictionary reading and writing strategy via pybFoam."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Optional

import pybFoam as pyf

from neofoam.io.strategies.subdict import SubdictMixin


_INT_PATTERN = re.compile(r"^[+-]?\d+$")
_FLOAT_PATTERN = re.compile(r"^[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?$")


class OpenFOAMStrategy(SubdictMixin):
    """Read/write OpenFOAM dictionaries with optional subdict support.

    Uses ``pybFoam.dictionary`` as backend and supports subdict extraction and
    merge semantics via ``SubdictMixin``.
    """

    def __init__(self, subdict_path: Optional[str] = None):
        super().__init__(subdict_path)

    def read(self, path: Path, encoding: str = "utf-8") -> dict[str, Any]:
        """Read OpenFOAM dictionary file.

        Args:
            path: Path to OpenFOAM dictionary file
            encoding: Unused (kept for strategy protocol compatibility)

        Returns:
            Parsed configuration data (full file or subdict)

        Raises:
            FileNotFoundError: If the file does not exist
            KeyError: If the subdict path doesn't resolve
        """

        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        full_dict = pyf.dictionary.read(str(path))
        full_data = self._dictionary_to_python(full_dict)
        return self._extract_subdict(full_data)

    def write(self, data: dict[str, Any], path: Path, encoding: str = "utf-8") -> None:
        """Write configuration to OpenFOAM dictionary file.

        Args:
            data: Configuration data to write
            path: Path to OpenFOAM dictionary file
            encoding: Unused (kept for strategy protocol compatibility)

        Raises:
            FileNotFoundError: If subdict write is requested for missing file
            KeyError: If requested subdict path does not exist in file
            TypeError: If value type cannot be serialized
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        if self.subdict_path:
            if not path.exists():
                raise FileNotFoundError(
                    f"Configuration file not found: {path} "
                    f"(required for subdict write '{self.subdict_path}')"
                )

            root_dict = pyf.dictionary.read(str(path))
            target_dict = self._resolve_subdict(root_dict)
            target_dict.clear()
            self._write_mapping(target_dict, data)
            root_dict.write(str(path))
            return

        root_dict = pyf.dictionary.read(str(path)) if path.exists() else pyf.dictionary()
        root_dict.clear()
        self._write_mapping(root_dict, data)
        root_dict.write(str(path))

    def _resolve_subdict(self, root_dict: Any) -> Any:
        """Resolve configured subdict path against pybFoam dictionary object."""
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

    def _dictionary_to_python(self, foam_dict: Any) -> dict[str, Any]:
        """Recursively convert pybFoam dictionary object to plain Python dict."""
        result: dict[str, Any] = {}
        for key_obj in foam_dict.toc():
            key = str(key_obj)
            if foam_dict.isDict(key):
                result[key] = self._dictionary_to_python(foam_dict.subDict(key))
            else:
                raw_value = foam_dict.get[str](key)
                result[key] = self._parse_scalar(str(raw_value))
        return result

    def _parse_scalar(self, value: str) -> Any:
        """Parse OpenFOAM scalar token string into Python scalar type."""
        text = value.strip()
        lower = text.lower()

        if lower in {"yes", "true", "on"}:
            return True
        if lower in {"no", "false", "off"}:
            return False

        if _INT_PATTERN.match(text):
            try:
                return int(text)
            except ValueError:
                pass

        if _FLOAT_PATTERN.match(text):
            try:
                return float(text)
            except ValueError:
                pass

        return text

    def _write_mapping(self, foam_dict: Any, data: dict[str, Any]) -> None:
        """Write Python mapping into pybFoam dictionary object recursively."""
        for key, value in data.items():
            if isinstance(value, dict):
                if not foam_dict.found(key) or not foam_dict.isDict(key):
                    raise KeyError(
                        f"Cannot create nested dictionary '{key}' via pybFoam bindings. "
                        "Pre-create subdictionaries in file before writing."
                    )
                sub = foam_dict.subDict(key)
                sub.clear()
                self._write_mapping(sub, value)
                continue

            self._write_scalar(foam_dict, key, value)

    def _write_scalar(self, foam_dict: Any, key: str, value: Any) -> None:
        """Write scalar value into pybFoam dictionary object."""
        if isinstance(value, bool):
            foam_dict.set(key, "yes" if value else "no")
            return

        if isinstance(value, (int, float, str)):
            foam_dict.set(key, value)
            return

        raise TypeError(
            f"Unsupported value type for OpenFOAM dictionary write at key '{key}': "
            f"{type(value).__name__}"
        )
