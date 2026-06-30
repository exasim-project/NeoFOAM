# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""JSON reading and writing strategy."""

import json
from typing import Any, Optional
from pathlib import Path

from pydantic import BaseModel

from neofoam.io.strategies.subdict import SubdictMixin


class JSONStrategy(SubdictMixin):
    """reading and writing strategy from JSON files with optional subdict support.

    Supports three modes via SubdictMixin:
    1. Full file (subdict_path=None): Read/write entire JSON file
    2. Flat subdict (subdict_path="database"): Read/write top-level key
    3. Nested subdict (subdict_path="services.database"): Read/write nested path with dot notation

    Examples:
        JSONStrategy()                       # Full file
        JSONStrategy("database")             # Flat subdict
        JSONStrategy("services.database")    # Nested subdict
    """

    def __init__(self, subdict_path: Optional[str] = None):
        """Initialize JSON strategy.

        Args:
            subdict_path: Optional path to subdict (e.g., "database" or "services.database").
                         If contains dots, treated as nested path.
        """
        super().__init__(subdict_path)

    def read(
        self, model_cls: type[BaseModel], path: Path, encoding: str = "utf-8"
    ) -> dict[str, Any]:
        """Read JSON configuration file.

        Args:
            model_cls: The model class
            path: Path to the JSON file
            encoding: File encoding (default: utf-8)

        Returns:
            Parsed configuration data (full file or subdict)

        Raises:
            FileNotFoundError: If the file does not exist
            KeyError: If the subdict path doesn't resolve
        """
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, "r", encoding=encoding) as f:
            full_data = json.load(f)

        return self._extract_subdict(full_data)

    def write(self, instance: BaseModel, path: Path, encoding: str = "utf-8") -> None:
        """Write configuration to JSON file.

        Args:
            instance: The model instance to write
            path: Path to the JSON file
            encoding: File encoding (default: utf-8)
        """
        # ``by_alias=True`` so a ``Field(alias="div(phi,U)")``-style declaration
        # round-trips under its on-disk key, not pydantic's sanitised Python
        # attribute name.
        data = instance.model_dump(mode="python", exclude_none=False, by_alias=True)
        path.parent.mkdir(parents=True, exist_ok=True)

        if not self.subdict_path:
            with open(path, "w", encoding=encoding) as f:
                json.dump(data, f, indent=2, sort_keys=False)
        else:
            existing_data = {}
            if path.exists():
                with open(path, "r", encoding=encoding) as f:
                    existing_data = json.load(f)

            merged = self._merge_subdict(existing_data, data)
            with open(path, "w", encoding=encoding) as f:
                json.dump(merged, f, indent=2)
