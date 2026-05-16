# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""YAML reading and writing strategy."""

from typing import Any, Optional
from pathlib import Path

import yaml
from pydantic import BaseModel

from neofoam.io.strategies.subdict import SubdictMixin


class YAMLStrategy(SubdictMixin):
    """reading and writing strategy from YAML files with optional subdict support.

    Supports three modes via SubdictMixin:
    1. Full file (subdict_path=None): Read/write entire YAML file
    2. Flat subdict (subdict_path="PIMPLE"): Read/write top-level key
    3. Nested subdict (subdict_path="solvers.p"): Read/write nested path with dot notation

    Examples:
        YAMLStrategy()                  # Full file
        YAMLStrategy("PIMPLE")          # Flat subdict
        YAMLStrategy("solvers.p")       # Nested subdict
    """

    def __init__(self, subdict_path: Optional[str] = None):
        """Initialize YAML strategy.

        Args:
            subdict_path: Optional path to subdict (e.g., "PIMPLE" or "solvers.p").
                         If contains dots, treated as nested path.
        """
        super().__init__(subdict_path)

    def read(
        self, model_cls: type[BaseModel], path: Path, encoding: str = "utf-8"
    ) -> dict[str, Any]:
        """Read YAML configuration file.

        Args:
            model_cls: The model class
            path: Path to the YAML file
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
            full_data = yaml.safe_load(f) or {}

        return self._extract_subdict(full_data)

    def write(self, instance: BaseModel, path: Path, encoding: str = "utf-8") -> None:
        """Write configuration to YAML file.

        Args:
            instance: The model instance to write
            path: Path to the YAML file
            encoding: File encoding (default: utf-8)
        """
        data = instance.model_dump(mode="python", exclude_none=False)
        path.parent.mkdir(parents=True, exist_ok=True)

        if not self.subdict_path:
            with open(path, "w", encoding=encoding) as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        else:
            existing_data: dict[str, Any] = {}
            if path.exists():
                with open(path, "r", encoding=encoding) as f:
                    existing_data = yaml.safe_load(f) or {}

            merged = self._merge_subdict(existing_data, data)
            with open(path, "w", encoding=encoding) as f:
                yaml.dump(merged, f, default_flow_style=False, sort_keys=False)
