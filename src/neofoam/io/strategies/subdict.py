# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""Shared subdict navigation logic for file strategies."""

from typing import Any, Optional


class SubdictMixin:
    """Shared subdict navigation for file strategies.

    Supports three modes:
    1. Full file (subdict_path=None): Read/write entire file
    2. Flat subdict (subdict_path="PIMPLE"): Read/write top-level key
    3. Nested subdict (subdict_path="solvers.p"): Read/write nested path with dot notation

    Examples:
        SubdictMixin()                   # Full file
        SubdictMixin("PIMPLE")           # Flat subdict
        SubdictMixin("solvers.p")        # Nested subdict
    """

    def __init__(self, subdict_path: Optional[str] = None):
        """Initialize subdict navigation.

        Args:
            subdict_path: Optional path to subdict (e.g., "PIMPLE" or "solvers.p").
                         If contains dots, treated as nested path.
        """
        self.subdict_path = subdict_path
        self.path_parts = subdict_path.split(".") if subdict_path else None

    def _get_nested(self, data: dict, path_parts: list[str]) -> dict:
        """Navigate to nested dict, raising KeyError if path doesn't exist.

        Args:
            data: Root dictionary to navigate
            path_parts: List of keys forming the path

        Returns:
            The nested dict at the given path

        Raises:
            KeyError: If any part of the path doesn't exist or isn't a dict
        """
        current = data
        traversed: list[str] = []
        for part in path_parts:
            if not isinstance(current, dict) or part not in current:
                full_path = ".".join(path_parts)
                traversed_path = ".".join(traversed) if traversed else "<root>"
                raise KeyError(
                    f"Subdict path '{full_path}' not found: "
                    f"key '{part}' does not exist at '{traversed_path}'"
                )
            traversed.append(part)
            current = current[part]
        if not isinstance(current, dict):
            full_path = ".".join(path_parts)
            raise KeyError(
                f"Subdict path '{full_path}' resolved to "
                f"{type(current).__name__}, expected dict"
            )
        return current

    def _set_nested(self, data: dict, path_parts: list[str], value: dict) -> None:
        """Set value at nested path, creating intermediate dicts as needed.

        Args:
            data: Root dictionary to modify
            path_parts: List of keys forming the path
            value: Dict value to set at the leaf
        """
        current = data
        for part in path_parts[:-1]:
            if part not in current or not isinstance(current[part], dict):
                current[part] = {}
            current = current[part]
        current[path_parts[-1]] = value

    def _extract_subdict(self, full_data: dict[str, Any]) -> dict[str, Any]:
        """Extract the subdict portion from full file data.

        Args:
            full_data: Complete parsed file data

        Returns:
            The subdict data

        Raises:
            KeyError: If the subdict path doesn't resolve
        """
        if not self.subdict_path:
            return full_data

        if len(self.path_parts) == 1:
            key = self.path_parts[0]
            if key not in full_data:
                raise KeyError(
                    f"Subdict '{key}' not found in file. "
                    f"Available top-level keys: {list(full_data.keys())}"
                )
            value = full_data[key]
            if not isinstance(value, dict):
                raise KeyError(
                    f"Subdict '{key}' resolved to {type(value).__name__}, expected dict"
                )
            return value
        else:
            return self._get_nested(full_data, self.path_parts)

    def _merge_subdict(
        self, existing_data: dict[str, Any], new_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Merge new subdict data into existing file data, preserving other sections.

        Args:
            existing_data: Current full file data
            new_data: New data to write at the subdict path

        Returns:
            Merged full file data
        """
        if not self.subdict_path:
            return new_data

        if len(self.path_parts) == 1:
            existing_data[self.path_parts[0]] = new_data
        else:
            self._set_nested(existing_data, self.path_parts, new_data)

        return existing_data
