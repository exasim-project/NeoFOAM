# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""Shared validation types to avoid circular imports."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Type

import yaml
from pydantic import BaseModel, ValidationError

can_load_toml = True
try:
    import tomllib  # type: ignore[import-not-found]
except ImportError:
    can_load_toml = False


def default_validation_strategy(
    baseModel: Type[BaseModel], file_path: Path, encoding: str
) -> None:
    """
    Default strategy to read and parse a file into a Pydantic model to validate it.

    Args:
        baseModel: The Pydantic model class to parse the data into.
        file_path: Path to the file to be read.
        encoding: Encoding of the file.
    """
    if not hasattr(baseModel, "from_file"):
        baseModel.from_file(file_path)
    if file_path.suffix == ".toml":
        if not can_load_toml:
            raise ValueError("TOML support is not available")
        with open(file_path, "rb") as f:
            data = tomllib.load(f)
    else:
        with open(file_path, "r", encoding=encoding) as f:
            if file_path.suffix in [".yaml", ".yml"]:
                data = yaml.safe_load(f)
            elif file_path.suffix == ".json":
                data = json.load(f)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
    baseModel.model_validate(data)


@dataclass(frozen=True)
class ValidationErrors:
    field: Any
    error_type: str
    message: str
    file_name: str
    input_value: Any = None
    subdict: Optional[str] = None


@dataclass(frozen=True)
class ModelInputDefinition:
    baseModel: Type[BaseModel]
    relative_path: str
    encoding: str = "utf-8"
    required: bool = True
    description: str = ""
    reading_strategy: Callable[[Type[BaseModel], Path, str], None] = (
        default_validation_strategy
    )

    def validate(self, case_dir: str = ".") -> list[ValidationErrors]:
        """
        Validate that the file exists and is valid according to the baseModel.

        Args:
            case_dir: Path to the case directory

        Returns:
            errors: List[ValidationErrors]
        """
        file_path = Path(case_dir) / self.relative_path
        validation_errors: list[ValidationErrors] = []

        if not file_path.exists() and self.required:
            validation_errors.append(
                ValidationErrors(
                    field=None,
                    message="File not found",
                    file_name=str(file_path),
                    error_type="FileNotFound",
                )
            )
            return validation_errors

        try:
            self.reading_strategy(self.baseModel, file_path, self.encoding)
        except ValidationError as errors:
            for err in errors.errors(include_url=False):
                validation_errors.append(
                    ValidationErrors(
                        field=err.get("loc", [None]),
                        message=err.get("msg", "Validation error"),
                        file_name=str(file_path),
                        input_value=err.get("input", None),
                        error_type=err.get("type", "UnknownError"),
                    )
                )

        return validation_errors
