# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

import json
import tomllib
import yaml

from dataclasses import dataclass
from typing import Any, Callable, Tuple, Type
from pathlib import Path
from pydantic import BaseModel, ValidationError

can_load_toml = True
try:
    import tomllib
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
    if hasattr(baseModel, "from_file"):
        baseModel.from_file(file_path)
        return
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
            # Try to load and validate the file
            self.reading_strategy(self.baseModel, file_path, self.encoding)
        except ValidationError as errors:
            for err in errors.errors(include_url=False):
                # Collect validation errors
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


class ModelInputCollection:
    """ModelInputCollection that holds model classes and their file specifications."""

    def __init__(self) -> None:
        self._inputs: list[ModelInputDefinition] = []

    def add(self, input_validation_data: ModelInputDefinition) -> None:
        self._inputs.append(input_validation_data)

    def find(self, index: int | Path | str | Type[BaseModel]) -> ModelInputDefinition:
        if isinstance(index, type) and issubclass(index, BaseModel):
            for input_def in self._inputs:
                if input_def.baseModel == index:
                    return input_def
            raise KeyError(f"No input definition found for model: {index}")
        elif isinstance(index, Path) or isinstance(index, str):
            for input_def in self._inputs:
                if input_def.relative_path == str(index):
                    return input_def
            raise KeyError(f"No input definition found for path: {index}")
        elif isinstance(index, int):
            return self._inputs[index]
        else:
            raise TypeError(f"Unsupported index type: {type(index)}")

    def remove(self, index: int) -> None:
        self._inputs.pop(index)

    def validate_case(
        self, case_dir: str | Path = "."
    ) -> Tuple[bool, list[ValidationErrors]]:
        """
        Validate that all required files exist and are valid for the given case.

        Args:
            case_dir: Path to the OpenFOAM case directory

        Returns:
            Tuple of (is_valid: bool, errors: List[ValidationErrors])
        """
        case_path = Path(case_dir)
        validation_errors: list[ValidationErrors] = []

        for input_validation_data in self._inputs:
            errors = input_validation_data.validate(case_dir=str(case_path))
            validation_errors.extend(errors)

        return len(validation_errors) == 0, validation_errors
