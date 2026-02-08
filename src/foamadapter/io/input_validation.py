# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

from pathlib import Path
from typing import Tuple, Type

from pydantic import BaseModel, ValidationError

from .strategies import BaseConfig
from .validation_types import ModelInputDefinition, ValidationErrors


def validate_models(models: list[BaseConfig]) -> list[ValidationErrors]:
    """
    Validate BaseConfig models and return detailed validation errors.

    Uses the file_name and subdict properties from BaseConfig to provide
    context about where the validation error occurred.

    Args:
        models: List of BaseConfig instances to validate

    Returns:
        List of ValidationErrors with file and subdict context
    """
    validation_errors: list[ValidationErrors] = []

    for model in models:
        try:
            model_type = type(model)
            model_type.model_validate(model.model_dump())
        except ValidationError as errors:
            # Collect errors from this model
            for err in errors.errors(include_url=False):
                validation_errors.append(
                    ValidationErrors(
                        field=err.get("loc", [None]),
                        message=err.get("msg", "Validation error"),
                        file_name=model.file_name,
                        input_value=err.get("input", None),
                        error_type=err.get("type", "UnknownError"),
                        subdict=model.subdict,
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
