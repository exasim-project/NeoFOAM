# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

from .base import BaseConfig
from .validation_types import ValidationErrors


def validate_models(models: list[BaseConfig]) -> list[ValidationErrors]:
    """Validate BaseConfig models and return detailed validation errors.

    Delegates to each model's :meth:`~BaseConfig.validate` method which
    re-validates via ``model_validate`` and converts errors into
    ``ValidationErrors`` with file and subdict context.

    This is the batch entry-point for the staged workflow::

        models = [Config.load(case_dir, validate=False) for Config in registry]
        errors = validate_models(models)

    Args:
        models: List of BaseConfig instances to validate

    Returns:
        List of ValidationErrors with file and subdict context
    """
    return [err for model in models for err in model.check_validation()]
