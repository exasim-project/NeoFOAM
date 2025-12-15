# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

"""
Model Registry

Central registry for inter-model communication during initialization.
"""

from typing import Any


class ModelRegistry:
    """
    Central registry for inter-model communication during RESOLVE_DEPENDENCIES stage.

    Models are registered by name and can be retrieved by other models
    during the RESOLVE_DEPENDENCIES stage to establish dependencies.
    """

    def __init__(self):
        self._models: dict[str, Any] = {}

    def register(self, name: str, model: Any) -> None:
        """
        Register a model by name.

        Args:
            name: Unique identifier for the model
            model: The model instance to register
        """
        self._models[name] = model

    def get(self, name: str) -> Any:
        """
        Get a registered model by name.

        Args:
            name: Name of the model to retrieve

        Returns:
            The model instance, or None if not found
        """
        return self._models.get(name)

    def all(self) -> dict[str, Any]:
        """
        Get all registered models.

        Returns:
            Dictionary mapping model names to model instances
        """
        return self._models.copy()

    def contains(self, name: str) -> bool:
        """
        Check if a model is registered.

        Args:
            name: Name of the model to check

        Returns:
            True if the model is registered, False otherwise
        """
        return name in self._models

    def get_by_type(self, model_type: type) -> list[Any]:
        """
        Get all registered models of a specific type.

        Useful for working with multiple instances of the same model type
        (e.g., multiple heat sources, multiple porous zones).

        Args:
            model_type: The type/class to filter by

        Returns:
            List of all model instances of the specified type

        Example:
            heat_sources = registry.get_by_type(HeatSource)
            for source in heat_sources:
                source.enabled = False
        """
        return [
            model for model in self._models.values() if isinstance(model, model_type)
        ]

    def get_by_prefix(self, prefix: str) -> dict[str, Any]:
        """
        Get all models with names starting with a prefix.

        Useful for finding related model instances that follow a naming
        convention (e.g., "heat_source_1", "heat_source_2").

        Args:
            prefix: The prefix to match against model names

        Returns:
            Dictionary of models with matching names

        Example:
            sources = registry.get_by_prefix("heat_source_")
            for name, source in sources.items():
                print(f"{name}: {source.power}W")
        """
        return {
            name: model
            for name, model in self._models.items()
            if name.startswith(prefix)
        }

    def get_adaptable_fields(self, model_name: str) -> dict[str, Any]:
        """
        Get all adaptable fields and their current values from a model.

        Scans a model's Pydantic field definitions for fields marked with
        AdaptableField and returns their current values.

        Args:
            model_name: Name of the model to inspect

        Returns:
            Dictionary mapping adaptable field names to their current values,
            or empty dict if model not found or has no adaptable fields

        Example:
            adaptable = registry.get_adaptable_fields("pressure_algorithm")
            # Returns: {"use_buoyancy": False, "algorithm": "SIMPLE"}

            # Can check what's adaptable before modifying
            if "use_buoyancy" in adaptable:
                pressure.use_buoyancy = True
        """
        model = self.get(model_name)
        if not model:
            return {}

        # Check if model has Pydantic fields
        if not hasattr(model.__class__, "model_fields"):
            return {}

        result = {}
        for field_name, field_info in model.__class__.model_fields.items():
            # Check if field is marked as adaptable
            if field_info.json_schema_extra and field_info.json_schema_extra.get(
                "adaptable", False
            ):
                result[field_name] = getattr(model, field_name)

        return result
