# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

"""
Configuration Context

Central context for inter-model configuration exchange during initialization.
"""

from typing import Any, Optional


def is_configurable_field(field_info: Any) -> bool:
    """
    Check if a field is marked as Configurable.
    """
    # Check if field has metadata with "configurable"
    if hasattr(field_info, "metadata") and field_info.metadata:
        return "configurable" in field_info.metadata
    return False


class ConfigContext:
    """
    Context for inter-model configuration exchange during RESOLVE_DEPENDENCIES stage.

    Models are registered by name and can be retrieved by other models
    during the RESOLVE_DEPENDENCIES stage to establish dependencies and
    exchange configuration.

    Supports both intra-region (same region) and inter-region (cross-region)
    model access via dot notation:
    - "model_name" → model in current region
    - "region.model_name" → model in specified region
    """

    def __init__(self, current_region: str = "default"):
        """
        Initialize the configuration context.

        Args:
            current_region: Name of the current region (default: "default")
        """
        self.current_region = current_region
        self._regions: dict[str, dict[str, Any]] = {current_region: {}}

    def _parse_path(self, path: str) -> tuple[str, str]:
        """Parse a model path into (region, name)."""
        if "." in path:
            maybe_region, rest = path.split(".", 1)
            if maybe_region in self._regions:
                return maybe_region, rest
        return self.current_region, path

    def register(self, name: str, model: Any, region: Optional[str] = None) -> None:
        """
        Register a model by name in a region.

        Args:
            name: Unique identifier for the model within the region
            model: The model instance to register
            region: Region name (defaults to current_region)
        """
        region = region or self.current_region
        if region not in self._regions:
            self._regions[region] = {}
        self._regions[region][name] = model

    def get(self, path: str) -> Any:
        """
        Get a registered model by path.

        Supports both intra-region and inter-region access:
        - "model_name" → model in current region
        - "region.model_name" → model in specified region

        Args:
            path: Model path (name or region.name)

        Returns:
            The model instance, or None if not found

        Example:
            # Same region access
            velocity = config.get("velocity")
            transport = config.get("transport")

            # Cross-region access
            fluid_temp = config.get("fluid.temperature")
            solid_temp = config.get("solid.temperature")
        """
        region, name = self._parse_path(path)
        return self._regions.get(region, {}).get(name)

    def all(self, region: Optional[str] = None) -> dict[str, Any]:
        """
        Get all registered models in a region.

        Args:
            region: Region name (defaults to current_region)

        Returns:
            Dictionary mapping model names to model instances
        """
        region = region or self.current_region
        return self._regions.get(region, {}).copy()

    def contains(self, path: str) -> bool:
        """
        Check if a model is registered.

        Supports both intra-region and inter-region paths.

        Args:
            path: Model path (name or region.name)

        Returns:
            True if the model is registered, False otherwise
        """
        region, name = self._parse_path(path)
        return region in self._regions and name in self._regions[region]

    def get_by_type(self, model_type: type, region: Optional[str] = None) -> list[Any]:
        """
        Get all registered models of a specific type in a region.

        Useful for working with multiple instances of the same model type
        (e.g., multiple heat sources, multiple porous zones).

        Args:
            model_type: The type/class to filter by
            region: Region name (defaults to current_region)

        Returns:
            List of all model instances of the specified type

        Example:
            heat_sources = config.get_by_type(HeatSource)
            for source in heat_sources:
                source.enabled = False
        """
        region = region or self.current_region
        models = self._regions.get(region, {})
        return [model for model in models.values() if isinstance(model, model_type)]

    def get_by_prefix(
        self, prefix: str, region: Optional[str] = None
    ) -> dict[str, Any]:
        """
        Get all models with names starting with a prefix in a region.

        Useful for finding related model instances that follow a naming
        convention (e.g., "heat_source_1", "heat_source_2").

        Args:
            prefix: The prefix to match against model names
            region: Region name (defaults to current_region)

        Returns:
            Dictionary of models with matching names

        Example:
            sources = config.get_by_prefix("heat_source_")
            for name, source in sources.items():
                print(f"{name}: {source.power}W")
        """
        region = region or self.current_region
        models = self._regions.get(region, {})
        return {
            name: model for name, model in models.items() if name.startswith(prefix)
        }

    def get_configurable_fields(self, path: str) -> dict[str, Any]:
        """
        Get all configurable fields and their current values from a model.

        Scans a model's field definitions for fields marked with
        Configurable[T] type annotation and returns their current values.

        Args:
            path: Model path (name or region.name)

        Returns:
            Dictionary mapping configurable field names to their current values,
            or empty dict if model not found or has no configurable fields

        Example:
            configurable = config.get_configurable_fields("pressure_algorithm")
            # Returns: {"use_buoyancy": False, "algorithm": "SIMPLE"}

            # Can check what's configurable before modifying
            if "use_buoyancy" in configurable:
                pressure.use_buoyancy = True
        """
        model = self.get(path)
        if not model:
            return {}

        # Check if model has Pydantic fields
        if not hasattr(model.__class__, "model_fields"):
            return {}

        result = {}
        for field_name, field_info in model.__class__.model_fields.items():
            # Check if field is marked as configurable using type annotation
            if is_configurable_field(field_info):
                result[field_name] = getattr(model, field_name)

        return result

    def __getattr__(self, name: str) -> Any:
        """
        Enable attribute-style access to registered models.

        This allows models to be accessed as attributes:
            config.algorithm  # equivalent to config.get("algorithm")

        Args:
            name: The model name

        Returns:
            The model instance, or raises AttributeError if not found
        """
        # Avoid infinite recursion for internal / dunder attributes
        if name.startswith("_"):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

        model = self.get(name)
        if model is None:
            raise AttributeError(
                f"No model registered with name '{name}' in region '{self.current_region}'"
            )
        return model

    @property
    def regions(self) -> list[str]:
        """
        Get list of all registered region names.

        Returns:
            List of region names
        """
        return list(self._regions.keys())
