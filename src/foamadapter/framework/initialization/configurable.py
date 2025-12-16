# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

"""
Configurable Type Annotation

Type annotation for fields that can be modified by other models during
the RESOLVE_DEPENDENCIES stage.
"""

from typing import Annotated, TypeVar, Any

T = TypeVar("T")


class _ConfigurableMeta(type):
    """Metaclass to make Configurable subscriptable."""

    def __getitem__(cls, item: type) -> type:
        """Allow Configurable[T] syntax."""
        return Annotated[item, "configurable"]


class Configurable(metaclass=_ConfigurableMeta):
    """
    Type annotation for configurable fields.

    Configurable fields are parameters that can be modified by other models
    during the RESOLVE_DEPENDENCIES stage to change a model's behavior or
    select different implementations.

    The type annotation itself documents which fields are intended for
    inter-model configuration, making the design intent explicit in the
    type system.

    Usage:
        from pydantic import BaseModel

        class PressureAlgorithm(BaseModel):
            # Configurable field - other models can modify this
            use_buoyancy: Configurable[bool] = False
            gravity: Configurable[tuple[float, float, float]] = (0, 0, -9.81)

            # Regular field - not modifiable by other models
            tolerance: float = 1e-6

            # Implementation dispatch based on configurable field
            _implementations = {
                False: StandardPressure,
                True: BuoyantPressure
            }

            def get_operations(self):
                impl = self._implementations[self.use_buoyancy]
                return impl().get_operations()

        class BuoyancyModel(BaseModel):
            enabled: bool = True
            gravity: tuple[float, float, float] = (0, 0, -9.81)

            @Model.resolve_dependencies
            def resolve(self, config: ConfigContext):
                if self.enabled:
                    # Modify configurable field in another model
                    pressure = config.get("pressure_algorithm")
                    pressure.use_buoyancy = True  # Switches implementation
                    pressure.gravity = self.gravity
    """

    pass


def is_configurable_field(field_info: Any) -> bool:
    """
    Check if a field is marked as Configurable.

    Args:
        field_info: The Pydantic FieldInfo object

    Returns:
        True if the field is marked as Configurable

    Example:
        from pydantic import BaseModel

        class MyModel(BaseModel):
            config_field: Configurable[bool] = False
            regular_field: float = 1.0

        for name, field_info in MyModel.model_fields.items():
            if is_configurable_field(field_info):
                print(f"{name} is configurable")
    """
    # Check if field has metadata with "configurable"
    if hasattr(field_info, "metadata") and field_info.metadata:
        return "configurable" in field_info.metadata
    return False


def get_configurable_fields(model_class: type) -> dict[str, type]:
    """
    Get all configurable fields from a model class.

    Scans a Pydantic model's field definitions for fields marked with
    Configurable and returns their names and types.

    Args:
        model_class: The model class to inspect

    Returns:
        Dictionary mapping configurable field names to their value types

    Example:
        class MyModel(BaseModel):
            use_buoyancy: Configurable[bool] = False
            gravity: Configurable[tuple] = (0, 0, -9.81)
            tolerance: float = 1e-6

        fields = get_configurable_fields(MyModel)
        # Returns: {"use_buoyancy": bool, "gravity": tuple}
    """
    if not hasattr(model_class, "model_fields"):
        return {}

    result = {}
    for field_name, field_info in model_class.model_fields.items():
        if is_configurable_field(field_info):
            # Get the annotation type
            result[field_name] = field_info.annotation

    return result
