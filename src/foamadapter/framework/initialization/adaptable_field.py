# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

"""
Adaptable Field

Factory function for creating Pydantic fields that can be modified by other
models during the RESOLVE_DEPENDENCIES stage.
"""


def AdaptableField(**kwargs):
    """
    Mark a field as adaptable by other models during RESOLVE_DEPENDENCIES stage.

    Adaptable fields are parameters that change a model's behavior or operations.
    Other models can modify these fields during the RESOLVE_DEPENDENCIES stage to select
    different implementations or algorithm variants.

    This is a wrapper around Pydantic's Field that adds 'adaptable' metadata,
    allowing the framework to identify which parameters are intended for
    inter-model configuration.

    Args:
        **kwargs: All standard Pydantic Field arguments (default, gt, ge, lt, le,
                  description, etc.)

    Returns:
        A Pydantic Field with adaptable metadata

    Example:
        class PressureAlgorithm(BaseModel):
            # Adaptable field - other models can modify this
            use_buoyancy: bool = AdaptableField(
                default=False,
                description="Use buoyancy-modified pressure equation"
            )

            # Regular field - not modifiable by other models
            tolerance: float = Field(default=1e-6, gt=0)

            # Implementation dispatch based on adaptable field
            _implementations = {
                False: StandardPressure,
                True: BuoyantPressure
            }

            def get_operations(self):
                impl = self._implementations[self.use_buoyancy]
                return impl().get_operations()

        class BuoyancyModel(BaseModel):
            @Model.resolve_dependencies
            def resolve(self, registry: ModelRegistry):
                # Modify adaptable field in another model
                pressure = registry.get("pressure_algorithm")
                pressure.use_buoyancy = True  # Switches implementation
    """
    from pydantic import Field

    # Add adaptable metadata
    json_schema_extra = kwargs.get("json_schema_extra", {}) or {}
    json_schema_extra["adaptable"] = True
    kwargs["json_schema_extra"] = json_schema_extra

    return Field(**kwargs)
