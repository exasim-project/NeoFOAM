# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""Base class for optional physics models that extend IncompressibleFluid solver."""

from abc import abstractmethod
from typing import Any

from pydantic import BaseModel

from foamadapter.core.plugin_system import PluginSystem
from foamadapter.framework.decorator import decorated_member_functions
from foamadapter.framework.operations import (
    Operation,
    OperationCollection,
)


@PluginSystem.register(discriminator_variable="model", discriminator="model_type")
class IncompressibleFluidModel(BaseModel):
    """
    Base class for optional physics models that extend IncompressibleFluid.

    Models implementing this class:
    - Participate in 3-stage initialization (LOAD, RESOLVE_DEPENDENCIES, BUILD)
    - Contribute operations to the execution graph
    - Are registered via PluginSystem for type-safe configuration

    Core components (pressure_velocity, transport, turbulence) are NOT
    IncompressibleFluidModels - they have their own base classes above.

    Example models: Buoyancy, Radiation, Species Transport, etc.
    """

    model_config = {"arbitrary_types_allowed": True}

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique model identifier."""
        ...

    @classmethod
    def create(cls, *, model: dict[str, Any]) -> Any:
        """Factory classmethod to create fluid model from config.

        Implemented explicitly for type safety. Calls plugin_model generated
        by @PluginSystem.register decorator.
        """
        return cls.plugin_model(model=model)  # type: ignore[attr-defined]

    @classmethod
    def detect_models(cls) -> list["IncompressibleFluidModel"]:
        """Auto-detect models from case files."""
        # Ensure all models are imported so they register
        import pkgutil
        import importlib
        import foamadapter.models as models_pkg

        # Avoid redundant imports if already loaded, though importlib handles it
        for _, name, _ in pkgutil.iter_modules(models_pkg.__path__):
            importlib.import_module(f"foamadapter.models.{name}")

        detected = []
        registry = PluginSystem.get_registered(cls.__name__)
        # from pybFoam import Info
        # Info(f"DEBUG: Found {len(registry.plugin_registry) if registry else 0} registered models")
        if registry:
            for model_class in registry.plugin_registry:
                is_detected = hasattr(model_class, "detect") and model_class.detect()
                # Info(f"DEBUG: Model {model_class.__name__} detected: {is_detected}")
                if is_detected:
                    detected.append(model_class())
        return detected

    def operations(self) -> OperationCollection:
        """
        Return operations contributed by this model.

        Default implementation discovers @Model.operation decorated methods.
        """
        funcs = decorated_member_functions(self)
        ops = OperationCollection()
        for func in funcs:
            op = Operation.create_SeqOp(func)
            ops.add(op)
        return ops
