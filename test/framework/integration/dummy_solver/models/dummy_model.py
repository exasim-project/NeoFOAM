# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
Base class for DummySolver models.

Mimics SimpleSolverModel structure for testing.
"""

from abc import abstractmethod
from typing import Any

from pydantic import BaseModel

from foamadapter.core.plugin_system import PluginSystem
from foamadapter.framework.initialization.lazy_init import LazyInit
from foamadapter.framework.operations import OperationCollection, Operation
from foamadapter.framework.model_factory import ModelInstance


def Model(name: str) -> ModelInstance:
    """
    Factory for dummy models using the ModelInstance API.
    """
    return ModelInstance(name)


@PluginSystem.register(discriminator_variable="model", discriminator="model_type")
class DummyModel(BaseModel):
    """
    Base class for DummySolver optional models.

    Provides infrastructure for automatic model detection and integration.
    """

    model_config = {"arbitrary_types_allowed": True}

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique model identifier."""
        ...

    @property
    def enabled(self) -> bool:
        """Whether the model is enabled."""
        return True

    @classmethod
    def create(cls, *, model: dict[str, Any]) -> Any:
        """Factory classmethod to create model from config."""
        return cls.plugin_model(model=model)  # type: ignore[attr-defined]

    @classmethod
    def detect_models(cls) -> list[Any]:
        """
        Scan for available models and instantiate configured ones.

        Returns:
            List of enabled model instances
        """
        import pkgutil
        import importlib
        from pathlib import Path

        # Get current package name (e.g., 'framework.dummy_solver.models')
        package_name = ".".join(__name__.split(".")[:-1])
        package_dir = Path(__file__).parent

        # Ensure all models are imported
        for _, name, _ in pkgutil.iter_modules([str(package_dir)]):
            if name != "dummy_model" and name != "__init__":
                importlib.import_module(f"{package_name}.{name}")

        detected = []

        # 1. Find registered classes via PluginSystem
        registry = PluginSystem.get_registered(cls.__name__)
        if registry:
            for model_class in registry.plugin_registry:
                is_detected = True
                if hasattr(model_class, "detect"):
                    is_detected = model_class.detect()

                if is_detected:
                    detected.append(model_class())

        # 2. Find ModelInstance objects in the imported modules
        import sys

        for mod_name, module in sys.modules.items():
            if mod_name.startswith(package_name) and mod_name != __name__:
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if isinstance(attr, ModelInstance):
                        # Avoid duplicates
                        if attr not in detected:
                            # Run detection to determine if model should be enabled
                            is_enabled = attr.run_detect()
                            attr.enabled = is_enabled
                            if is_enabled:
                                detected.append(attr)

        return detected

    @abstractmethod
    def build(self) -> list[LazyInit]:
        """
        Build stage: Create lazy initializers for model fields.
        """
        pass

    def configure_algorithm(self, algorithm: Any) -> None:
        """
        Configure stage: Modify algorithm behavior for this model.
        """
        pass

    @property
    def operations(self) -> OperationCollection:
        """
        Get operations to insert into execution graph.
        """
        from foamadapter.framework.decorator import decorated_member_functions

        funcs = decorated_member_functions(self)
        ops = OperationCollection()
        for func in funcs:
            op = Operation.create_SeqOp(func)
            ops.add(op)
        return ops
