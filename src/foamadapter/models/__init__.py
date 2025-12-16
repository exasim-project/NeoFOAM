# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
Model Registry and Discovery System

Provides registration and discovery of pluggable solver models. Models register
themselves using the @register_model decorator, enabling configuration-driven
and programmatic composition of solver capabilities.

Example:
    # Define and register a model
    @register_model("buoyancy")
    @Model
    class BuoyancyModel(BaseModel):
        name: str = "buoyancy"
        beta: float = 3e-3
        ...

    # Create via registry
    buoyancy = get_model("buoyancy", beta=2e-3)

    # Add to solver
    solver = IncompressibleFluid(argv=["case"])
    solver.add_model(buoyancy)

    # List available models
    available = list_models()  # ["buoyancy", "PIMPLE", "SIMPLE", ...]
"""

from typing import Any, Callable

from foamadapter.framework.model_protocol import SolverModel


# Global registry of model types
MODEL_REGISTRY: dict[str, type[SolverModel]] = {}


def register_model(name: str) -> Callable[[type[SolverModel]], type[SolverModel]]:
    """
    Decorator to register a model type in the global registry.

    Registered models can be instantiated via get_model() by name,
    enabling configuration-driven composition and plugin discovery.

    Args:
        name: Unique identifier for this model type

    Returns:
        Decorator function that registers the class

    Example:
        @register_model("turbulence")
        @Model
        class TurbulenceModel(BaseModel):
            name: str = "turbulence"
            model_type: str = "kEpsilon"
            ...

        # Later, create instance
        turb = get_model("turbulence", model_type="kOmegaSST")

    Raises:
        ValueError: If model name already registered (duplicate)
    """

    def decorator(cls: type[SolverModel]) -> type[SolverModel]:
        if name in MODEL_REGISTRY:
            raise ValueError(
                f"Model '{name}' already registered. "
                f"Existing: {MODEL_REGISTRY[name]}, New: {cls}"
            )
        MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def get_model(name: str, **kwargs: Any) -> SolverModel:
    """
    Create a model instance by name.

    Factory function that looks up the model type in the registry
    and instantiates it with the provided keyword arguments.

    Args:
        name: Registered model name
        **kwargs: Arguments passed to model constructor

    Returns:
        Instantiated model

    Example:
        # Create buoyancy model with custom beta
        buoyancy = get_model("buoyancy", beta=3.5e-3, TRef=290.0)

        # Create algorithm model
        pimple = get_model("PIMPLE", pRefCell=0, pRefValue=0.0)

    Raises:
        ValueError: If model name not found in registry
    """
    if name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(f"Unknown model: '{name}'. Available models: {available}")

    model_class = MODEL_REGISTRY[name]
    return model_class(**kwargs)


def list_models() -> list[str]:
    """
    List all registered model names.

    Returns:
        Sorted list of model names available in the registry

    Example:
        >>> list_models()
        ['PIMPLE', 'SIMPLE', 'buoyancy', 'radiation', 'turbulence']
    """
    return sorted(MODEL_REGISTRY.keys())


def is_registered(name: str) -> bool:
    """
    Check if a model name is registered.

    Args:
        name: Model name to check

    Returns:
        True if model is registered, False otherwise

    Example:
        if is_registered("buoyancy"):
            buoyancy = get_model("buoyancy")
    """
    return name in MODEL_REGISTRY


# Export public API
__all__ = [
    "MODEL_REGISTRY",
    "SolverModel",
    "register_model",
    "get_model",
    "list_models",
    "is_registered",
]
