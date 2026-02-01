# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
DummyInit - 3-stage initialization for DummySolver.

Implements explicit 3-stage initialization pattern:
- LOAD: Load configuration from files
- RESOLVE: Connect models and validate dependencies
- BUILD: Create lazy initializers for runtime objects

This follows the IncompressibleFluidInitializer pattern.
"""

import yaml
import inspect
from typing import Any, Type, TypeVar
from pathlib import Path

from pydantic import BaseModel, Field, PrivateAttr

from foamadapter.framework.initialization import (
    StagedInit,
    LoadResult,
    ValidationError,
    ConfigContext,
    InitializerBuilder,
)
from foamadapter.framework.initialization.lazy_init import LazyInit

# Import relative to test directory
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

T = TypeVar("T", bound="BaseConfig")


class BaseConfig(BaseModel):
    """Base class for configurations with a load method."""

    @classmethod
    def load(cls: Type[T], path: Path) -> T:
        """Load configuration from a YAML file."""
        if not path.exists():
            return cls()
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)


class SolverConfig(BaseConfig):
    """Main solver configuration."""

    name: str = "DummySolver"
    param1: float
    param2: float
    dt: float
    endTime: float
    parameters: dict[str, Any]


class MeshConfig(BaseConfig):
    """Mesh configuration."""

    name: str = "domain"
    nPoints: int


class Model1Config(BaseConfig):
    """Configuration for Model1 (generic physics model)."""

    enabled: bool = True
    prop1: float
    prop2: float
    parameters: dict[str, Any] = Field(default_factory=dict)


class CoreModel2(BaseModel):
    """An additional core model."""

    name: str = "CoreModel2"
    status: str = "active"


class DummyAlgorithm(BaseModel):
    param1: float
    _use_model1: bool = PrivateAttr(default=False)
    _iteration_count: int = PrivateAttr(default=0)

    def solve(self) -> bool:
        """Run one algorithm iteration."""
        self._iteration_count += 1
        return self._iteration_count < 3  # Stop after 3 iterations


# Create the StagedInit instance
init = StagedInit("DummySolver")


def create_init() -> StagedInit:
    """
    Factory function for dependency injection.

    Returns the global StagedInit instance.
    """
    return init


@init.load
def load_config() -> LoadResult:
    """
    LOAD stage: Load configuration from files.

    Returns LoadResult with core_models and optional_models.
    """
    from .models import DummyModel

    # Load configurations
    config_dir = Path(__file__).parent / "configs"

    solver_config = SolverConfig.load(config_dir / "solver_config.yaml")
    mesh_config = MeshConfig.load(config_dir / "mesh_config.yaml")

    # Create core models
    algorithm = DummyAlgorithm(param1=solver_config.param1)
    core_model2 = CoreModel2()

    # Store configs in algorithm for later access
    algorithm._solver_config = solver_config
    algorithm._mesh_config = mesh_config

    # Detect optional models
    optional_models = DummyModel.detect_models()

    # Run LOAD on optional models
    for model in optional_models:
        model.run_load()

    return LoadResult(
        core_models=[algorithm, core_model2], optional_models=optional_models
    )

    return LoadResult(
        core_models=[algorithm, core_model2], optional_models=optional_models
    )


@init.validate_load
def validate_load_stage(core_models: list) -> list[ValidationError]:
    """Validate configuration after LOAD stage."""
    errors = []
    if not core_models or len(core_models) < 2:
        errors.append(ValidationError("core_models", "Missing core models"))
        return errors

    algorithm = core_models[0]
    if not hasattr(algorithm, "_solver_config") or algorithm._solver_config is None:
        errors.append(
            ValidationError("solver_config", "Failed to read solver configuration")
        )
    if not hasattr(algorithm, "_mesh_config") or algorithm._mesh_config is None:
        errors.append(
            ValidationError("mesh_config", "Failed to read mesh configuration")
        )

    return errors


@init.resolve
def resolve_models(_: list, optional_models: list, config: ConfigContext) -> None:
    """RESOLVE stage: Connect models and validate dependencies."""
    for model in optional_models:
        if hasattr(model, "resolve"):
            model.resolve(config)


@init.validate_resolve
def validate_resolve_stage(
    optional_models: list, config: ConfigContext | None = None
) -> list[ValidationError]:
    """Validate model connections after RESOLVE stage."""
    errors = []
    for model in optional_models:
        if hasattr(model, "validate_stage"):
            errors.extend(model.validate_stage(config))
    return errors


def _normalize_lazy_init(li: LazyInit) -> LazyInit:
    """Normalize field name and wrap initializer for dict support."""
    if not li.name.startswith(("fields.", "models.", "operators.")):
        li.name = f"fields.{li.name}"

    orig_func = li.initializer
    if orig_func is None:
        return li

    # Check if function takes context parameter
    sig = inspect.signature(orig_func)
    takes_ctx = len(sig.parameters) > 0

    def wrapper(ctx=None):
        res = orig_func(ctx) if takes_ctx and ctx else orig_func()
        return res.get("value", res) if isinstance(res, dict) else res

    li.initializer = wrapper
    return li


@init.build
def build_lazy(core_models: list, optional_models: list) -> list[LazyInit]:
    """
    BUILD stage: Create lazy initializers for runtime objects.

    Args:
        core_models: List of core models from load stage
        optional_models: List of optional models from load stage

    Returns:
        List of LazyInit objects.
    """
    algorithm, core_model2 = core_models
    solver_config = algorithm._solver_config
    mesh_config = algorithm._mesh_config

    builder = InitializerBuilder()

    # Resources and configuration
    builder.add_resource("mesh", mesh_config.model_dump())
    builder.add_resource("domain", mesh_config.model_dump())
    builder.add_resource("config", solver_config)

    # Core models - adds models and calls their build() methods if available
    builder.add_core_models(
        [
            ("algorithm", algorithm),
            ("core2", core_model2),
        ]
    )
    builder.add_model("config", solver_config.model_dump())

    # Solver-specific fields
    builder.add_field("field1", depends_on=["mesh"], value=1.0)
    builder.add_field("field2", depends_on=["mesh"], value=101325.0)
    builder.add_field(
        "field3",
        depends_on=["fields.field1"],
        value=lambda ctx: ctx["fields.field1"] * 0.01,
    )

    # Optional models - configure algorithm then add their build() LazyInits
    for model in optional_models:
        if hasattr(model, "configure_algorithm") and algorithm:
            model.configure_algorithm(algorithm)

    builder.add_optional_models(optional_models)

    # Metadata - store optional models reference
    builder.add_model("optional_models", optional_models)

    return builder.build()
