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
from dataclasses import dataclass, field as dc_field
from typing import Any, Type, TypeVar, Protocol, runtime_checkable, Iterable
from pathlib import Path
from functools import wraps

from pydantic import BaseModel, Field, PrivateAttr

from foamadapter.framework.context import Context
from foamadapter.framework.initialization import (
    StagedInit,
    LoadResult,
    ValidationError,
    ConfigContext,
    field,
    lazy,
)
from foamadapter.framework.initialization.lazy_init import LazyInit

# Import relative to test directory
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

T = TypeVar("T", bound="BaseConfig")


@runtime_checkable
class OptionalModel(Protocol):
    """Protocol for optional models supporting 3-stage initialization."""

    def build(self) -> list[LazyInit]:
        """Create lazy initializers for model fields."""
        ...

    def configure_algorithm(self, algorithm: Any) -> None:
        """Configure solver algorithm for this model."""
        ...


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


@dataclass
class InitializationData:
    """State container for initialization stages."""

    # Core models - always present
    algorithm: Any = None
    core_model2: Any = None

    # Configuration
    solver_config: SolverConfig | None = None
    mesh_config: MeshConfig | None = None

    # Optional models - physics extensions
    optional_models: list = dc_field(default_factory=list)

    # Validation state
    load_validated: bool = False
    resolve_validated: bool = False

    @property
    def core_models(self) -> list[Any]:
        """Core models that define solver structure."""
        return [m for m in [self.algorithm, self.core_model2] if m is not None]


# Create the StagedInit instance
init = StagedInit("DummySolver")
init.data = InitializationData()

# Detect optional models for global init
try:
    from framework.dummy_solver.models import DummyModel

    init.data.optional_models = DummyModel.detect_models()
except ImportError:
    init.data.optional_models = []


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
    # Load configurations
    config_dir = Path(__file__).parent / "configs"

    init.data.solver_config = SolverConfig.load(config_dir / "solver_config.yaml")
    init.data.mesh_config = MeshConfig.load(config_dir / "mesh_config.yaml")

    # Create core models
    algorithm = DummyAlgorithm(param1=init.data.solver_config.param1)
    core_model2 = CoreModel2()

    # Store in init.data for later access
    init.data.algorithm = algorithm
    init.data.core_model2 = core_model2

    # Detect optional models
    from framework.dummy_solver.models import DummyModel

    optional_models = DummyModel.detect_models()
    init.data.optional_models = optional_models

    # Run LOAD on optional models
    for model in optional_models:
        if hasattr(model, "load"):
            model.load()

    return LoadResult(
        core_models=[algorithm, core_model2], optional_models=optional_models
    )


@init.validate_load
def validate_load_stage() -> list[ValidationError]:
    """Validate configuration after LOAD stage."""
    errors = []

    if init.data.algorithm is None:
        errors.append(ValidationError("algorithm", "No algorithm configured"))

    if init.data.solver_config is None:
        errors.append(
            ValidationError("solver_config", "Failed to read solver configuration")
        )

    if init.data.mesh_config is None:
        errors.append(
            ValidationError("mesh_config", "Failed to read mesh configuration")
        )

    init.data.load_validated = len([e for e in errors if e.severity == "error"]) == 0
    return errors


@init.resolve
def resolve_models(
    core_models: list, optional_models: list, config: ConfigContext
) -> None:
    """
    RESOLVE stage: Connect models and validate dependencies.

    Args:
        core_models: List of core models from load stage
        optional_models: List of optional models from load stage
        config: ConfigContext with registered models
    """
    # Resolve optional models
    for model in optional_models:
        if hasattr(model, "resolve"):
            model.resolve(config)


@init.validate_resolve
def validate_resolve_stage(
    config: ConfigContext | None = None,
) -> list[ValidationError]:
    """Validate model connections after RESOLVE stage."""
    errors = []

    # Check optional model connections
    for model in init.data.optional_models:
        if hasattr(model, "validate_stage"):
            model_errors = model.validate_stage(config)
            errors.extend(model_errors)

    init.data.resolve_validated = len([e for e in errors if e.severity == "error"]) == 0
    return errors


def _normalize_lazy_init(li: LazyInit) -> LazyInit:
    """Normalize field name and wrap initializer for dict support."""
    if not li.name.startswith(("fields.", "models.", "operators.")):
        li.name = f"fields.{li.name}"

    orig_func = li.initializer
    if orig_func is None:
        return li

    sig = inspect.signature(orig_func)
    takes_ctx = len(sig.parameters) > 0

    @wraps(orig_func)
    def wrapper(ctx=None):
        res = orig_func(ctx) if takes_ctx and ctx else orig_func()
        return res.get("value", res) if isinstance(res, dict) else res

    li.initializer = wrapper
    return li


@init.build
def build_lazy(core_models: list, optional_models: list) -> list[Any]:
    """
    BUILD stage: Create lazy initializers for runtime objects.

    Args:
        core_models: List of core models from load stage
        optional_models: List of optional models from load stage

    Returns:
        List of LazyInit objects.
    """
    from foamadapter.framework.initialization import model as model_lazy

    algorithm, core_model2 = core_models
    initializers = []

    # 1. Mesh and configuration
    initializers.extend(
        [
            lazy("mesh", create=lambda ctx: init.data.mesh_config.model_dump()),
            lazy("domain", create=lambda ctx: init.data.mesh_config.model_dump()),
            lazy("solver_config", create=lambda ctx: init.data.solver_config),
        ]
    )

    # 2. Core models and their configuration
    initializers.extend(
        [
            model_lazy("algorithm", create=lambda ctx: algorithm),
            model_lazy("core2", create=lambda ctx: core_model2),
            model_lazy(
                "config", create=lambda ctx: init.data.solver_config.model_dump()
            ),
        ]
    )

    # 3. Basic Fields
    initializers.extend(
        [
            field("field1", depends_on=["mesh"], create=lambda ctx: 1.0),
            field("field2", depends_on=["mesh"], create=lambda ctx: 101325.0),
            field(
                "field3",
                depends_on=["fields.field1"],
                create=lambda ctx: ctx["fields.field1"] * 0.01,
            ),
        ]
    )

    # 4. Optional models (Physics extensions)
    for model in optional_models:
        if isinstance(model, OptionalModel):
            initializers.extend(_normalize_lazy_init(li) for li in model.build())
            if algorithm:
                model.configure_algorithm(algorithm)

    # 5. Metadata
    initializers.append(
        model_lazy("optional_models", create=lambda ctx: optional_models)
    )

    return initializers
