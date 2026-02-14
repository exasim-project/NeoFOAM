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

from typing import Any, TypeVar
from pathlib import Path

from pydantic import Field, PrivateAttr

from neofoam.framework.initialization import (
    StagedInit,
    LoadResult,
    ConfigContext,
    InitializerBuilder,
    InitStep,
)
from neofoam.io import (
    BaseConfig,
    YAML,
    IOStrategy,
)
from .models.dummy_model import DummyModelInterface


T = TypeVar("T", bound="BaseConfig")


@IOStrategy(YAML("solver_config.yaml"))
class SolverConfig(BaseConfig):
    """Main solver configuration."""

    name: str = "DummySolver"
    param1: float = Field(gt=0, description="Parameter 1 must be positive")
    param2: float
    dt: float
    endTime: float = Field(gt=0, description="End time must be positive")
    parameters: dict[str, Any]


@IOStrategy(YAML("mesh_config.yaml"))
class MeshConfig(BaseConfig):
    """Mesh configuration."""

    name: str = "domain"
    nPoints: int = Field(gt=0, description="Number of points must be positive")


@IOStrategy(YAML("core_model2_config.yaml"))
class CoreModel2(BaseConfig):
    """An additional core model."""

    name: str = "CoreModel2"
    status: str = Field(
        description="Status of the model, e.g. active/inactive",
        pattern="^(active|inactive)$",
    )


@IOStrategy(YAML("algorithm_config.yaml"))
class DummyAlgorithm(BaseConfig):
    param1: float = Field(gt=0, description="A parameter for the algorithm")
    _use_model1: bool = PrivateAttr(default=False)
    _iteration_count: int = PrivateAttr(default=0)

    def solve(self) -> bool:
        """Run one algorithm iteration."""
        self._iteration_count += 1
        return self._iteration_count < 3  # Stop after 3 iterations


# Create the StagedInit instance
init = StagedInit("DummySolver")


def create_init(case_dir: Path = None) -> StagedInit:
    """
    Factory function for dependency injection.

    Args:
        case_dir: Optional path to configuration directory. If None, uses default 'configs' subdirectory.

    Returns the global StagedInit instance with case_dir stored for load stage.
    """
    # Store case_dir for use in load_config
    init._case_dir = case_dir
    return init


@init.load
def load_config() -> LoadResult:
    """
    LOAD stage: Load configuration from files using ModelInputDefinition.

    Returns LoadResult with core_models and optional_models.
    """
    # Use stored case_dir or default to configs subdirectory
    if hasattr(init, "_case_dir") and init._case_dir is not None:
        case_dir = init._case_dir
    else:
        case_dir = Path(__file__).parent / "configs"

    # Load configurations using ModelInputDefinition paths
    solver_config = SolverConfig.load(case_dir=case_dir)
    mesh_config = MeshConfig.load(case_dir=case_dir)
    algorithm = DummyAlgorithm.load(case_dir=case_dir)
    core_model2 = CoreModel2.load(case_dir=case_dir)

    # Detect optional models
    optional_models = DummyModelInterface.detect_models()

    # Run LOAD on optional models
    for model in optional_models:
        model.run_load(case_dir=case_dir)

    return LoadResult(
        core_models=[algorithm, core_model2, solver_config, mesh_config],
        optional_models=optional_models,
    )


@init.resolve
def resolve_models(config: ConfigContext) -> None:
    """RESOLVE stage: Connect models and validate dependencies."""
    for model in init.optional_models:
        model.resolve(config)


@init.build
def build_lazy(core_models: list, optional_models: list) -> list[InitStep]:
    """
    BUILD stage: Create lazy initializers for runtime objects.

    Args:
        core_models: List of core models from load stage
        optional_models: List of optional models from load stage

    Returns:
        List of LazyInit objects.
    """
    # Extract models by type
    algorithm = next((m for m in core_models if isinstance(m, DummyAlgorithm)), None)
    core_model2 = next((m for m in core_models if isinstance(m, CoreModel2)), None)
    solver_config = next((m for m in core_models if isinstance(m, SolverConfig)), None)
    mesh_config = next((m for m in core_models if isinstance(m, MeshConfig)), None)

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
