# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
DummyInit — 3-stage initialization for DummySolver.

Implements the explicit 3-stage pattern:
  LOAD    → load configs + instantiate optional ModelRuntimes
  RESOLVE → wire inter-model dependencies via ConfigContext
  BUILD   → produce lazy InitSteps for execute_initialization
"""

from typing import Any, Optional
from pathlib import Path

from pydantic import Field, PrivateAttr

from neofoam.framework.initialization import (
    ConfigContext,
    InitializerBuilder,
    InitStep,
    LoadResult,
    StagedInitRunner,
    StagedInitSpec,
)
from neofoam.framework.model import ModelRuntime
from neofoam.io import BaseConfig, YAML, IOStrategy

from .models.dummy_model import DummyModelInterface


@IOStrategy(YAML("solver_config.yaml"))
class SolverConfig(BaseConfig):
    name: str = "DummySolver"
    param1: float = Field(gt=0, description="Parameter 1 must be positive")
    param2: float
    dt: float
    endTime: float = Field(gt=0, description="End time must be positive")
    parameters: dict[str, Any]


@IOStrategy(YAML("mesh_config.yaml"))
class MeshConfig(BaseConfig):
    name: str = "domain"
    nPoints: int = Field(gt=0, description="Number of points must be positive")


@IOStrategy(YAML("core_model2_config.yaml"))
class CoreModel2(BaseConfig):
    name: str = "CoreModel2"
    status: str = Field(
        description="Status of the model",
        pattern="^(active|inactive)$",
    )


@IOStrategy(YAML("algorithm_config.yaml"))
class DummyAlgorithm(BaseConfig):
    param1: float = Field(gt=0, description="A parameter for the algorithm")
    _use_model1: bool = PrivateAttr(default=False)
    _iteration_count: int = PrivateAttr(default=0)

    def solve(self) -> bool:
        self._iteration_count += 1
        return self._iteration_count < 3


def create_init(case_dir: Optional[Path] = None) -> StagedInitRunner:
    """Build a fresh StagedInitRunner for DummySolver.

    Closures over ``runner`` give the resolve stage access to the live
    optional_models list (populated by the load stage on the same runner).
    """
    spec_builder = StagedInitSpec.build("DummySolver")
    resolved_case_dir = case_dir or Path(__file__).parent / "configs"

    @spec_builder.load
    def load_config() -> LoadResult:
        algorithm = DummyAlgorithm.load(case_dir=resolved_case_dir, validate=False)
        core_model2 = CoreModel2.load(case_dir=resolved_case_dir, validate=False)
        solver_cfg = SolverConfig.load(case_dir=resolved_case_dir, validate=False)
        mesh_cfg = MeshConfig.load(case_dir=resolved_case_dir, validate=False)

        optional_models: list[ModelRuntime] = [
            spec.instantiate(case_dir=resolved_case_dir, instance_id=spec.name)
            for spec in DummyModelInterface.detect_specs()
        ]

        return LoadResult(
            core_models=[algorithm, core_model2, solver_cfg, mesh_cfg],
            optional_models=optional_models,
        )

    @spec_builder.resolve
    def resolve_models(config: ConfigContext) -> None:
        for runtime in runner.optional_models:
            runtime.run_resolve(config)

    @spec_builder.build
    def build_lazy(
        core_models: list[Any], optional_models: list[Any]
    ) -> list[InitStep]:
        algorithm = next(m for m in core_models if isinstance(m, DummyAlgorithm))
        core_model2 = next(m for m in core_models if isinstance(m, CoreModel2))
        solver_cfg = next(m for m in core_models if isinstance(m, SolverConfig))
        mesh_cfg = next(m for m in core_models if isinstance(m, MeshConfig))

        builder = InitializerBuilder()
        builder.add_resource("mesh", mesh_cfg.model_dump())
        builder.add_resource("domain", mesh_cfg.model_dump())
        builder.add_core_models([("algorithm", algorithm), ("core2", core_model2)])
        builder.add_model("config", solver_cfg.model_dump())
        builder.add_field("field1", depends_on=["mesh"], value=1.0)
        builder.add_field("field2", depends_on=["mesh"], value=101325.0)
        builder.add_field(
            "field3",
            depends_on=["fields.field1"],
            value=lambda ctx: ctx["fields.field1"] * 0.01,
        )

        for runtime in optional_models:
            builder.extend(runtime.run_build())

        builder.add_model("optional_models", optional_models)
        return builder.build()

    runner = StagedInitRunner(spec_builder.finalize())
    return runner
