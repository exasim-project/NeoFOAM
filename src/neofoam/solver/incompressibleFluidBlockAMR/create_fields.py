# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""3-stage initialization for the incompressibleFluidBlockAMR solver.

LOAD    → detect the projection algorithm + any optional models; instantiate the
          reused ``solutionLoop`` / ``fieldWriter`` core Models.
RESOLVE → wire optional-model dependencies via ConfigContext.
BUILD   → emit lazy InitSteps that construct the ``blockamr`` mesh + the
          projection state (fields + equations), register ``U`` / ``p`` / ``phi``
          + the state into the Context, and inject the blockAMR loop/writer
          backends through the framework seams.

Unlike the pybFoam / NeoN solvers there is no ``Foam::Time`` / OpenFOAM mesh: the
framework core Models are pure-Python (``LoopState`` + ``FieldWriter``), and the
block-structured DSL (``blockamr``) carries the physics. The mesh and the
validated dict configs are init-only resources (leading underscore keeps them off
the Context); the projection state is re-exposed as ``models.projection_state`` so
the ``project`` operation injects it by name.
"""

from pathlib import Path
from typing import Any, Optional

from neofoam.algorithms.field_writer.field_writer import fieldWriter
from neofoam.algorithms.solution_loop.solution_loop import solutionLoop
from neofoam.framework.context import Context
from neofoam.framework.initialization import (
    ConfigContext,
    InitializerBuilder,
    InitStep,
    LoadResult,
    StagedInitRunner,
    StagedInitSpec,
    lazy,
    model as init_model,
)
from neofoam.framework.model import ModelRuntime, bind_owned_interfaces

from .configs import (
    BlockAMRSolutionConfig,
    ControlDictConfig,
    FvSchemesConfig,
    MeshDictConfig,
    PSolutionConfig,
    USolutionConfig,
)
from .models.blockamr_backend import loop_backend_steps, writer_backend_steps
from .models.incompressibleFluidBlockAMRModel import incompressibleFluidBlockAMRModel
from .models.mesh_factory import build_mesh
from .models.projection.base import ProjectionAlgorithm


def _optional_models_by_name(optional_models: list[Any]) -> dict[str, Any]:
    """Map each active optional model to its model name."""
    return {m.name: m for m in optional_models}


def create_init(case_dir: Optional[Path] = None) -> StagedInitRunner:
    """Build a fresh :class:`StagedInitRunner` for incompressibleFluidBlockAMR."""
    spec_builder = StagedInitSpec.build("incompressibleFluidBlockAMR")
    resolved_case_dir = case_dir or Path(".")

    @spec_builder.load
    def load_config() -> LoadResult:
        projection_model = ProjectionAlgorithm.detect_and_create(resolved_case_dir)
        optional_models = incompressibleFluidBlockAMRModel.detect_models(
            resolved_case_dir
        )

        # solutionLoop (advances time) + fieldWriter (persists fields) are the
        # reused framework core Models; loaded here so their controlDict is
        # validated up front. Both are composed by the execution graph.
        solution_loop_model = solutionLoop.instantiate(resolved_case_dir, "main")
        field_writer_model = fieldWriter.instantiate(resolved_case_dir, "main")

        core_models: list[Any] = [
            projection_model,
            solution_loop_model,
            field_writer_model,
            ControlDictConfig.load(case_dir=resolved_case_dir, validate=False),
        ]

        return LoadResult(core_models=core_models, optional_models=optional_models)

    @spec_builder.resolve
    def resolve_models(config: ConfigContext) -> None:
        for opt in runner.optional_models:
            opt.run_resolve(config)

    @spec_builder.build
    def build_lazy(
        core_models: list[Any], optional_models: list[Any]
    ) -> list[InitStep]:
        projection_model = core_models[0]

        def _by_spec(spec_name: str) -> Any:
            return next(
                m
                for m in core_models
                if isinstance(m, ModelRuntime) and m.spec.name == spec_name
            )

        solution_loop_model = _by_spec("solutionLoop")
        field_writer_model = _by_spec("fieldWriter")

        # --- init-only resources: the validated dict configs + the mesh. The
        # engine (built by chorinProjection.@build) reads them by name. ---
        def load_mesh_cfg(_ctx: dict[str, Any]) -> MeshDictConfig:
            return MeshDictConfig.load(case_dir=resolved_case_dir)

        def load_solution_cfg(_ctx: dict[str, Any]) -> BlockAMRSolutionConfig:
            return BlockAMRSolutionConfig.load(case_dir=resolved_case_dir)

        def load_control_cfg(_ctx: dict[str, Any]) -> ControlDictConfig:
            return ControlDictConfig.load(case_dir=resolved_case_dir)

        def load_fvschemes_cfg(_ctx: dict[str, Any]) -> FvSchemesConfig:
            return FvSchemesConfig.load(case_dir=resolved_case_dir)

        def load_sol_u_cfg(_ctx: dict[str, Any]) -> USolutionConfig:
            return USolutionConfig.load(case_dir=resolved_case_dir)

        def load_sol_p_cfg(_ctx: dict[str, Any]) -> PSolutionConfig:
            return PSolutionConfig.load(case_dir=resolved_case_dir)

        def make_mesh(ctx: dict[str, Any]) -> Any:
            return build_mesh(ctx["_mesh_cfg"])

        builder = InitializerBuilder()
        builder.add(lazy("_mesh_cfg", load_mesh_cfg))
        builder.add(lazy("_solution_cfg", load_solution_cfg))
        builder.add(lazy("_control_cfg", load_control_cfg))
        builder.add(lazy("_fvschemes_cfg", load_fvschemes_cfg))
        builder.add(lazy("_sol_u_cfg", load_sol_u_cfg))
        builder.add(lazy("_sol_p_cfg", load_sol_p_cfg))
        builder.add(lazy("_blockamr_mesh", make_mesh, depends_on=["_mesh_cfg"]))

        # solutionLoop + fieldWriter core Models; the blockAMR touch-points
        # (BlockAMRTimeBackend, PlotfileWriteHook, logger, reporter) are injected
        # by the *_backend_steps through the framework seams.
        builder.add_core_models(
            [
                ("solution_loop_model", solution_loop_model),
                ("field_writer_model", field_writer_model),
            ]
        )
        builder.extend(loop_backend_steps())
        builder.extend(
            writer_backend_steps(ControlDictConfig.load(case_dir=resolved_case_dir))
        )

        # The projection ModelSpec is used as its own runtime (no instantiate):
        # its @build constructs the engine and registers U/p/phi + the engine.
        builder.add_core_models([("projection", projection_model)])
        if projection_model._build_func is not None:
            builder.extend(projection_model._build_func(projection_model))

        builder.add_optional_models(optional_models)
        for name, opt in _optional_models_by_name(optional_models).items():
            builder.add_model(name, opt)

        # Bind the solutionLoop runtime's owned interfaces (timeStepConstraint /
        # loopCondition) to the active contributing optional models. Bound
        # against an EMPTY Context — capturing live objects here would create a
        # GC reference cycle; the live fold re-binds at call time.
        def wire_loop_interfaces(_work: dict[str, Any]) -> Any:
            return bind_owned_interfaces(
                solution_loop_model, optional_models, Context(fields={}, models={})
            )

        builder.add(
            init_model(
                "solutionLoop",
                wire_loop_interfaces,
                depends_on=["models.solution_loop"],
            )
        )

        return builder.build()

    runner = StagedInitRunner(spec_builder.finalize())
    return runner
