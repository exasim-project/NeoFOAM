# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Staged initialization of incompressibleVoFNeoN."""

from pathlib import Path
from typing import Any, Optional

import pybFoam as pyf

from neofoam import neofoam_bindings as nfb
from neofoam.framework.initialization import (
    ConfigContext,
    InitializerBuilder,
    InitStep,
    LoadResult,
    StagedInitRunner,
    StagedInitSpec,
    lazy,
)
from neofoam.framework.initialization import model as init_model
from neofoam.framework.model import ModelRuntime
from neofoam.solver.incompressibleFluidNeoN.models.field_writer import fieldWriter
from neofoam.solver.incompressibleFluidNeoN.models.solution_loop import (
    neon_loop_backend_steps,
    solutionLoop,
)
from neofoam.solver.incompressibleVoF.configs import ControlDictConfig
from neofoam.solver.neon_runtime import requested_executor

from .models.alpha_advection import advectionModelNeoN
from .models.field_writer import neon_vof_writer_backend_steps
from .models.pressure_velocity import PressureVelocityAlgorithmVoFNeoN
from .models.time_step import adaptiveTimeStepNeoN

# The linear-solver dicts mapped to Ginkgo; the alpha predictor's is mapped by NeoNMulesOps.
_MAPPED_SOLVER_DICTS = ("p_rgh", "p_rghFinal", "U", "UFinal")

# interFoam's stress divergence key and the one NeoN's viscous-stress operator looks up.
_CASE_STRESS_KEY = "div(((rho*nuEff)*dev2(T(grad(U)))))"
_NEON_STRESS_KEY = "div((nuEff*dev2(T(grad(U)))))"


def _time_step_models(case_dir: Path) -> list[ModelRuntime]:
    """The adaptive time-step model when ``adjustTimeStep yes``, else none (fixed deltaT)."""
    if not ControlDictConfig.load(case_dir=case_dir, validate=False).adjustTimeStep:
        return []
    return [adaptiveTimeStepNeoN.instantiate(case_dir, adaptiveTimeStepNeoN.name)]


def _map_dictionaries(runtime: Any) -> None:
    """Map fvSchemes/fvSolution to NeoN's names, once, before any operator reads them."""
    runtime.fv_schemes_dict = nfb.map_fv_schemes(runtime.fv_schemes_dict)
    div_schemes = runtime.fv_schemes_dict.subDict("divSchemes")
    if div_schemes.contains(_CASE_STRESS_KEY):
        div_schemes.insert_token_list(
            _NEON_STRESS_KEY, div_schemes.get_token_list(_CASE_STRESS_KEY)
        )
    solvers = runtime.fv_solution_dict.subDict("solvers")
    for name in _MAPPED_SOLVER_DICTS:
        nfb.map_solver_settings(solvers, name)


def _runtime_steps(argv: list[str]) -> list[InitStep]:
    """``Foam::Time`` and the NeoN ``RunTime`` adapter, both init-only (kept off the Context)."""

    def create_neon_runtime(ctx: dict[str, Any]) -> Any:
        runtime = nfb.create_adapter_run_time(ctx["_foam_time"], requested_executor())
        _map_dictionaries(runtime)
        return runtime

    return [
        lazy("_arg_list", lambda _ctx: pyf.argList(argv)),
        # Foam::Time keeps a raw reference to the argList: NeoNTimeSync pins both.
        lazy("_foam_time", lambda ctx: pyf.Time(ctx["_arg_list"]), depends_on=["_arg_list"]),
        lazy("_neon_runtime", create_neon_runtime, depends_on=["_foam_time"]),
        init_model("neon_runtime", lambda ctx: ctx["_neon_runtime"], depends_on=["_neon_runtime"]),
    ]


def create_init(case_dir: Optional[Path] = None) -> StagedInitRunner:
    """Build a fresh :class:`StagedInitRunner` for incompressibleVoFNeoN.

    Use it as the solver's ``Depends`` initializer; ``incompressibleVoF`` has its own for
    pybFoam. Each runner holds one run's state, so build one per solver instance.

    Example::

        def initialize(init: Annotated[StagedInitRunner, Depends(create_init)]) -> Context: ...
    """
    spec_builder = StagedInitSpec.build("incompressibleVoFNeoN")
    resolved_case_dir = case_dir or Path(".")

    @spec_builder.load
    def load_config() -> LoadResult:
        core_models: list[Any] = [
            advectionModelNeoN.detect_and_create(),
            PressureVelocityAlgorithmVoFNeoN.detect_and_create(),
            solutionLoop.instantiate(resolved_case_dir, "main"),
            fieldWriter.instantiate(resolved_case_dir, "main"),
        ]
        return LoadResult(
            core_models=core_models, optional_models=_time_step_models(resolved_case_dir)
        )

    @spec_builder.resolve
    def resolve_models(config: ConfigContext) -> None:
        return None

    @spec_builder.build
    def build_lazy(core_models: list[Any], optional_models: list[Any]) -> list[InitStep]:
        alpha_model, pressure_model = core_models[0], core_models[1]

        def by_spec(spec_name: str) -> Any:
            return next(
                member
                for member in core_models
                if isinstance(member, ModelRuntime) and member.spec.name == spec_name
            )

        solution_loop_model = by_spec("solutionLoop")
        builder = InitializerBuilder()
        builder.extend(_runtime_steps(runner.argv))
        builder.add_core_models(
            [
                ("solution_loop_model", solution_loop_model),
                ("field_writer_model", by_spec("fieldWriter")),
            ]
        )
        builder.extend(neon_loop_backend_steps())
        builder.extend(neon_vof_writer_backend_steps())

        # The bare specs only register; their @build steps carry the NeoN field reads.
        builder.add_core_models(
            [("alpha_advection", alpha_model), ("pressure_velocity", pressure_model)]
        )
        for spec in (alpha_model, pressure_model):
            builder.extend(spec.build_steps())
        # In ctx.models by name: the solution-loop limits dispatch to active models only.
        for optional_model in optional_models:
            builder.add_model(optional_model.name, optional_model)

        builder.add(
            init_model(
                "solutionLoop",
                lambda _ctx: solution_loop_model,
                depends_on=["models.solution_loop"],
            )
        )
        return builder.build()

    runner = StagedInitRunner(spec_builder.finalize())
    return runner
