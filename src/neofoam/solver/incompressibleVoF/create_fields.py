# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""3-stage initialization for the incompressibleVoF solver.

LOAD    → detect the PIMPLE algorithm and any optional models
RESOLVE → wire optional-model dependencies via ConfigContext
BUILD   → emit lazy InitSteps for runtime/mesh, the alpha-advection + PIMPLE
          core models (each owns the fields it registers) and the two-phase
          turbulence model.

Ported from the ``StagedInit`` / ``ModelInstance`` API to the
``StagedInitRunner`` / ``StagedInitSpec`` + ``ModelSpec`` API in
``stack/python_arch`` (mirrors ``incompressibleFluid.create_fields``).
"""

from pathlib import Path
from typing import Any, Optional

import pybFoam.multiphase as multiphase

from neofoam.foam.initialization import create_time_mesh
from neofoam.foam.libraries import load_libraries
from neofoam.framework.initialization import (
    ConfigContext,
    InitializerBuilder,
    InitStep,
    LoadResult,
    StagedInitRunner,
    StagedInitSpec,
    lazy,
)
from neofoam.framework.initialization import (
    model as init_model,
)
from neofoam.framework.model import ModelRuntime
from neofoam.postprocess import postProcess

from .models.alpha_advection import advectionModel  # noqa: F401  (registers schemes)
from .models.pressure_velocity.base import PressureVelocityAlgorithm
from .solver_spec import incompressibleVoF

# OpenFOAM libraries native interFoam links but pybFoam does not, so their
# runtime-selection entries (waveVelocity/waveAlpha) are otherwise unreachable —
# no tutorial declares them via `libs (...)`, the interFoam binary is the declaration.
EXTRA_LIBRARIES = ["libwaveModels.so"]


class _FoamTimeStepView:
    """``ctx.time`` for a solver whose loop *is* the C++ ``Foam::Time``.

    ``incompressibleFluid`` gets its ``ctx.time`` from the ``solutionLoop``
    model's pure-Python ``LoopState``; this solver owns its loop and advances
    ``Foam::Time`` directly, so the read surface every ``WriteControl`` policy
    needs (``value`` / ``index`` / ``write_time``, plus ``start_time`` for the
    restart-append decision) is projected off the live runtime instead.
    """

    def __init__(self, runtime: Any) -> None:
        self._runtime = runtime
        # Foam::Time binds no startTime(); before the first increment its value
        # still is it, and this step runs at init.
        self.start_time: float = float(runtime.value())

    @property
    def value(self) -> float:
        return float(self._runtime.value())

    @property
    def index(self) -> int:
        return int(self._runtime.timeIndex())

    @property
    def write_time(self) -> bool:
        return bool(self._runtime.outputTime())


def _build_steps(runtime: ModelRuntime) -> list[InitStep]:
    """The steps of the runtime's ``@build`` alone, without field synthesis."""
    build = runtime.spec._build_func
    return list(build(runtime)) if build is not None else []


def create_init(case_dir: Optional[Path] = None) -> StagedInitRunner:
    """Build a fresh :class:`StagedInitRunner` for incompressibleVoF."""
    load_libraries(EXTRA_LIBRARIES)
    spec_builder = StagedInitSpec.build("incompressibleVoF")
    resolved_case_dir = case_dir or Path(".")

    @spec_builder.load
    def load_config() -> LoadResult:
        # VoF always uses PIMPLE; detect (and warn on missing PIMPLE dict).
        pressure_model = PressureVelocityAlgorithm.detect_and_create()
        optional_models = incompressibleVoF.detect_optional_models(resolved_case_dir)

        # alpha-advection scheme is runtime-switchable (advectionScheme key in
        # system/fvSolution, default MULES); it runs first each outer corrector,
        # then PIMPLE.
        alpha_model = advectionModel.detect_and_create()

        # postProcess is a solver-agnostic core Model: it LOADs the case's tables
        # (its ``system/postProcess.py`` script and/or spec file) and is stepped
        # last in the time loop. A case declaring no table loads an empty set and
        # the model does nothing.
        post_process_model = postProcess.instantiate(resolved_case_dir, "main")

        core_models: list[Any] = [alpha_model, pressure_model, post_process_model]
        return LoadResult(core_models=core_models, optional_models=optional_models)

    @spec_builder.resolve
    def resolve_models(config: ConfigContext) -> None:
        for opt in runner.optional_models:
            opt.run_resolve(config)

    @spec_builder.build
    def build_lazy(core_models: list[Any], optional_models: list[Any]) -> list[InitStep]:
        # Core models, looked up by spec name (order-independent).
        alpha_names = set(advectionModel.registered_names())
        pressure_names = {s.name for s in PressureVelocityAlgorithm.all_specs()}
        alpha_model = next(m for m in core_models if m.name in alpha_names)
        pressure_model = next(m for m in core_models if m.name in pressure_names)
        post_process_model = next(
            m for m in core_models if isinstance(m, ModelRuntime) and m.spec.name == "postProcess"
        )

        builder = InitializerBuilder()

        # Foam::Time ("runtime") + fvMesh ("mesh"). create_time_mesh routes the
        # Time onto ctx.models["runtime"] and the mesh onto ctx.mesh.
        builder.extend(create_time_mesh(runner.argv))

        # ctx.time: the write-cadence read surface postProcess evaluates its
        # tables against (see :class:`_FoamTimeStepView`).
        builder.add(
            lazy(
                "time",
                lambda ctx: _FoamTimeStepView(ctx["runtime"]),
                depends_on=["runtime"],
            )
        )

        # Not add_core_models for alpha-advection and PIMPLE: its run_build would
        # synthesize their schema-only field declarations, while each @build
        # reads or creates those fields itself (phi, mixture, alpha1/alpha2, rho,
        # rhoPhi from alpha-advection; U, p_rgh, gh, ghf, p, pimple_control,
        # pressure_reference from PIMPLE).
        builder.add_model("alpha_advection", alpha_model)
        builder.add_model("pressure_velocity", pressure_model)
        builder.add_core_models([("post_process_model", post_process_model)])
        for runtime in (alpha_model, pressure_model):
            builder.extend(_build_steps(runtime))

        # Two-phase turbulence model (wraps rho, U, phi, rhoPhi, mixture).
        def create_turbulence(ctx: dict[str, Any]) -> Any:
            return multiphase.TwoPhaseTransportModel(
                ctx["fields.rho"],
                ctx["fields.U"],
                ctx["fields.phi"],
                ctx["fields.rhoPhi"],
                ctx["models.mixture"],
            )

        builder.add(
            init_model(
                "turbulence",
                create_turbulence,
                depends_on=[
                    "fields.rho",
                    "fields.U",
                    "fields.phi",
                    "fields.rhoPhi",
                    "models.mixture",
                ],
            )
        )

        builder.add_optional_models(optional_models)
        # Register each active optional model BY NAME so it stays discoverable in
        # ctx.models for a live run.
        for opt in optional_models:
            builder.add_model(opt.name, opt)

        return builder.build()

    runner = StagedInitRunner(spec_builder.finalize())
    return runner
