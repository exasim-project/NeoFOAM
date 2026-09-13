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
from neofoam.solver._post_process_guard import _refuse_parallel_post_processing

from .models.alpha_advection import advectionModel  # noqa: F401  (registers schemes)
from .models.incompressibleVoFModel import incompressibleVoFModel
from .models.pressure_velocity.base import PressureVelocityAlgorithm

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


def create_init(case_dir: Optional[Path] = None) -> StagedInitRunner:
    """Build a fresh :class:`StagedInitRunner` for incompressibleVoF."""
    load_libraries(EXTRA_LIBRARIES)
    spec_builder = StagedInitSpec.build("incompressibleVoF")
    resolved_case_dir = case_dir or Path(".")

    @spec_builder.load
    def load_config() -> LoadResult:
        # VoF always uses PIMPLE; detect (and warn on missing PIMPLE dict).
        pressure_model = PressureVelocityAlgorithm.detect_and_create()
        optional_models = incompressibleVoFModel.detect_models(resolved_case_dir)

        # alpha-advection scheme is runtime-switchable (advectionScheme key in
        # system/fvSolution, default MULES); it runs first each outer corrector,
        # then PIMPLE.
        alpha_model = advectionModel.detect_and_create()

        # postProcess is a solver-agnostic core Model: it LOADs the case's tables
        # (its ``system/postProcess.py`` script and/or spec file) and is stepped
        # last in the time loop. A case declaring no table loads an empty set and
        # the model does nothing.
        post_process_model = postProcess.instantiate(resolved_case_dir, "main")
        _refuse_parallel_post_processing(post_process_model.config)

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

        # Register the core models under their solver-facing names, then run
        # each spec's @build to emit the field/model InitSteps it owns (phi,
        # mixture, alpha1/alpha2, rho, rhoPhi from alpha-advection; U, p_rgh, gh,
        # ghf, p, pimple_control, pressure_reference from PIMPLE). The bare
        # ModelSpec is not a BuildsInitSteps runtime, so add_core_models only
        # registers it — the @build steps are added explicitly here.
        # postProcess *is* a ModelRuntime, so add_core_models runs its @build for
        # it (emitting models.post_processor) — unlike the two bare specs above.
        builder.add_core_models(
            [
                ("alpha_advection", alpha_model),
                ("pressure_velocity", pressure_model),
                ("post_process_model", post_process_model),
            ]
        )
        for spec in (alpha_model, pressure_model):
            builder.extend(spec.build_steps())

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
