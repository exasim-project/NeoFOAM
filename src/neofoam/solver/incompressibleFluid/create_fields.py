# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""3-stage initialization for the incompressibleFluid solver (minimal port).

LOAD    → detect the pressure-velocity algorithm and any optional models
RESOLVE → wire optional-model dependencies via ConfigContext
BUILD   → emit lazy InitSteps for mesh, runtime, fields, and transport/turbulence
"""

from pathlib import Path
from typing import Any, Optional

import pybFoam as pyf
from pybFoam.turbulence import singlePhaseTransportModel

from neofoam.framework.initialization import (
    ConfigContext,
    InitializerBuilder,
    InitStep,
    LoadResult,
    StagedInitRunner,
    StagedInitSpec,
    field,
    lazy,
    model as init_model,
)
from neofoam.framework.model import ModelSpec
from neofoam.turbulence import (
    OpenFOAMTurbulenceModel,
    SpecMomentumTransport,
    momentumTransportModel,
)
from neofoam.turbulence.config import TurbulencePropertiesConfig
from neofoam.turbulence.selection import model_name as turbulence_model_name
from neofoam.viscosity import select_viscosity_model
from neofoam.viscosity.config import TransportPropertiesConfig

from .configs import ControlDictConfig
from .models.field_writer import fieldWriter, writer_backend_steps
from .models.incompressibleFluidModel import incompressibleFluidModel
from .models.pressure_velocity.base import PressureVelocityAlgorithm
from .models.solution_loop import loop_backend_steps, solutionLoop


def _add_viscosity_model(
    builder: InitializerBuilder, selected: Any, case_dir: Path
) -> None:
    """Add the viscosity model + the ``fields.nu`` it owns.

    A native model (``ModelSpec``) is built here and its ``@build`` step emits
    ``fields.nu`` (run via ``ModelRuntime.run_build``). The OpenFOAM fallback is
    built lazily from the live transport and publishes ``nu`` from it.
    """
    if isinstance(selected, ModelSpec):
        runtime = selected.instantiate(case_dir)
        builder.add(init_model("viscosity", lambda _ctx: runtime))
        builder.extend(runtime.run_build())
        return

    def build_viscosity(ctx: dict[str, Any]) -> Any:
        return selected.build(transport=ctx["models.laminarTransport"])

    def create_nu(ctx: dict[str, Any]) -> Any:
        return ctx["models.viscosity"].nu_field()

    builder.add(
        init_model("viscosity", build_viscosity, depends_on=["models.laminarTransport"])
    )
    builder.add(field("nu", create_nu, depends_on=["models.viscosity"]))


def _add_turbulence_model(builder: InitializerBuilder, case_dir: Path) -> None:
    """Add the momentum-transport model + the ``fields.nut`` it owns *if necessary*.

    A native model (selected by name from ``turbulenceProperties``) is built here,
    wrapped by :class:`SpecMomentumTransport`. Its ``@build`` step registers the
    ``viscousStress`` the momentum equation uses (and ``fields.nut`` only when the
    closure has an eddy viscosity — laminar emits none). The OpenFOAM fallback is
    built lazily from the live ``U``/``phi``/transport and assembles its own stress,
    which it exposes via ``viscous_stress()`` for the momentum equation.
    """
    turb_config = TurbulencePropertiesConfig.load(case_dir=case_dir, validate=False)
    name = turbulence_model_name(turb_config)
    spec = momentumTransportModel.find_spec(name) if name is not None else None
    if spec is not None:
        runtime = spec.instantiate(case_dir)
        model_obj = SpecMomentumTransport(runtime)
        builder.add(init_model("turbulence", lambda _ctx: model_obj))
        # The model's @build registers ``models.viscousStress`` (and ``fields.nut``
        # if it has an eddy viscosity).
        builder.extend(runtime.run_build())
        return

    def build_turbulence(ctx: dict[str, Any]) -> Any:
        return OpenFOAMTurbulenceModel(
            ctx["fields.U"], ctx["fields.phi"], ctx["models.laminarTransport"]
        ).build()

    def create_viscous_stress(ctx: dict[str, Any]) -> Any:
        # The fallback assembles its own stress; ask the model for it so the
        # momentum equation resolves ``models.viscousStress`` uniformly.
        return ctx["models.turbulence"].viscous_stress()

    builder.add(
        init_model(
            "turbulence",
            build_turbulence,
            depends_on=["fields.U", "fields.phi", "models.laminarTransport"],
        )
    )
    builder.add(
        init_model(
            "viscousStress", create_viscous_stress, depends_on=["models.turbulence"]
        )
    )


def create_init(case_dir: Optional[Path] = None) -> StagedInitRunner:
    """Build a fresh :class:`StagedInitRunner` for incompressibleFluid.

    The closures over ``runner`` give the resolve/build stages access to
    the live ``argv`` and ``optional_models`` populated by load.
    """
    spec_builder = StagedInitSpec.build("incompressibleFluid")
    resolved_case_dir = case_dir or Path(".")

    @spec_builder.load
    def load_config() -> LoadResult:
        pressure_model = PressureVelocityAlgorithm.detect_and_create()
        optional_models = incompressibleFluidModel.detect_models(resolved_case_dir)

        # solutionLoop (advances time) and fieldWriter (persists fields) are
        # separate-concern Models, loaded here so their controlDict is validated
        # up front; both are composed by incompressibleFluid.execution_graph.
        solution_loop_model = solutionLoop.instantiate(resolved_case_dir, "main")
        field_writer_model = fieldWriter.instantiate(resolved_case_dir, "main")

        # Solver-core configs, so ``LoadResult.configs`` exposes the
        # configs the solver consumes (collectible / savable / printable).
        # ``validate=False`` keeps loading lenient — ``LoadResult.validate()``
        # reports any errors. ``pressure_model`` stays at index 0; these
        # append after it and are inert in the BUILD stage (selected there
        # by isinstance).
        #
        # The PIMPLE ``fvSchemes`` / ``fvSolution`` slices are *declared* on
        # the ``pimple`` spec, so ``LoadResult.config_classes`` lists them as
        # part of the schema set. Their instances are not loaded here: the
        # OpenFOAM reader does not yet parse the typed scheme values
        # (``DivScheme`` …) / ``dict`` solver entries those slices carry.
        core_models = [pressure_model, solution_loop_model, field_writer_model]
        core_models += [
            ControlDictConfig.load(case_dir=resolved_case_dir, validate=False),
            TransportPropertiesConfig.load(case_dir=resolved_case_dir, validate=False),
        ]

        return LoadResult(
            core_models=core_models,
            optional_models=optional_models,
        )

    @spec_builder.resolve
    def resolve_models(config: ConfigContext) -> None:
        for opt in runner.optional_models:
            opt.run_resolve(config)

    @spec_builder.build
    def build_lazy(
        core_models: list[Any], optional_models: list[Any]
    ) -> list[InitStep]:
        pressure_model = core_models[0]
        transport_config = next(
            (m for m in core_models if isinstance(m, TransportPropertiesConfig)), None
        )

        def _by_spec(spec_name: str) -> Any:
            return next(
                m
                for m in core_models
                if getattr(getattr(m, "spec", None), "name", None) == spec_name
            )

        solution_loop_model = _by_spec("solutionLoop")
        field_writer_model = _by_spec("fieldWriter")
        argv = runner.argv

        def create_foam_time(_ctx: dict[str, Any]) -> Any:
            argList = pyf.argList(argv)
            return pyf.Time(argList)

        def create_mesh(ctx: dict[str, Any]) -> Any:
            return pyf.fvMesh(ctx["_foam_time"])

        def create_laminar_transport(ctx: dict[str, Any]) -> Any:
            # Raw pybFoam transport: drives correct() and feeds the OpenFOAM
            # turbulence fallback factory, which expects this concrete object.
            return singlePhaseTransportModel(ctx["fields.U"], ctx["fields.phi"])

        builder = InitializerBuilder()
        # the pybFoam Foam::Time is an init-only resource ("_foam_time"): it
        # parents the mesh objectRegistry and is the write(True) target, but is
        # never routed onto the Context (the leading underscore keeps it hidden).
        builder.add(lazy("_foam_time", create_foam_time))
        builder.add(lazy("mesh", create_mesh, depends_on=["_foam_time"]))

        # solutionLoop + fieldWriter are the framework *core* Models, instantiated
        # as real ModelRuntimes: add_core_models registers them and runs each
        # @build, emitting the LoopState (ctx.time) + engine and the FieldWriter
        # steps. They are backend-agnostic; the pybFoam touch-points (the FoamTime
        # LoopBackend, Courant provider, logger, write hook, step reporter) are
        # injected by the *_backend_steps below through the framework seams.
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

        # PIMPLE is passed to add_core_models so it lands in models.pressure_velocity.
        # Its lazy field/model InitSteps come straight from the ModelSpec (used here
        # as both spec and "runtime", no instantiate). Mirror ModelRuntime.run_build:
        # synthesise the ``pimple.field(...)`` declarations (``U`` / ``p``) *first*,
        # then the ``@build`` steps (``phi``, pimpleControl, …) which depend on them.
        from neofoam.fields.synthesis import synthesize_init_step

        builder.add_core_models([("pressure_velocity", pressure_model)])
        for decl in pressure_model.field_decls:
            builder.add(synthesize_init_step(decl))
        if pressure_model._build_func is not None:
            builder.extend(pressure_model._build_func(pressure_model))

        builder.add(
            init_model(
                "laminarTransport",
                create_laminar_transport,
                depends_on=["fields.U", "fields.phi"],
            )
        )

        # Each fluid-property model REGISTERS THE FIELDS IT OWNS. A native model
        # is config-only (no mesh needed to select/instantiate), so it is built
        # here and its ``@build`` step (``run_build``) emits the field InitSteps:
        # the viscosity model emits ``fields.nu``; a turbulence model emits
        # ``fields.nut`` only if it has an eddy viscosity (laminar emits none —
        # ``LinearViscousStress`` then treats ``nut`` as zero). The OpenFOAM
        # fallbacks need the live transport, so they are built lazily and publish
        # their field through the model resolved from the Context. Each turbulence
        # path also registers ``models.viscousStress`` (native: the model's @build;
        # fallback: from the model's viscous_stress()).
        _add_viscosity_model(
            builder, select_viscosity_model(transport_config), resolved_case_dir
        )
        _add_turbulence_model(builder, resolved_case_dir)

        builder.add_optional_models(optional_models)
        builder.add_model("optional_models", optional_models)

        return builder.build()

    runner = StagedInitRunner(spec_builder.finalize())
    return runner
