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

from neofoam.fields.synthesis import synthesize_init_step
from neofoam.foam.initialization import new_mesh, refuse_mesh_refinement
from neofoam.framework.context import Context
from neofoam.framework.initialization import (
    ConfigContext,
    InitializerBuilder,
    InitStep,
    LoadResult,
    StagedInitRunner,
    StagedInitSpec,
    field,
    lazy,
)
from neofoam.framework.initialization import (
    model as init_model,
)
from neofoam.framework.model import ModelRuntime, ModelSpec, bind_owned_interfaces
from neofoam.framework.tools import tool_graph_steps
from neofoam.tools.run import detect_tools
from neofoam.turbulence.config import TurbulencePropertiesConfig
from neofoam.turbulence.selection import select_turbulence_model
from neofoam.viscosity import select_viscosity_model
from neofoam.viscosity.config import TransportPropertiesConfig

from .configs import ControlDictConfig
from .models.field_writer import fieldWriter, writer_backend_steps
from .models.incompressibleFluidModel import incompressibleFluidModel
from .models.pressure_velocity.base import PressureVelocityAlgorithm
from .models.solution_loop import loop_backend_steps, solutionLoop


def _add_viscosity_model(builder: InitializerBuilder, selected: Any, case_dir: Path) -> None:
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

    builder.add(init_model("viscosity", build_viscosity, depends_on=["models.laminarTransport"]))
    builder.add(field("nu", create_nu, depends_on=["models.viscosity"]))


def _add_turbulence_model(builder: InitializerBuilder, case_dir: Path) -> None:
    """Add the momentum-transport model on the pybFoam-OpenFOAM fallback path.

    ``incompressibleFluid`` consumes every turbulence model through the pybFoam
    fallback (``select_turbulence_model(..., fallback=True)``): the returned
    :class:`~neofoam.turbulence.fallback.FallbackHandle` wraps the pybFoam model
    (which owns ``nut`` and assembles its own momentum stress) and schedules the
    model file's co-located ``fallback=True`` ``correct`` op after the loop. The
    handle is built lazily from the live ``U``/``phi``/transport, and its stress is
    exposed via ``viscous_stress()`` at ``models.viscousStress`` for the momentum
    equation. A configured model with no registered spec is built straight from
    OpenFOAM's own selection table (the selector says so on stdout); a registered
    model with no fallback op, or one of the wrong family, raises at selection time.
    """

    def build_turbulence(ctx: dict[str, Any]) -> Any:
        # Validated load so ``model_name`` resolves the RAS/LES model name (a
        # ``validate=False`` load leaves the sub-configs as plain dicts).
        turb_config = TurbulencePropertiesConfig.load(case_dir=case_dir)
        return select_turbulence_model(
            turb_config,
            fallback=True,
            case_dir=case_dir,
            U=ctx["fields.U"],
            phi=ctx["fields.phi"],
            transport=ctx["models.laminarTransport"],
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
        init_model("viscousStress", create_viscous_stress, depends_on=["models.turbulence"])
    )


def _optional_models_by_name(optional_models: list[Any]) -> dict[str, Any]:
    """Map each active optional model to its model name.

    Registering each detected optional-model runtime under ``rt.name`` keeps it
    discoverable in ``ctx.models`` (instead of stashing them under one opaque
    ``"optional_models"`` list). Gating of interface contributions is no longer a
    name lookup: it is intrinsic to the bound contributor runtimes that the MI7
    auto-wiring step binds onto the solutionLoop owner runtime.
    """
    return {m.name: m for m in optional_models}


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

        # Detect the mesh-preprocessing pipeline up front (config-only — no
        # mesh is touched). ``--no-preprocess`` on the argv skips detection so
        # the default disk-read ``mesh`` step is always used. Detection resolves
        # against the shared tool registry (solver-agnostic), so no solver import
        # is needed here.
        runner.preprocess_tools = (
            [] if "--no-preprocess" in runner.argv else detect_tools(resolved_case_dir)
        )

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
    def build_lazy(core_models: list[Any], optional_models: list[Any]) -> list[InitStep]:
        pressure_model = core_models[0]
        transport_config = next(
            (m for m in core_models if isinstance(m, TransportPropertiesConfig)), None
        )

        def _by_spec(spec_name: str) -> Any:
            return next(
                m for m in core_models if isinstance(m, ModelRuntime) and m.spec.name == spec_name
            )

        solution_loop_model = _by_spec("solutionLoop")
        field_writer_model = _by_spec("fieldWriter")
        argv = runner.argv

        def create_foam_arglist(_ctx: dict[str, Any]) -> Any:
            return pyf.argList(argv)

        def create_foam_time(ctx: dict[str, Any]) -> Any:
            # ``pyf.Time`` keeps only a raw reference to the argList, so the
            # argList must outlive it. In a ``-parallel`` run the argList owns the
            # MPI session and its destructor calls MPI_Finalize; letting it fall
            # out of scope here would finalize MPI mid-run (every later Pstream
            # broadcast then aborts). Keep it as its own init resource so it lives
            # as long as the Time / Context does.
            return pyf.Time(ctx["_foam_arglist"])

        def create_mesh(ctx: dict[str, Any]) -> Any:
            # First, so an AMR case is refused before the argList and the Foam::Time
            # it would be built on are resolved from the context.
            refuse_mesh_refinement()
            # pimpleFoam is a moving-mesh solver (``createDynamicFvMesh.H``): a case
            # with ``constant/dynamicMeshDict`` gets that dictionary's motion solver,
            # every other case the plain static fvMesh it always had.
            return new_mesh(ctx["_foam_arglist"], ctx["_foam_time"])

        def create_laminar_transport(ctx: dict[str, Any]) -> Any:
            # Raw pybFoam transport: drives correct() and feeds the OpenFOAM
            # turbulence fallback factory, which expects this concrete object.
            return singlePhaseTransportModel(ctx["fields.U"], ctx["fields.phi"])

        builder = InitializerBuilder()
        # the pybFoam Foam::Time is an init-only resource ("_foam_time"): it
        # parents the mesh objectRegistry and is the write(True) target, but is
        # never routed onto the Context (the leading underscore keeps it hidden).
        # ``_foam_arglist`` is held alongside it purely to keep the argList (and,
        # under ``-parallel``, the MPI session it owns) alive for the whole run.
        builder.add(lazy("_foam_arglist", create_foam_arglist))
        builder.add(lazy("_foam_time", create_foam_time, depends_on=["_foam_arglist"]))
        builder.add(lazy("mesh", create_mesh, depends_on=["_foam_time", "_foam_arglist"]))

        # When a mesh-preprocessing pipeline is active, its steps build the mesh
        # in-process; the terminal alias carries ``replaces=["mesh"]`` so
        # ``builder.build()`` drops the default disk-read ``mesh`` step above.
        # An empty pipeline adds nothing, so the disk-read default survives.
        builder.extend(tool_graph_steps(runner.preprocess_tools))

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
        builder.extend(writer_backend_steps(ControlDictConfig.load(case_dir=resolved_case_dir)))

        # PIMPLE is passed to add_core_models so it lands in models.pressure_velocity.
        # Its lazy field/model InitSteps come straight from the ModelSpec (used here
        # as both spec and "runtime", no instantiate). Mirror ModelRuntime.run_build:
        # synthesise the ``pimple.field(...)`` declarations (``U`` / ``p``) *first*,
        # then the ``@build`` steps (``phi``, pimpleControl, …) which depend on them.
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

        # Each fluid-property model REGISTERS THE FIELDS IT OWNS. The viscosity
        # model is config-only (no mesh needed to select/instantiate): it is built
        # here and its ``@build`` step (``run_build``) emits ``fields.nu``. The
        # turbulence model on this solver is always the pybFoam fallback: it owns
        # its own ``nut`` and momentum stress, built lazily from the live transport,
        # and registers ``models.viscousStress`` from the model's viscous_stress().
        _add_viscosity_model(builder, select_viscosity_model(transport_config), resolved_case_dir)
        _add_turbulence_model(builder, resolved_case_dir)

        builder.add_optional_models(optional_models)
        # Register each active optional model BY NAME so it stays discoverable in
        # ``ctx.models`` for a live run, instead of a single opaque list.
        for name, opt in _optional_models_by_name(optional_models).items():
            builder.add_model(name, opt)

        # MI7 auto-wiring: bind the solutionLoop runtime's owned interfaces
        # (timeStepConstraint / loopCondition) to the case's active contributing
        # optional-model runtimes, and register the owner runtime under its spec
        # name so the resolver finds ctx.models["solutionLoop"]. The bound
        # interfaces are stored, not folded this iteration — the live deltaT drive
        # stays deferred, so they are bound against an EMPTY Context. Capturing the
        # live pybFoam fields/models here would create a reference cycle holding
        # mesh-bound pybFoam objects that segfaults at GC across in-process solver
        # runs; the future live drive re-binds against the live Context at call time.
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
