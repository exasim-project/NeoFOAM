# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""3-stage initialization for the incompressibleFluidNeoN solver.

LOAD    → detect the pressure-velocity algorithm and any optional models
RESOLVE → wire optional-model dependencies via ConfigContext
BUILD   → emit lazy InitSteps for the NeoN runtime, fields, and turbulence

Differences from the pybFoam ``incompressibleFluid`` init graph:

* no ``mesh`` step and no mesh-preprocess pipeline — ``create_adapter_run_time``
  constructs its own ``MeshAdapter`` (fvMesh) on the ``Foam::Time`` registry;
  also creating a pybFoam ``fvMesh`` would double-register the default region;
* the NeoN ``RunTime`` adapter is the central resource: an init-only
  ``_neon_runtime`` (never routed onto the Context) plus a
  ``models.neon_runtime`` alias so operations inject it by name;
* viscosity is not a Python model family: the C++ factory is already
  runtime-selected from ``constant/transportProperties``, so one ``nu_vol``
  init step suffices;
* turbulence is one ``turbulence`` init step selecting the native NeoN ModelSpec
  closure by name from ``constant/turbulenceProperties`` (the single
  :mod:`neofoam.turbulence.momentumTransport` family, ``fallback=False``); a model
  with no native NeoN closure raises cleanly at selection time.
"""

from pathlib import Path
from typing import Any, Optional

import pybFoam as pyf

from neofoam import neofoam_bindings as nfb  # NeoFOAM Python bindings
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
from neofoam.solver.neon_runtime import requested_executor
from neofoam.turbulence.config import TurbulencePropertiesConfig
from neofoam.turbulence.selection import select_turbulence_model

from .configs import ControlDictConfig
from .models.field_writer import fieldWriter, neon_writer_backend_steps
from .models.incompressibleFluidNeoNModel import incompressibleFluidNeoNModel
from .models.pressure_velocity.base import PressureVelocityAlgorithmNeoN
from .models.solution_loop import neon_loop_backend_steps, solutionLoop

# Solver-solution subdicts mapped from OpenFOAM to NeoN/Ginkgo equivalents. The
# base solver subdicts AND the *Final subdicts so the final outer-corrector
# pass selects a converted dict (mirrors the legacy neoPimpleFoam port). The
# turbulence transport unknowns are included so the pure-Python NeoN closures
# (kEpsilon / SpalartAllmaras / kOmegaSST) find their converted solver dicts.
_MAPPED_SOLVER_DICTS = (
    "p",
    "U",
    "pFinal",
    "UFinal",
    "k",
    "kFinal",
    "epsilon",
    "epsilonFinal",
    "nuTilda",
    "nuTildaFinal",
    "omega",
    "omegaFinal",
)


def _optional_models_by_name(optional_models: list[Any]) -> dict[str, Any]:
    """Map each active optional model to its model name (see incompressibleFluid)."""
    return {m.name: m for m in optional_models}


def create_init(case_dir: Optional[Path] = None) -> StagedInitRunner:
    """Build a fresh :class:`StagedInitRunner` for incompressibleFluidNeoN.

    The closures over ``runner`` give the resolve/build stages access to
    the live ``argv`` and ``optional_models`` populated by load.
    """
    spec_builder = StagedInitSpec.build("incompressibleFluidNeoN")
    resolved_case_dir = case_dir or Path(".")

    @spec_builder.load
    def load_config() -> LoadResult:
        pressure_model = PressureVelocityAlgorithmNeoN.detect_and_create()
        optional_models = incompressibleFluidNeoNModel.detect_models(resolved_case_dir)

        # solutionLoop (advances time) and fieldWriter (persists fields) are
        # separate-concern Models, loaded here so their controlDict is validated
        # up front; both are composed by incompressibleFluidNeoN.execution_graph.
        solution_loop_model = solutionLoop.instantiate(resolved_case_dir, "main")
        field_writer_model = fieldWriter.instantiate(resolved_case_dir, "main")

        core_models: list[Any] = [
            pressure_model,
            solution_loop_model,
            field_writer_model,
        ]
        core_models.append(ControlDictConfig.load(case_dir=resolved_case_dir, validate=False))

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

        def _by_spec(spec_name: str) -> Any:
            return next(
                m for m in core_models if isinstance(m, ModelRuntime) and m.spec.name == spec_name
            )

        solution_loop_model = _by_spec("solutionLoop")
        field_writer_model = _by_spec("fieldWriter")
        argv = runner.argv

        def create_arg_list(_ctx: dict[str, Any]) -> Any:
            return pyf.argList(argv)

        def create_foam_time(ctx: dict[str, Any]) -> Any:
            # Foam::Time keeps a raw reference to the argList — it must stay
            # alive for the whole run (the NeoNTimeSync backend pins both;
            # letting the argList be GC'd corrupts later dictionary reads,
            # e.g. the fvSchemes conversion inside create_adapter_run_time).
            return pyf.Time(ctx["_arg_list"])

        def create_neon_runtime(ctx: dict[str, Any]) -> Any:
            rt = nfb.create_adapter_run_time(ctx["_foam_time"], requested_executor())
            # Map OpenFOAM dictionaries to NeoN/Ginkgo equivalents once, up
            # front (mirrors the legacy port's setup block).
            rt.fv_schemes_dict = nfb.map_fv_schemes(rt.fv_schemes_dict)
            solvers = rt.fv_solution_dict.subDict("solvers")
            for name in _MAPPED_SOLVER_DICTS:
                # Resolves OpenFOAM regex keys ("(U|k|epsilon)"), skips absent fields
                # and never maps one entry twice.
                nfb.map_solver_settings(solvers, name)
            return rt

        def alias_neon_runtime(ctx: dict[str, Any]) -> Any:
            return ctx["_neon_runtime"]

        def create_nu_vol(ctx: dict[str, Any]) -> Any:
            # Volume nu for the explicit dev2 viscous-stress term (nuEff/nut
            # come from the turbulence model below).
            rt = ctx["_neon_runtime"]
            return nfb.create_uniform_volume_field(rt, "nu", nfb.read_transport_viscosity(rt))

        def create_turbulence(ctx: dict[str, Any]) -> Any:
            # Runtime-selected turbulence model (per constant/turbulenceProperties):
            # the native NeoN ModelSpec closure (laminar / kEpsilon /
            # SpalartAllmaras / kOmegaSST) registered under the configured name in
            # the single ``momentumTransportModel`` family. The native closures own
            # their wall functions, so there is no C++ factory branch: a model with
            # no native closure raises cleanly at selection time. The handle exposes
            # nut / nu_eff / rotate_old_times / correct / write; for laminar nut = 0
            # and nuEff = nu.
            rt = ctx["_neon_runtime"]
            # Validated load: with ``validate=False`` the RAS/LES sub-configs stay
            # plain dicts and ``model_name`` cannot resolve the RAS/LES model name.
            turb_config = TurbulencePropertiesConfig.load(case_dir=resolved_case_dir)
            turb = select_turbulence_model(
                turb_config,
                fallback=False,
                runtime=rt,
                nu=ctx["models.nu_vol"],
                case_dir=resolved_case_dir,
            )
            turb.validate(ctx["fields.U"])
            return turb

        builder = InitializerBuilder()
        # Both runtimes are init-only resources (leading underscore keeps them
        # off the Context): the pybFoam Foam::Time parents the case registry;
        # the NeoN RunTime adapter is re-exposed as ``models.neon_runtime`` so
        # operations can inject it. The NeoNTimeSync backend holds the
        # Foam::Time reference so the adapter's MeshAdapter never dangles.
        builder.add(lazy("_arg_list", create_arg_list))
        builder.add(lazy("_foam_time", create_foam_time, depends_on=["_arg_list"]))
        builder.add(lazy("_neon_runtime", create_neon_runtime, depends_on=["_foam_time"]))
        builder.add(init_model("neon_runtime", alias_neon_runtime, depends_on=["_neon_runtime"]))

        # solutionLoop + fieldWriter are the framework *core* Models; the NeoN
        # touch-points (NeoNTimeSync backend, write hook, logger, reporter) are
        # injected by the *_backend_steps through the framework seams.
        builder.add_core_models(
            [
                ("solution_loop_model", solution_loop_model),
                ("field_writer_model", field_writer_model),
            ]
        )
        builder.extend(neon_loop_backend_steps())
        builder.extend(neon_writer_backend_steps())

        # PIMPLE is passed to add_core_models so it lands in
        # models.pressure_velocity. Its InitSteps come straight from the
        # ModelSpec (used as both spec and "runtime", no instantiate). The
        # field declarations are NOT synthesized (that path is pybFoam-only):
        # the spec's @build emits the NeoN U/p/phi reads itself.
        builder.add_core_models([("pressure_velocity", pressure_model)])
        if pressure_model._build_func is not None:
            builder.extend(pressure_model._build_func(pressure_model))

        builder.add(init_model("nu_vol", create_nu_vol, depends_on=["_neon_runtime"]))
        builder.add(
            init_model(
                "turbulence",
                create_turbulence,
                depends_on=["_neon_runtime", "models.nu_vol", "fields.U"],
            )
        )

        builder.add_optional_models(optional_models)
        # Register each active optional model BY NAME so it stays discoverable
        # in ``ctx.models`` for a live run.
        for name, opt in _optional_models_by_name(optional_models).items():
            builder.add_model(name, opt)

        # Register the solutionLoop owner runtime under its spec name so it
        # stays discoverable as ctx.models["solutionLoop"]. Its gather hooks
        # (timeStepConstraint / loopCondition) need no wiring step: the hooks
        # a consumer injects are bound to the live Context on every operation
        # call, so nothing case-bound is ever captured across runs.
        def register_loop_runtime(_work: dict[str, Any]) -> Any:
            return solution_loop_model

        builder.add(
            init_model(
                "solutionLoop",
                register_loop_runtime,
                depends_on=["models.solution_loop"],
            )
        )

        return builder.build()

    runner = StagedInitRunner(spec_builder.finalize())
    return runner
