# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""3-stage initialization for the incompressibleVoFNeon solver.

LOAD    → detect the PIMPLE algorithm and any optional models
RESOLVE → wire optional-model dependencies via ConfigContext
BUILD   → emit lazy InitSteps for the NeoN runtime, the alpha-advection + PIMPLE
          core models (each owns the fields it registers), and the solution-loop
          + field-writer engines with their NeoN backends.

Mirrors ``incompressibleFluidNeoN.create_fields`` (NeoN runtime plumbing) with
the VoF specifics of ``incompressibleVoF.create_fields`` (alpha advection +
density-weighted PIMPLE) — but laminar, so there is no turbulence init step. As
in the legacy ``neoInterFoam.setup``, the VoF operator scheme keys
(``div(rhoPhi,U)`` / the dev2 stress / ``laplacian(muf,U)`` /
``laplacian(rAUf,p_rgh)``) are registered on the mapped ``fvSchemes`` up front.
"""

from pathlib import Path
from typing import Any, Optional

import pybFoam as pyf
import neon._neon as nn  # NeoN Python bindings
from neofoam import neofoam_bindings as nfb  # NeoFOAM Python bindings

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
from neofoam.framework.model import ModelRuntime, ModelSpec, bind_owned_interfaces

from .configs import ControlDictConfig
from .models.alpha_advection.alphaAdvectionModel import alpha_advection_model
from .models.field_writer import fieldWriter, neon_writer_backend_steps
from .models.incompressibleVoFNeonModel import incompressibleVoFNeonModel
from .models.pressure_velocity.base import PressureVelocityAlgorithmNeoN
from .models.shared import ALPHA1_FIELD, prefixed_alpha_solver_key
from .models.solution_loop import neon_loop_backend_steps, solutionLoop

# Solver-solution subdicts mapped from OpenFOAM to NeoN/Ginkgo equivalents — the
# VoF momentum/pressure fields and their *Final variants (the alpha equation
# uses the MULES primitive directly, so it needs no linear-solver mapping).
_MAPPED_SOLVER_DICTS = ("p_rgh", "p_rghFinal", "U", "UFinal")


def _optional_models_by_name(optional_models: list[Any]) -> dict[str, Any]:
    """Map each active optional model to its model name."""
    return {m.name: m for m in optional_models}


# MULES control keys carried in the ``alpha.water`` fvSolution subdict — read by
# the alpha model, but not Ginkgo config keys (Ginkgo rejects unknown keys), so
# they are stripped from the copy handed to the predictor's linear solver.
_ALPHA_MULES_KEYS = (
    "MULESCorr",
    "nAlphaCorr",
    "nAlphaSubCycles",
    "cAlpha",
    "nLimiterIter",
    "icAlpha",
    "scAlpha",
    "alphaApplyPrevCorr",
)


def _register_alpha_predictor_solver(rt: Any) -> None:
    """Register the MULESCorr predictor's Ginkgo solver dict under ``ALPHA1_FIELD``.

    The OpenFOAM alpha solver subdict is keyed by a regex (damBreak:
    ``"alpha.water.*"``) and carries MULES control keys. The generic scalar PDE
    solver looks the dict up by the exact field name and hands it to Ginkgo, so
    map the linear-solver settings and register a MULES-key-stripped copy under
    the exact key. No-op when there is no regex-keyed alpha solver subdict.
    """
    solvers = rt.fv_solution_dict.subDict("solvers")
    key = prefixed_alpha_solver_key(solvers)
    if key is None:
        return
    mapped = nfb.map_fv_solution(solvers.subDict(key))
    for junk in _ALPHA_MULES_KEYS:
        if mapped.contains(junk):
            mapped.remove(junk)
    solvers.insert_dict(ALPHA1_FIELD, mapped)


def create_init(case_dir: Optional[Path] = None) -> StagedInitRunner:
    """Build a fresh :class:`StagedInitRunner` for incompressibleVoFNeon."""
    spec_builder = StagedInitSpec.build("incompressibleVoFNeon")
    resolved_case_dir = case_dir or Path(".")

    @spec_builder.load
    def load_config() -> LoadResult:
        pressure_model = PressureVelocityAlgorithmNeoN.detect_and_create()
        optional_models = incompressibleVoFNeonModel.detect_models(resolved_case_dir)

        # solutionLoop (advances time) and fieldWriter (persists fields) are
        # separate-concern Models, loaded here so their controlDict is validated
        # up front; both are composed by incompressibleVoFNeon.execution_graph.
        solution_loop_model = solutionLoop.instantiate(resolved_case_dir, "main")
        field_writer_model = fieldWriter.instantiate(resolved_case_dir, "main")

        core_models: list[Any] = [
            pressure_model,
            alpha_advection_model,
            solution_loop_model,
            field_writer_model,
        ]
        core_models.append(
            ControlDictConfig.load(case_dir=resolved_case_dir, validate=False)
        )

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
        # Core models, looked up by spec name (order-independent): the bare
        # PIMPLE / alpha-advection ModelSpecs and the instantiated
        # solutionLoop / fieldWriter ModelRuntimes.
        def _bare_spec(names: set[str]) -> Any:
            return next(
                m for m in core_models if isinstance(m, ModelSpec) and m.name in names
            )

        pressure_model = _bare_spec(
            {s.name for s in PressureVelocityAlgorithmNeoN.all_specs()}
        )
        alpha_model = _bare_spec({alpha_advection_model.name})

        def _by_spec(spec_name: str) -> Any:
            return next(
                m
                for m in core_models
                if isinstance(m, ModelRuntime) and m.spec.name == spec_name
            )

        solution_loop_model = _by_spec("solutionLoop")
        field_writer_model = _by_spec("fieldWriter")
        argv = runner.argv

        def create_arg_list(_ctx: dict[str, Any]) -> Any:
            return pyf.argList(argv)

        def create_foam_time(ctx: dict[str, Any]) -> Any:
            # Foam::Time keeps a raw reference to the argList — it must stay
            # alive for the whole run (the NeoNTimeSync backend pins both).
            return pyf.Time(ctx["_arg_list"])

        def create_neon_runtime(ctx: dict[str, Any]) -> Any:
            rt = nfb.create_adapter_run_time(ctx["_foam_time"])
            # Map OpenFOAM dictionaries to NeoN/Ginkgo equivalents once, up front.
            rt.fv_schemes_dict = nfb.map_fv_schemes(rt.fv_schemes_dict)
            solvers = rt.fv_solution_dict.subDict("solvers")
            for name in _MAPPED_SOLVER_DICTS:
                if solvers.contains(name):
                    solvers.insert_dict(
                        name, nfb.map_fv_solution(solvers.subDict(name))
                    )
            # VoF operator scheme keys (mirror neoInterFoam.setup): the implicit
            # rhoPhi convection + the dev2 laminar viscous-stress divergence, and
            # the muf / rAUf laplacians the momentum/pressure solves look up.
            div_schemes = rt.fv_schemes_dict.subDict("divSchemes")
            div_schemes.insert_token_list(
                "div(rhoPhi,U)", nn.TokenList(["Gauss", "linear"])
            )
            div_schemes.insert_token_list(
                "div((nuEff*dev2(T(grad(U)))))", nn.TokenList(["Gauss", "linear"])
            )
            lap = rt.fv_schemes_dict.subDict("laplacianSchemes")
            for key in ("laplacian(muf,U)", "laplacian(rAUf,p_rgh)"):
                lap.insert_token_list(
                    key, nn.TokenList(["Gauss", "linear", "uncorrected"])
                )
            # MULESCorr implicit predictor (alphaEqn.H): the upwind convection of
            # the alpha transport and the exact-key Ginkgo solver dict the generic
            # scalar PDE solver looks up. Its solver dict is keyed by the OpenFOAM
            # regex ("alpha.water.*"); the generic PDE solver looks it up by the
            # exact field name "alpha.water" (no regex matching in the NeoN dict),
            # and the MULES control keys (MULESCorr/nAlphaCorr/cAlpha/...) are not
            # Ginkgo config keys, so a stripped copy is registered under the exact
            # key. Harmless when MULESCorr is off (the explicit path never builds
            # the predictor).
            div_schemes.insert_token_list(
                f"div(phi,{ALPHA1_FIELD})", nn.TokenList(["Gauss", "upwind"])
            )
            _register_alpha_predictor_solver(rt)
            return rt

        def alias_neon_runtime(ctx: dict[str, Any]) -> Any:
            return ctx["_neon_runtime"]

        builder = InitializerBuilder()
        # Both runtimes are init-only resources (leading underscore keeps them
        # off the Context): the pybFoam Foam::Time parents the case registry; the
        # NeoN RunTime adapter is re-exposed as ``models.neon_runtime`` so
        # operations can inject it. The NeoNTimeSync backend holds the Foam::Time
        # reference so the adapter's MeshAdapter never dangles.
        builder.add(lazy("_arg_list", create_arg_list))
        builder.add(lazy("_foam_time", create_foam_time, depends_on=["_arg_list"]))
        builder.add(
            lazy("_neon_runtime", create_neon_runtime, depends_on=["_foam_time"])
        )
        builder.add(
            init_model("neon_runtime", alias_neon_runtime, depends_on=["_neon_runtime"])
        )

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

        # alpha-advection + PIMPLE are the VoF core models. Their InitSteps come
        # straight from the ModelSpec (used as both spec and "runtime", no
        # instantiate). The field declarations are NOT synthesized (that path is
        # pybFoam-only): each spec's @build emits the NeoN field reads itself.
        builder.add_core_models(
            [
                ("alpha_advection", alpha_model),
                ("pressure_velocity", pressure_model),
            ]
        )
        for spec in (alpha_model, pressure_model):
            builder.extend(spec.build_steps())

        builder.add_optional_models(optional_models)
        # Register each active optional model BY NAME so it stays discoverable in
        # ``ctx.models`` for a live run.
        for name, opt in _optional_models_by_name(optional_models).items():
            builder.add_model(name, opt)

        # MI7 auto-wiring: bind the solutionLoop runtime's owned interfaces
        # (timeStepConstraint / loopCondition) to the case's active contributing
        # optional-model runtimes. Bound against an EMPTY Context — capturing live
        # backend objects here would create a reference cycle that segfaults at GC
        # across in-process solver runs; the live fold re-binds against the live
        # Context at call time.
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
