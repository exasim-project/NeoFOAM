# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""incompressibleFluid — solver entrypoint (minimal port).

Adapted from ``feat/python_solvers`` to the SolverSpec/ModelSpec API in
``stack/python_arch``. Only PIMPLE is wired up; SIMPLE/PISO/boussinesq/SA
are omitted in this minimal version. The main iteration loop is owned by the
``solution_loop`` engine (see ``neofoam.algorithms.solution_loop.solution_loop`` and
``models.solution_loop``): it drives advancement, deltaT adjustment from
injectable stability constraints, and the write decision.
"""

from typing import Annotated, Any, Optional

from pybFoam import Info

from neofoam.framework.context import Context
from neofoam.framework.graph import DAGResolver
from neofoam.framework.initialization import Depends, StagedInitRunner
from neofoam.framework.operations import (
    IterativeOp,
    Operation,
    Operations,
    StepBuilder,
)
from neofoam.framework.solver import Solver
from neofoam.framework.types import OperationMetadata
from neofoam.turbulence import momentumTransportModel
from neofoam.viscosity import viscosityModel

from .configs import ControlDictConfig
from .create_fields import create_init
from .models.incompressibleFluidModel import incompressibleFluidModel
from .models.pressure_velocity.base import PressureVelocityAlgorithm
from .models.solution_loop import SolutionLoopPredicate


# model lookup helper: find an instantiated core model by its spec name
def _core_model(state: Any, spec_name: str) -> Any:
    return next(
        m
        for m in state.core_models
        if getattr(getattr(m, "spec", None), "name", None) == spec_name
    )


incompressibleFluid = Solver("incompressibleFluid")

# Declare the full config schema on the spec, case-free: the solver's own
# configs plus the model families it owns. Every member's configs join the
# schema; per-case detection (in create_fields) picks which members run.
# Viscosity (transport) and momentum-transport (turbulence) are core model
# families: each contributes its own config (transportProperties /
# turbulenceProperties) through its registered models, so the solver does not
# name those config classes itself.
incompressibleFluid.config(ControlDictConfig)
incompressibleFluid.core_models(PressureVelocityAlgorithm)  # required: pick ONE
incompressibleFluid.core_models(viscosityModel)  # molecular nu (transportProperties)
incompressibleFluid.core_models(
    momentumTransportModel
)  # nut + stress (turbulenceProperties)
incompressibleFluid.optional_models(incompressibleFluidModel)  # zero or more


@incompressibleFluid.initializer
def initialize(
    self: Any, init: Annotated[StagedInitRunner, Depends(create_init)]
) -> Context:
    """Initialize using create_init factory with dependency injection."""
    return init.run()


@incompressibleFluid.execution_graph_step
def execution_graph(
    self: Any,
    ctx: Context,
    domain_name: Optional[str] = None,
) -> tuple[StepBuilder, Operations]:
    """Build the time-loop + PIMPLE inner-loop execution graph."""
    _ = domain_name

    builder = StepBuilder()

    # The pressure-velocity algorithm (pimple) is the first core model.
    # We pass the spec as its own runtime — it stores configured
    # algorithm_type / use_boussinesq attributes directly.
    algorithm_model = self.state.core_models[0]
    algo_ops = Operations(algorithm_model._build_operations_for(algorithm_model))

    # The main iteration loop is the solutionLoop model: its operations
    # (set_time_step / increment_time) and predicate own advancement and deltaT
    # adjustment. Persisting fields is the separate fieldWriter model.
    loop_ops = Operations(_core_model(self.state, "solutionLoop").operations)
    writer_ops = Operations(_core_model(self.state, "fieldWriter").operations)

    time_loop_op = Operation(
        func=IterativeOp(SolutionLoopPredicate()),
        metadata=OperationMetadata(op_name="time_loop"),
    )

    # The fluid-property models own their operations; the solver steps them after
    # the pressure-velocity loop (the end-of-step *correct* phase, matching
    # pimpleFoam) — viscosity before turbulence (turbulence reads nu), each iterated
    # since a model may contribute several ops. ``nuEff`` is primed at init and the
    # momentum predictor uses the value the turbulence model refreshed at the end of
    # the previous step; the OpenFOAM fallbacks advance their pybFoam model here.
    viscosity_ops = ctx.models["viscosity"].operations
    turbulence_ops = ctx.models["turbulence"].operations

    with builder.loop(time_loop_op) as time_builder:
        time_builder.step(loop_ops["set_time_step"])
        time_builder.step(loop_ops["increment_time"])

        with time_builder.loop(algo_ops["inner_loop"]) as inner_builder:
            inner_builder.step(algo_ops["momentum"])
            inner_builder.step(algo_ops["continuity"])

        for op in viscosity_ops:
            time_builder.step(op)
        for op in turbulence_ops:
            time_builder.step(op)

        time_builder.step(writer_ops["write_output"])

    # Optional-model operations are merged by the resolver (placed by their own
    # depends_on / operation_number).
    model_ops = Operations()
    for opt in self.state.optional_models:
        model_ops.add(opt.operations)

    return builder, model_ops


def run(
    argv: Optional[list[str]] = None,
    log_file: Optional[Any] = None,
) -> Context:
    """Run one full simulation and return the final Context.

    If ``log_file`` is given, fd 1 (stdout) is redirected to that file
    for the duration of the solve so C++ ``Info`` output ends up there
    instead of the calling process's stdout.
    """
    import os
    import sys
    from pathlib import Path

    redirect = log_file is not None
    saved_fd: Optional[int] = None
    if redirect:
        log_path = Path(str(log_file))
        sys.stdout.flush()
        saved_fd = os.dup(1)
        log_fd = os.open(str(log_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        os.dup2(log_fd, 1)
        os.close(log_fd)

    try:
        solver = incompressibleFluid.instantiate(argv=argv or [])
        ctx = solver.initialize()

        Info("Starting time loop")

        builder, model_ops = solver.execution_graph(ctx=ctx)

        resolver = DAGResolver()
        resolved = resolver.resolve(builder, model_ops)
        resolved.operations.run(ctx)

        Info("End")
        return ctx
    finally:
        if redirect and saved_fd is not None:
            sys.stdout.flush()
            os.dup2(saved_fd, 1)
            os.close(saved_fd)
