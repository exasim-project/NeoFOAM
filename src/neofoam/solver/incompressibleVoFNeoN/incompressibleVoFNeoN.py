# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""incompressibleVoFNeoN — solver entrypoint.

Framework NeoN port of an interFoam-style two-phase VoF solver: the same
SolverSpec / ModelSpec / StagedInit composition as the pybFoam
``incompressibleVoF`` (its physics reference), with the NeoN bindings as the
backend.
Laminar only. One time step is::

    set_time_step        # folds courant/maxDeltaT; NeoNTimeSync syncs rt.dt
    increment_time       # "Time = ..." print; advance; sync rt.t / rt.dt
    alpha_advection      # MULES phase-fraction step; rebuild rho/mu/rhoPhi
    momentum             # density-weighted UEqn + buoyant/capillary source
    continuity           # p_rgh PISO correction; recompute static pressure p
    write_output         # NeoN write hook on write steps + step reporter

There is no outer PIMPLE residual loop — interFoam runs one outer corrector per
step (the inner PISO ``nCorrectors`` loop lives inside ``continuity``).
"""

from typing import Annotated, Any, Optional

from neofoam.framework.context import Context
from neofoam.framework.graph import DAGResolver
from neofoam.framework.initialization import Depends, StagedInitRunner
from neofoam.framework.model import ModelRuntime, ModelSpec
from neofoam.framework.operations import (
    IterativeOp,
    Operation,
    Operations,
    StepBuilder,
)
from neofoam.framework.solver import Solver
from neofoam.framework.types import OperationMetadata
from neofoam.solver.neon_runtime import ensure_neon_initialized

from .configs import ControlDictConfig
from .create_fields import create_init
from .models.alpha_advection.alphaAdvectionModel import alpha_advection_model
from .models.incompressibleVoFNeoNModel import incompressibleVoFNeoNModel
from .models.pressure_velocity.base import PressureVelocityAlgorithmNeoN
from .models.solution_loop import SolutionLoopPredicate


def _core_model(state: Any, spec_name: str) -> Any:
    """Find an instantiated core model by its spec name."""
    return next(
        m
        for m in state.core_models
        if isinstance(m, ModelRuntime) and m.spec.name == spec_name
    )


def _core_spec(state: Any, names: set[str]) -> Any:
    """Find a bare core-model ModelSpec by name (order-independent lookup)."""
    return next(
        m for m in state.core_models if isinstance(m, ModelSpec) and m.name in names
    )


incompressibleVoFNeoN = Solver("incompressibleVoFNeoN")

# Declare the full config schema on the spec, case-free: the solver's own
# configs plus the model families it owns. Two-phase transport / gravity carry
# no Python config classes — the NeoN C++ factories read
# constant/transportProperties / constant/g directly.
incompressibleVoFNeoN.config(ControlDictConfig)

incompressibleVoFNeoN.models(PressureVelocityAlgorithmNeoN, required=True)  # pick ONE
incompressibleVoFNeoN.models(alpha_advection_model, required=True)  # phase advection
incompressibleVoFNeoN.models(incompressibleVoFNeoNModel)  # optional: zero or more


@incompressibleVoFNeoN.initializer
def initialize(
    self: Any, init: Annotated[StagedInitRunner, Depends(create_init)]
) -> Context:
    """Initialize using the create_init factory with dependency injection."""
    return init.run()


@incompressibleVoFNeoN.execution_graph_step
def execution_graph(
    self: Any,
    ctx: Context,
    domain_name: Optional[str] = None,
) -> tuple[StepBuilder, Operations]:
    """Build the VoF time loop: alpha_advection -> momentum -> continuity."""
    _ = domain_name

    builder = StepBuilder()

    # Core models, looked up by spec name (order-independent): PIMPLE and the
    # alpha-advection model. Each spec is used as its own runtime binding for
    # operation building.
    pimple_model = _core_spec(
        self.state, {s.name for s in PressureVelocityAlgorithmNeoN.all_specs()}
    )
    alpha_model = _core_spec(self.state, {alpha_advection_model.name})
    algo_ops = Operations(pimple_model.build_operations_for(pimple_model))
    alpha_ops = Operations(alpha_model.build_operations_for(alpha_model))

    loop_ops = Operations(_core_model(self.state, "solutionLoop").operations)
    writer_ops = Operations(_core_model(self.state, "fieldWriter").operations)

    time_loop_op = Operation(
        func=IterativeOp(SolutionLoopPredicate()),
        metadata=OperationMetadata(op_name="time_loop"),
    )

    with builder.loop(time_loop_op) as time_builder:
        time_builder.step(loop_ops["set_time_step"])
        time_builder.step(loop_ops["increment_time"])
        time_builder.step(alpha_ops["alpha_advection"])
        time_builder.step(algo_ops["momentum"])
        time_builder.step(algo_ops["continuity"])
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

    Kokkos (via NeoN) may only be initialized once per process — the shared
    guard from the legacy port is reused (one process-wide flag).

    If ``log_file`` is given, fd 1 (stdout) is redirected to that file for the
    duration of the solve.
    """
    import os
    import sys
    from pathlib import Path

    ensure_neon_initialized(list(argv) if argv else ["incompressibleVoFNeoN"])

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
        solver = incompressibleVoFNeoN.instantiate(argv=argv or [])
        ctx = solver.initialize()

        builder, model_ops = solver.execution_graph(ctx=ctx)

        resolver = DAGResolver()
        resolved = resolver.resolve(builder, model_ops)
        resolved.operations.run(ctx)

        print("End")
        return ctx
    finally:
        if redirect and saved_fd is not None:
            sys.stdout.flush()
            os.dup2(saved_fd, 1)
            os.close(saved_fd)
