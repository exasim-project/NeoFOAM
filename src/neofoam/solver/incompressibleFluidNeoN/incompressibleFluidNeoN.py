# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""incompressibleFluidNeoN — solver entrypoint.

Framework port of :mod:`neofoam.solver.neoPimpleFoam` (the NeoN-backed PIMPLE
solver): the same SolverSpec / ModelSpec / StagedInit composition as
``incompressibleFluid``, with the NeoN bindings as the backend. The main
iteration loop is owned by the framework ``solution_loop`` engine; the legacy
``sync_run_times`` is decomposed into the ``timeStepConstraint`` fold
(courant / maxDeltaT contribution models) plus the ``NeoNTimeSync`` loop
backend (see ``models.solution_loop``).
"""

from typing import Annotated, Any, Optional

from neofoam.framework.context import Context
from neofoam.framework.graph import DAGResolver
from neofoam.framework.initialization import Depends, StagedInitRunner
from neofoam.framework.model import ModelRuntime
from neofoam.framework.operations import (
    IterativeOp,
    Operation,
    Operations,
    StepBuilder,
)
from neofoam.framework.solver import Solver
from neofoam.framework.tools import PreprocessConfig
from neofoam.framework.types import OperationMetadata
from neofoam.solver.neon_runtime import ensure_neon_initialized
from neofoam.tools.block_mesh import BlockMeshDictConfig
from neofoam.tools.snappy_hex_mesh import SnappyHexMeshDictConfig
from neofoam.turbulence import momentumTransportModel
from neofoam.turbulence.config import TurbulencePropertiesConfig
from neofoam.viscosity.config import TransportPropertiesConfig

from .configs import ControlDictConfig
from .create_fields import create_init
from .models.incompressibleFluidNeoNModel import incompressibleFluidNeoNModel
from .models.pressure_velocity.base import PressureVelocityAlgorithmNeoN
from .models.solution_loop import SolutionLoopPredicate


def _core_model(state: Any, spec_name: str) -> Any:
    """Find an instantiated core model by its spec name."""
    return next(
        m
        for m in state.core_models
        if isinstance(m, ModelRuntime) and m.spec.name == spec_name
    )


incompressibleFluidNeoN = Solver("incompressibleFluidNeoN")

# Declare the full config schema on the spec, case-free: the solver's own
# configs plus the model families it owns. Viscosity carries no Python config
# class (the NeoN C++ factory reads constant/transportProperties directly);
# turbulence is runtime-selected in create_fields from
# constant/turbulenceProperties (pure-Python NeoN family, C++ fallback).
incompressibleFluidNeoN.config(ControlDictConfig)
incompressibleFluidNeoN.config(
    PreprocessConfig
)  # mesh pipeline enable file (configs())
# The two mesh-input dicts: writer configs so the wizard/MCP fill and persist them
# like any other case file (blockMesh/snappyHexMesh read them at launch) — without
# them the sweep's mesh dimension is unavailable.
incompressibleFluidNeoN.config(BlockMeshDictConfig)
incompressibleFluidNeoN.config(SnappyHexMeshDictConfig)
# The NeoN C++ factories read constant/transportProperties /
# constant/turbulenceProperties directly at solve time; declaring the (single-phase)
# Python config classes here does not change that — it only surfaces the two
# mandatory files in the case-authoring schema so the wizard/agent fills them.
incompressibleFluidNeoN.config(TransportPropertiesConfig)  # molecular nu
incompressibleFluidNeoN.config(TurbulencePropertiesConfig)  # simulationType

incompressibleFluidNeoN.models(PressureVelocityAlgorithmNeoN, required=True)  # pick ONE
# The single turbulence family, shared with incompressibleFluid: binding it makes
# every registered turbulence model discoverable via the MCP model_catalog for the
# NeoN solver too (create_fields selects the native member with fallback=False).
incompressibleFluidNeoN.models(momentumTransportModel, required=True)  # nut + stress
incompressibleFluidNeoN.models(incompressibleFluidNeoNModel)  # optional: zero or more


@incompressibleFluidNeoN.initializer
def initialize(
    self: Any, init: Annotated[StagedInitRunner, Depends(create_init)]
) -> Context:
    """Initialize using the create_init factory with dependency injection."""
    return init.run()


@incompressibleFluidNeoN.execution_graph_step
def execution_graph(
    self: Any,
    ctx: Context,
    domain_name: Optional[str] = None,
) -> tuple[StepBuilder, Operations]:
    """Build the time-loop + PIMPLE inner-loop execution graph.

    One time step mirrors the legacy ``neoPimpleFoam`` body::

        set_time_step        # folds courant/maxDeltaT; NeoNTimeSync syncs rt.dt
        increment_time       # "Time = ..." print; advance; sync rt.t / rt.dt
        rotate_and_report    # rotate old times, Courant print, reset residuals
        [inner PIMPLE loop]  # momentum + continuity while control.loop(residuals)
        turbulence_correct   # once per step, after the PIMPLE loop
        write_output         # NeoN write hook on write steps + step reporter
    """
    _ = domain_name

    builder = StepBuilder()

    # The pressure-velocity algorithm (pimpleNeoN) is the first core model.
    # The spec is passed as its own runtime — it stores the configured
    # algorithm_type attribute directly.
    algorithm_model = self.state.core_models[0]
    algo_ops = Operations(algorithm_model._build_operations_for(algorithm_model))

    loop_ops = Operations(_core_model(self.state, "solutionLoop").operations)
    writer_ops = Operations(_core_model(self.state, "fieldWriter").operations)

    time_loop_op = Operation(
        func=IterativeOp(SolutionLoopPredicate()),
        metadata=OperationMetadata(op_name="time_loop"),
    )

    with builder.loop(time_loop_op) as time_builder:
        time_builder.step(loop_ops["set_time_step"])
        time_builder.step(loop_ops["increment_time"])
        time_builder.step(algo_ops["rotate_and_report"])

        with time_builder.loop(algo_ops["inner_loop"]) as inner_builder:
            inner_builder.step(algo_ops["momentum"])
            inner_builder.step(algo_ops["continuity"])

        time_builder.step(algo_ops["turbulence_correct"])
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
    guard from the legacy port is reused (one process-wide flag), so running
    the legacy and the framework solver in one process cannot double-init.

    If ``log_file`` is given, fd 1 (stdout) is redirected to that file for the
    duration of the solve.
    """
    import os
    import sys
    from pathlib import Path

    ensure_neon_initialized(list(argv) if argv else ["incompressibleFluidNeoN"])

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
        solver = incompressibleFluidNeoN.instantiate(argv=argv or [])
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
