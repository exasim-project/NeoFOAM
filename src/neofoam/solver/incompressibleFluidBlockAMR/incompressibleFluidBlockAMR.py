# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""incompressibleFluidBlockAMR — solver entrypoint.

A framework ``SolverSpec`` mirroring ``incompressibleFluid``'s composition
(SolverSpec / ModelSpec / StagedInit) but backed by the block-structured AMReX
DSL ``blockamr`` instead of pybFoam.

One time step is a Chorin fractional-step projection::

    set_time_step     # fold time-step constraints; push dt onto the state
    increment_time    # "Time = ..." print; advance the LoopState
    momentum          # interpolate -> MAC project -> momentum predictor
    continuity        # pressure Poisson -> velocity correct -> IBM apply
    write_output      # PlotfileWriteHook writes an AMReX plotfile on write steps

The outer time loop, write control, and field-writer are the reused framework
core Models; the block-structured backends are injected in ``create_fields``.
"""

import os
import sys
from pathlib import Path
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
from neofoam.framework.types import OperationMetadata
from neofoam.algorithms.solution_loop.solution_loop import SolutionLoopPredicate

from .configs import (
    BlockAMRSolutionConfig,
    ControlDictConfig,
    FvSchemesConfig,
    MeshDictConfig,
    PSolutionConfig,
    USolutionConfig,
)
from .create_fields import create_init
from .models.incompressibleFluidBlockAMRModel import incompressibleFluidBlockAMRModel
from .models.projection.base import ProjectionAlgorithm


def _core_model(state: Any, spec_name: str) -> Any:
    """Find an instantiated core model by its spec name."""
    return next(
        m
        for m in state.core_models
        if isinstance(m, ModelRuntime) and m.spec.name == spec_name
    )


incompressibleFluidBlockAMR = Solver("incompressibleFluidBlockAMR")

# Declare the full config schema on the spec, case-free.
incompressibleFluidBlockAMR.config(ControlDictConfig)
incompressibleFluidBlockAMR.config(MeshDictConfig)
incompressibleFluidBlockAMR.config(FvSchemesConfig)
incompressibleFluidBlockAMR.config(USolutionConfig)
incompressibleFluidBlockAMR.config(PSolutionConfig)
incompressibleFluidBlockAMR.config(BlockAMRSolutionConfig)

incompressibleFluidBlockAMR.models(ProjectionAlgorithm, required=True)  # pick ONE
incompressibleFluidBlockAMR.models(incompressibleFluidBlockAMRModel)  # optional: 0+


@incompressibleFluidBlockAMR.initializer
def initialize(
    self: Any, init: Annotated[StagedInitRunner, Depends(create_init)]
) -> Context:
    """Initialize using the create_init factory with dependency injection."""
    return init.run()


@incompressibleFluidBlockAMR.execution_graph_step
def execution_graph(
    self: Any,
    ctx: Context,
    domain_name: Optional[str] = None,
) -> tuple[StepBuilder, Operations]:
    """Build the time-loop + projection execution graph."""
    _ = domain_name

    builder = StepBuilder()

    algorithm_model = self.state.core_models[0]
    algo_ops = Operations(algorithm_model.operations)

    loop_ops = Operations(_core_model(self.state, "solutionLoop").operations)
    writer_ops = Operations(_core_model(self.state, "fieldWriter").operations)

    time_loop_op = Operation(
        func=IterativeOp(SolutionLoopPredicate()),
        metadata=OperationMetadata(op_name="time_loop"),
    )

    with builder.loop(time_loop_op) as time_builder:
        time_builder.step(loop_ops["set_time_step"])
        time_builder.step(loop_ops["increment_time"])
        time_builder.step(algo_ops["momentum"])
        time_builder.step(algo_ops["continuity"])
        time_builder.step(writer_ops["write_output"])

    model_ops = Operations()
    for opt in self.state.optional_models:
        model_ops.add(opt.operations)

    return builder, model_ops


def run(
    argv: Optional[list[str]] = None,
    log_file: Optional[Any] = None,
) -> Context:
    """Run one full simulation and return the final Context.

    AMReX must be initialized for the whole solve — the run is wrapped in
    ``blockamr.runtime()`` (init/finalize), and the executor (CPU/GPU) is set
    from ``system/controlDict``'s ``executor`` before the mesh/engine are built.

    If ``log_file`` is given, fd 1 (stdout) is redirected there for the solve.
    """
    # blockamr stays a local import: it pulls the AMReX/GPU extension, and the
    # package is kept importable GPU-free (config introspection, CLI listing)
    # by never importing it at module scope.
    import blockamr  # noqa: PLC0415

    redirect = log_file is not None
    saved_fd: Optional[int] = None
    if redirect:
        log_path = Path(str(log_file))
        sys.stdout.flush()
        saved_fd = os.dup(1)
        log_fd = os.open(str(log_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        os.dup2(log_fd, 1)
        os.close(log_fd)

    def _solve() -> Context:
        solver = incompressibleFluidBlockAMR.instantiate(argv=argv or [])
        ctx = solver.initialize()

        builder, model_ops = solver.execution_graph(ctx=ctx)

        resolver = DAGResolver()
        resolved = resolver.resolve(builder, model_ops)
        resolved.operations.run(ctx)

        print("End")
        return ctx

    try:
        try:
            executor = ControlDictConfig.load(case_dir=Path(".")).executor
        except Exception:
            executor = "cpu"
        blockamr.set_executor(executor)

        # AMReX may only be initialized once per process. If a caller (e.g. a
        # test session fixture) already opened a blockamr runtime, reuse it;
        # otherwise open our own — but do NOT finalize. ``amrex::Finalize`` frees
        # the AMReX arena allocator's device memory, which aborts with
        # ``CUDA error 709: context is destroyed`` when the co-loaded JAX/NeoN
        # Kokkos-CUDA runtimes have already dropped the shared CUDA context
        # (neofoam always co-loads NeoN). Leaving AMReX un-finalized lets the OS
        # reclaim GPU memory at process exit instead. ``__enter__`` is depth-
        # guarded, so a second ``run()`` in one process reuses the session.
        already_initialized = getattr(blockamr, "initialized", lambda: False)()
        if not already_initialized:
            blockamr.runtime().__enter__()
        return _solve()
    finally:
        if redirect and saved_fd is not None:
            sys.stdout.flush()
            os.dup2(saved_fd, 1)
            os.close(saved_fd)
