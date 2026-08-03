# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""incompressibleVoF solver — interFoam-style VoF solver entrypoint.

Ported from the ``feat/incompressibleVoF`` solver_factory API to the
SolverSpec/ModelSpec API in ``stack/python_arch``. Mirrors the
``incompressibleFluid`` entrypoint with an extra ``alpha_advection`` step
at the head of the PIMPLE inner loop.

Time-loop mechanics (set_time_step with the interFoam dual Courant/alpha-Courant
adaptive dt, increment_time, write_output) stay solver-owned operations here —
they read the Foam::Time that ``create_time_mesh`` routes onto
``ctx.models["runtime"]``.
"""

import os
import sys
from typing import Annotated, Any, Optional, Protocol

import pybFoam as pyf
from pybFoam import (
    Info,
    dimensionedScalar,
    dimless,
    fvc,
    surfaceScalarField,
    volScalarField,
)

from neofoam.framework.context import Context, FieldUpdates
from neofoam.framework.graph import DAGResolver
from neofoam.framework.initialization import Depends, StagedInitRunner
from neofoam.framework.operations import (
    IterativeOp,
    Operation,
    Operations,
    StepBuilder,
)
from neofoam.framework.solver import Solver
from neofoam.framework.tools import PreprocessConfig
from neofoam.framework.types import OperationMetadata
from neofoam.tools.block_mesh import BlockMeshDictConfig
from neofoam.tools.snappy_hex_mesh import SnappyHexMeshDictConfig

from .configs import (
    ControlDictConfig,
    GravityConfig,
    TransportPropertiesConfig,
    TurbulencePropertiesConfig,
)
from .create_fields import create_init
from .models.alpha_advection import advectionModel  # noqa: F401 (registers members)
from .models.incompressibleVoFModel import incompressibleVoFModel
from .models.pressure_velocity.base import PressureVelocityAlgorithm

# Interface band for the alpha Courant number: only cells whose phase fraction
# straddles the interface (0.01 <= alpha1 <= 0.99) contribute. Mirrors
# interfaceProperties::nearInterface(), used by interFoam/interIsoFoam
# alphaCourantNo.H.
_ALPHA_CO_LO = dimensionedScalar(pyf.Word("alphaCoLo"), dimless, 0.01)
_ALPHA_CO_HI = dimensionedScalar(pyf.Word("alphaCoHi"), dimless, 0.99)


def compute_alpha_courant_number(
    phi: surfaceScalarField, alpha1: volScalarField
) -> tuple[float, float]:
    """Interface (alpha) Courant number, composed from generic pybFoam primitives.

    A pure-Python transcription of OpenFOAM's ``alphaCourantNo.H``: like the flow
    CFL number but the per-cell face-flux sum is masked to *near-interface cells*
    — ``interfaceProperties::nearInterface()``, i.e. the ``pos0`` band
    ``0.01 <= alpha1 <= 0.99`` on the **cell** values — before the ``gMax``/
    ``gSum`` reduction. The ``pos0`` mask is exactly 0/1, so the result is
    bitwise-identical to the native routine (which drives the alpha-CFL branch
    of the adaptive dt).

    Returns ``(alphaCoNum, meanAlphaCo)``.
    """
    mesh = phi.mesh()
    if mesh.nInternalFaces() == 0:
        return 0.0, 0.0

    dt = mesh.time().deltaTValue()
    # nearInterface(): pos0(alpha1 - 0.01) * pos0(0.99 - alpha1), 0/1 valued.
    near_interface = volScalarField(
        pyf.Word("nearInterface"),
        pyf.pos0(alpha1 - _ALPHA_CO_LO) * pyf.pos0(-(alpha1 - _ALPHA_CO_HI)),
    )
    sum_phi_alpha = volScalarField(
        pyf.Word("sumPhiAlpha"), near_interface * fvc.surfaceSum(pyf.mag(phi))
    ).internalField()

    volumes = mesh.V()
    alpha_co = 0.5 * pyf.gMax(sum_phi_alpha / volumes) * dt
    mean_alpha_co = 0.5 * (pyf.sum(sum_phi_alpha) / pyf.sum(volumes)) * dt
    return alpha_co, mean_alpha_co


class CorrectableModel(Protocol):
    def correct(self) -> None: ...


def time_loop(ctx: Context) -> bool:
    """Predicate: whether the outer time loop should continue."""
    return bool(ctx.models["runtime"].run())


def _core_spec(state: Any, names: set[str]) -> Any:
    """Find a core-model spec by name (order-independent lookup)."""
    return next(m for m in state.core_models if m.name in names)


incompressibleVoF = Solver("incompressibleVoF")

# Declare the full case-authoring schema on the spec, case-free (mirrors
# ``incompressibleFluid``): the solver's own configs plus the model families it
# owns. Every member's configs/fields join the schema; per-case detection (in
# create_fields) still picks which members run. Two-phase transport, gravity and
# turbulence are read by the C++ backend directly, so they are declared as solver
# configs (VoF has no single-phase viscosity/turbulence model families).
incompressibleVoF.config(ControlDictConfig)
incompressibleVoF.config(PreprocessConfig)  # mesh pipeline enable file (configs())
# The two mesh-input dicts: writer configs so the wizard/MCP fill and persist them
# like any other case file (blockMesh/snappyHexMesh read them at launch).
incompressibleVoF.config(BlockMeshDictConfig)
incompressibleVoF.config(SnappyHexMeshDictConfig)
incompressibleVoF.config(TransportPropertiesConfig)  # two-phase phases/nu/rho/sigma
incompressibleVoF.config(GravityConfig)  # constant/g
incompressibleVoF.config(TurbulencePropertiesConfig)  # simulationType

incompressibleVoF.models(advectionModel, required=True)  # alpha advection (pick ONE)
incompressibleVoF.models(PressureVelocityAlgorithm, required=True)  # VoF PIMPLE
incompressibleVoF.models(incompressibleVoFModel)  # optional: zero or more


@incompressibleVoF.initializer
def initialize(self: Any, init: Annotated[StagedInitRunner, Depends(create_init)]) -> Context:
    """Initialize fields and models via the staged-init runner."""
    return init.run()


@incompressibleVoF.execution_graph_step
def execution_graph(
    self: Any,
    ctx: Context,
    domain_name: Optional[str] = None,
) -> tuple[StepBuilder, Operations]:
    """Build the VoF time loop: alpha_advection -> momentum -> continuity."""
    _ = domain_name

    builder = StepBuilder()

    # Core models, looked up by spec name (order-independent): the active
    # alpha-advection scheme and PIMPLE. Each spec is used as its own runtime
    # binding for operation building.
    alpha_model = _core_spec(self.state, set(advectionModel.registered_names()))
    pimple_model = _core_spec(
        self.state, {spec.name for spec in PressureVelocityAlgorithm.all_specs()}
    )
    alpha_ops = Operations(alpha_model.build_operations_for(alpha_model))
    pimple_ops = Operations(pimple_model.build_operations_for(pimple_model))

    # Solver-owned time-loop operations.
    ops = self.operations

    time_loop_op = Operation(
        func=IterativeOp(time_loop),
        metadata=OperationMetadata(op_name="time_loop"),
    )

    with builder.loop(time_loop_op) as time_builder:
        time_builder.step(ops["set_time_step"])
        time_builder.step(ops["increment_time"])

        with time_builder.loop(pimple_ops["inner_loop"]) as inner_builder:
            # interFoam opens every outer corrector with mesh.update(); a static
            # case sees a no-op here.
            inner_builder.step(pimple_ops["mesh_update"])
            inner_builder.step(alpha_ops["alpha_advection"])
            inner_builder.step(pimple_ops["momentum"])
            inner_builder.step(pimple_ops["continuity"])
            inner_builder.step(ops["turbulence_correction"])

        time_builder.step(ops["write_output"])

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
    """Run one full incompressibleVoF simulation and return the final Context.

    If ``log_file`` is given, fd 1 (stdout) is redirected to that file for the
    duration of the solve so C++ ``Info`` output ends up there.
    """
    redirect = log_file is not None
    saved_fd: Optional[int] = None
    if redirect:
        log_path = str(log_file)
        sys.stdout.flush()
        saved_fd = os.dup(1)
        log_fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        os.dup2(log_fd, 1)
        os.close(log_fd)

    try:
        solver = incompressibleVoF.instantiate(argv=argv or [])
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


# ---------------------------------------------------------------------------
# Solver-owned time-loop operations
# ---------------------------------------------------------------------------


@incompressibleVoF.operation()
def set_time_step(
    ctx: Context,
    phi: surfaceScalarField,
    alpha1: volScalarField,
    runtime: Annotated[Any, "models"],
) -> None:
    """Adjust the time step from both flow-CFL and alpha-CFL (interFoam
    setInitialDeltaT.H + setDeltaT.H)."""
    # Re-read system/controlDict every step — faithful to OpenFOAM's
    # runTimeModifiable handling of adjustTimeStep/maxCo/maxAlphaCo/maxDeltaT
    # (Time.controlDict() is not bound, so the file is read directly). Missing
    # keys fall back to the interFoam defaults; a malformed value is a fatal
    # OpenFOAM IO error (not catchable from Python), exactly as in interFoam.
    ctrl_dict = pyf.dictionary.read("system/controlDict")

    if not ctrl_dict.getOrDefault[bool]("adjustTimeStep", False):
        return

    max_co = float(ctrl_dict.getOrDefault[float]("maxCo", 1.0))
    max_alpha_co = float(ctrl_dict.getOrDefault[float]("maxAlphaCo", 1.0))
    max_delta_t = float(ctrl_dict.getOrDefault[float]("maxDeltaT", 1.0))

    # setInitialDeltaT.H: interFoam runs it once (with CourantNo.H) before the
    # loop, so on the first pass it precedes the setDeltaT.H below. The reduction
    # is undamped and can only lower the step, but the setDeltaT call it makes
    # snaps onto the write time, which is what the damped pass then grows from.
    if runtime.timeIndex() == 0:
        initial_co_num = pyf.computeCFLNumber(phi)[0]
        if initial_co_num > 1e-15:
            dt0 = runtime.deltaTValue()
            runtime.setDeltaT(min(max_co * dt0 / initial_co_num, min(dt0, max_delta_t)))

    # Flow + interface (alpha) Courant numbers.
    maxCoNum, meanCoNum = pyf.computeCFLNumber(phi)
    alphaCoNum, meanAlphaCo = compute_alpha_courant_number(phi, alpha1)

    Info(f"Courant Number mean: {meanCoNum:.4g}  max: {maxCoNum:.4g}")
    Info(f"Interface Courant Number mean: {meanAlphaCo:.4g}  max: {alphaCoNum:.4g}")

    # Mirror interFoam setDeltaT.H: limit by both Co constraints, allow a ramped
    # increase (at most +20% per step) when Co is well below the limit.
    current_dt = runtime.deltaTValue()
    max_delta_t_fact = min(
        max_co / (maxCoNum + 1e-15),
        max_alpha_co / (alphaCoNum + 1e-15),
    )
    delta_t_fact = min(min(max_delta_t_fact, 1.0 + 0.1 * max_delta_t_fact), 1.2)
    new_dt = min(delta_t_fact * current_dt, max_delta_t)
    runtime.setDeltaT(new_dt)
    Info(f"deltaT = {new_dt:.6g}")


@incompressibleVoF.operation()
def increment_time(runtime: Annotated[Any, "models"]) -> None:
    """Print current simulation time and advance."""
    Info(f"Time = {runtime.timeName()}")
    runtime.increment()


@incompressibleVoF.operation(depends_on=["continuity"])
def turbulence_correction(
    turbulence: Annotated[Optional[CorrectableModel], "models"],
    pimple_control: Annotated[Any, "models"],
) -> FieldUpdates:
    """Correct the two-phase turbulence model after pressure-velocity coupling.

    interFoam/interIsoFoam guard the call with ``if (pimple.turbCorr())``, which
    is ``!turbOnFinalIterOnly || finalIter()``: with the native default the
    turbulence is corrected **once per time step**, on the last outer corrector,
    not once per outer corrector. Under ``nOuterCorrectors > 1`` an ungated call
    feeds a ``nut`` native has not yet updated into the next outer iteration's
    momentum assembly, which moves ``p_rgh`` and from there ``phi`` and alpha.
    """
    if turbulence and pimple_control.turbCorr():
        turbulence.correct()
    return FieldUpdates({})


@incompressibleVoF.operation(depends_on=["turbulence_correction"])
def write_output(runtime: Annotated[Any, "models"]) -> None:
    """Write fields to disk and report execution time."""
    runtime.write(True)
    runtime.printExecutionTime()
