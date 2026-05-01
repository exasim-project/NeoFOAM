# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""PIMPLE pressure-velocity algorithm using NeoN bindings.

PIMPLE = PISO + outer corrector loop. Has its own momentum/continuity
operations matching neoIcoFoam's algorithm.
"""

from typing import Annotated, Any, Callable

import neon._neon as nn
from neofoam import neofoam_bindings as nfb
from neofoam.solver.pisoControl import PisoControl

from neofoam.framework.context import Context, FieldUpdates
from neofoam.framework.initialization import field, model
from neofoam.framework.operations import (
    IterativeOp,
    Operation,
    Operations,
    SequentialOp,
)
from neofoam.framework.operation_wrapper import wrap_operation
from neofoam.framework.types import OperationMetadata
from neofoam.framework.model import Model
from pybFoam import Info


pimple = Model("Pimple")


class PimpleControl(PisoControl):
    """PIMPLE control — PISO correctors + outer corrector loop."""

    def __init__(
        self,
        n_outer_correctors: int = 1,
        n_correctors: int = 2,
        n_non_orthogonal_correctors: int = 0,
        momentum_predictor: bool = False,
    ) -> None:
        super().__init__(
            n_correctors=n_correctors,
            n_non_orthogonal_correctors=n_non_orthogonal_correctors,
            momentum_predictor=momentum_predictor,
        )
        self._n_outer: int = n_outer_correctors
        self._outer: int = 0

    def loop(self) -> bool:
        """Outer corrector loop. Returns True for nOuterCorrectors passes."""
        self._outer += 1
        if self._outer <= self._n_outer:
            return True
        self._outer = 0
        return False


# ---------------------------------------------------------------------------
# Field / model creators
# ---------------------------------------------------------------------------


def _create_neon_runtime(context: dict[str, Any]) -> Any:
    """Create NeoN runtime adapter from OpenFOAM Time."""
    run_time: Any = context["runtime"]
    rt = nfb.create_adapter_run_time(run_time)

    solvers = rt.fv_solution_dict.subDict("solvers")
    for name in ["p", "U", "nuTilda", "k", "epsilon"]:
        if solvers.contains(name):
            solvers.insert_dict(name, nfb.map_fv_solution(solvers.subDict(name)))
    rt.fv_schemes_dict = nfb.map_fv_schemes(rt.fv_schemes_dict)

    return rt


def _create_pimple_control(context: dict[str, Any]) -> PimpleControl:
    """Create PimpleControl from fvSolution PIMPLE sub-dictionary."""
    rt: Any = context["models.neon_runtime"]
    d = rt.fv_solution_dict.subDict("PIMPLE")
    return PimpleControl(
        n_outer_correctors=d.get_int("nOuterCorrectors")
        if d.contains("nOuterCorrectors")
        else 1,
        n_correctors=d.get_int("nCorrectors"),
        n_non_orthogonal_correctors=d.get_int("nNonOrthogonalCorrectors"),
        momentum_predictor=d.get_string("momentumPredictor").lower()
        in ("yes", "true", "on", "1")
        if d.contains("momentumPredictor")
        else False,
    )


def _create_nu_laminar_value(context: dict[str, Any]) -> float:
    """Read laminar viscosity scalar value from transportProperties."""
    rt: Any = context["models.neon_runtime"]
    return float(nfb.read_transport_viscosity(rt))


def _create_p(context: dict[str, Any]) -> Any:
    """Read pressure field via NeoN."""
    rt: Any = context["models.neon_runtime"]
    return nfb.read_scalar_volume_field(rt, "p")


def _create_U(context: dict[str, Any]) -> Any:
    """Read velocity field via NeoN."""
    rt: Any = context["models.neon_runtime"]
    return nfb.read_vector_volume_field(rt, "U")


def _create_phi(context: dict[str, Any]) -> Any:
    """Create face flux field via NeoN."""
    rt: Any = context["models.neon_runtime"]
    return nfb.create_phi(rt, "U")


def _create_pressure_reference(context: dict[str, Any]) -> Any:
    """Set pressure reference cell and value from PIMPLE dict."""
    rt: Any = context["models.neon_runtime"]
    return nfb.set_ref_cell(rt, "p", "PIMPLE")


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


@pimple.build
def build(self: Any) -> list[Any]:
    return [
        model("neon_runtime", _create_neon_runtime, depends_on=["runtime"]),
        model(
            "piso_control",
            _create_pimple_control,
            depends_on=["models.neon_runtime"],
        ),
        model(
            "nu_laminar_value",
            _create_nu_laminar_value,
            depends_on=["models.neon_runtime"],
        ),
        field("p", _create_p, depends_on=["models.neon_runtime"]),
        field("U", _create_U, depends_on=["models.neon_runtime"]),
        field("phi", _create_phi, depends_on=["models.neon_runtime"]),
        model(
            "pressure_reference",
            _create_pressure_reference,
            depends_on=["models.neon_runtime"],
        ),
    ]


# ---------------------------------------------------------------------------
# Loop controller
# ---------------------------------------------------------------------------


class _PimpleLoop:
    """Outer corrector loop for PIMPLE."""

    def __init__(self) -> None:
        self._control: PimpleControl | None = None

    def __call__(self, ctx: Context) -> bool:
        if self._control is None:
            self._control = ctx.models["piso_control"]
        return self._control.loop()


# ---------------------------------------------------------------------------
# Operations — own implementations matching neoIcoFoam
# ---------------------------------------------------------------------------


def _alias_operation(
    op_func: Callable[..., FieldUpdates],
    *,
    operation_name: str,
    depends_on: list[str],
) -> Operation:
    return Operation(
        func=SequentialOp(op_func),
        metadata=OperationMetadata(
            op_name=operation_name,
            depends_on=depends_on,
            shape="box",
            color="lightblue",
        ),
    )


@pimple.operation(operation_number="2.1")
def momentum(
    U: Any,
    phi: Any,
    p: Any,
    nuEff_surface: Any,
    neon_runtime: Annotated[Any, "models"],
    piso_control: Annotated[PimpleControl, "models"],
    div_dev_reff_correction: Any = None,
) -> FieldUpdates:
    """Build and optionally solve the momentum equation."""
    rt = neon_runtime

    expr = nn.imp.ddt(U) + nn.imp.div(phi, U) - nn.imp.laplacian(nuEff_surface, U)

    _ = div_dev_reff_correction

    UEqn = nfb.PDESolverVec3(expr, U, rt)
    ddt_scheme = UEqn.ddt_scheme()

    if piso_control.momentum_predictor():
        UEqn.solve_with_source(-1.0 * nn.exp.grad(p))
    else:
        UEqn.assemble()

    return FieldUpdates({"UEqn": UEqn, "ddt_scheme": ddt_scheme})


@pimple.operation(operation_number="2.2", depends_on=["momentum"])
def continuity(
    U: Any,
    p: Any,
    phi: Any,
    UEqn: Any,
    ddt_scheme: Any,
    neon_runtime: Annotated[Any, "models"],
    piso_control: Annotated[PimpleControl, "models"],
    pressure_reference: Annotated[tuple[int, float, bool], "models"],
) -> FieldUpdates:
    """PISO pressure-velocity corrector loop."""
    rt = neon_runtime
    p_ref_cell, p_ref_value, needs_ref = pressure_reference

    while piso_control.correct():
        rAU, hByA = nfb.compute_rau_and_hbya(UEqn)
        nfb.constrain_hbya(U, p, hByA)

        interp = nn.SurfaceInterpolationScalar(
            rt.executor, rt.nf_mesh, nn.TokenList(["linear"])
        )
        rAUf = interp.interpolate(rAU)
        rAUf.name = "rAUf"

        phiHbyA = nfb.flux(hByA) + rAUf * nfb.ddt_flux_corr(
            U, phi, rt.dt, ddt_scheme
        )

        while piso_control.correct_non_orthogonal():
            pEqn = nfb.PDESolverScalar(
                nn.imp.laplacian(rAUf, p) - nn.exp.div(phiHbyA),
                p,
                rt,
            )

            if needs_ref:
                pEqn.set_reference(p_ref_cell, p_ref_value)

            pEqn.solve()
            p.correct_boundary_conditions()

            if piso_control.final_non_orthogonal_iter():
                nfb.update_face_velocity(phiHbyA, pEqn, phi)

        nfb.update_velocity(hByA, rAU, p, U)
        U.correct_boundary_conditions()

    return FieldUpdates({"U": U, "p": p, "phi": phi})


# ---------------------------------------------------------------------------
# Operation collection
# ---------------------------------------------------------------------------


@pimple.operation_collection
def collected_operations(self: Any) -> Operations:
    model_ops = Operations()

    model_ops.add(
        Operation(
            func=IterativeOp(_PimpleLoop()),
            metadata=OperationMetadata(op_name="inner_loop"),
        )
    )

    wrapped_momentum = wrap_operation(momentum, self, pimple._dependency_resolver)
    wrapped_continuity = wrap_operation(continuity, self, pimple._dependency_resolver)

    model_ops.add(
        _alias_operation(wrapped_momentum, operation_name="momentum", depends_on=[])
    )
    model_ops.add(
        _alias_operation(
            wrapped_continuity, operation_name="continuity", depends_on=["momentum"]
        )
    )

    # Include turbulence correction if a turbulence model is attached
    if hasattr(self, "_turbulence_model") and self._turbulence_model is not None:
        turb = self._turbulence_model
        if hasattr(turb, "_build_operations_for"):
            turb_raw = turb._build_operations_for(turb)
            turb_ops = Operations(turb_raw)
            model_ops.add(turb_ops["turbulence_correction"])

    return model_ops
