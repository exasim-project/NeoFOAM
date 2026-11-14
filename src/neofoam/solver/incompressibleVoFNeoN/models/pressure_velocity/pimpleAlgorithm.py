# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

# NB: no ``from __future__ import annotations`` — the dependency_resolver
# matches ``Annotated[..., "models"]`` live.

"""interFoam's UEqn.H and pEqn.H on NeoN, laminar and without surface tension."""

from dataclasses import dataclass
from typing import Annotated, Any, Callable

import neon._neon as nn
import pybFoam as pyf

from neofoam import neofoam_bindings as nfb
from neofoam.fields import (
    CalculatedBC,
    CyclicBC,
    EmptyBC,
    FixedFluxPressureBC,
    FixedValueBC,
    GenericBC,
    InletOutletBC,
    NoSlipBC,
    PressureInletOutletVelocityBC,
    Scalar,
    SlipBC,
    SymmetryBC,
    SymmetryPlaneBC,
    Vector,
    ZeroGradientBC,
)
from neofoam.foam import fvSchemes, fvSolution
from neofoam.framework.context import FieldUpdates
from neofoam.framework.initialization import field, model
from neofoam.framework.model import Model
from neofoam.framework.operations import Operation, Operations, SequentialOp
from neofoam.framework.types import OperationMetadata

__all__ = ["PimpleVoFControls", "PimpleVoFWorkspace", "pimpleVoFNeoN"]

pimpleVoFNeoN = Model("PimpleVoFNeoN")

# Schema-only, as on the pybFoam VoF pimple spec: surfaced through configurations(solver).
PimpleFvSchemes = pimpleVoFNeoN.config(fvSchemes)
PimpleFvSolution = pimpleVoFNeoN.config(fvSolution)
# The corrector counts _read_pimple_controls reads (only nOuterCorrectors 1 runs).
PimpleFvSolution.add_controls(
    "PIMPLE", nOuterCorrectors=int, nCorrectors=int, nNonOrthogonalCorrectors=int
)

pimpleVoFNeoN.field(
    "U",
    dimensions=[0, 1, -1, 0, 0, 0, 0],
    value_type=Vector,
    allowed_bcs=[
        NoSlipBC,
        FixedValueBC,
        ZeroGradientBC,
        SlipBC,
        InletOutletBC,
        PressureInletOutletVelocityBC,
        EmptyBC,
        SymmetryBC,
        SymmetryPlaneBC,
        CyclicBC,
        GenericBC,
    ],
    write=True,
)
pimpleVoFNeoN.field(
    "p_rgh",
    dimensions=[1, -1, -2, 0, 0, 0, 0],
    value_type=Scalar,
    allowed_bcs=[
        FixedFluxPressureBC,
        FixedValueBC,
        ZeroGradientBC,
        InletOutletBC,
        CalculatedBC,
        EmptyBC,
        SymmetryBC,
        SymmetryPlaneBC,
        CyclicBC,
        GenericBC,
    ],
    write=True,
)


@dataclass
class PimpleVoFControls:
    """The ``PIMPLE`` corrector counts and the running continuity error.

    Read once at build; the continuity step updates the error, so one time loop owns it.

    Example::

        PimpleVoFControls(n_correctors=2, n_non_orthogonal_correctors=0)
    """

    n_correctors: int
    n_non_orthogonal_correctors: int
    cumulative_continuity_error: float = 0.0


@dataclass(frozen=True)
class PimpleVoFWorkspace:
    """Fields every step overwrites in place, allocated once for the run.

    Use it through the ``workspace`` model so the time loop does not allocate; the fields'
    values are meaningful only inside the step that wrote them, on one time loop.

    Example::

        def operation(workspace: Annotated[PimpleVoFWorkspace, "models"]) -> None: ...
    """

    rho_nu: Any
    rho_nu_f: Any
    nut: Any
    rAUf: Any
    rho_rAU: Any
    rho_rAUf: Any
    force: Any
    phig: Any
    p_flux: Any


def _read_pimple_controls() -> PimpleVoFControls:
    """The ``PIMPLE`` dict of ``system/fvSolution``, refusing what this solver does not run."""
    pimple = pyf.dictionary.read("system/fvSolution").subDict("PIMPLE")
    if pimple.getOrDefault[bool]("momentumPredictor", True):
        raise NotImplementedError(
            "incompressibleVoFNeoN: momentumPredictor yes is not implemented; set it to no"
        )
    if pimple.getOrDefault[int]("nOuterCorrectors", 1) != 1:
        raise NotImplementedError("incompressibleVoFNeoN: only nOuterCorrectors 1 is implemented")
    return PimpleVoFControls(
        n_correctors=pimple.getOrDefault[int]("nCorrectors", 1),
        n_non_orthogonal_correctors=pimple.getOrDefault[int]("nNonOrthogonalCorrectors", 0),
    )


def _check_no_pressure_reference(runtime: Any) -> None:
    """A closed domain needs p_rgh levelling (pEqn.H's setReference), which is not ported."""
    _cell, _value, needs_reference = nfb.set_ref_cell(runtime, "p_rgh", "PIMPLE")
    if needs_reference:
        raise NotImplementedError(
            "incompressibleVoFNeoN: p_rgh needs a pressure reference (no fixed-value patch); "
            "closed domains are not implemented"
        )


@pimpleVoFNeoN.build
def build(self: Any) -> list[Any]:
    """U, p_rgh, gh/ghf, the static pressure p, the controls and the workspace."""

    def create_p_rgh(ctx: dict[str, Any]) -> Any:
        runtime = ctx["_neon_runtime"]
        p_rgh = nfb.read_scalar_volume_field(runtime, "p_rgh")
        _check_no_pressure_reference(runtime)
        return p_rgh

    def create_p(ctx: dict[str, Any]) -> Any:
        p = nfb.create_uniform_volume_field(ctx["_neon_runtime"], "p", 0.0)
        nfb.update_static_pressure(ctx["fields.p_rgh"], ctx["fields.rho"], ctx["fields.gh"], p)
        return p

    def create_workspace(ctx: dict[str, Any]) -> PimpleVoFWorkspace:
        runtime = ctx["_neon_runtime"]
        phi = ctx["fields.phi"]

        def volume(name: str) -> Any:
            return nfb.create_uniform_volume_field(runtime, name, 0.0)

        def surface(name: str) -> Any:
            face_field = phi * 0.0
            face_field.name = name
            return face_field

        return PimpleVoFWorkspace(
            rho_nu=volume("rhoNu"),
            rho_nu_f=surface("rhoNuf"),
            nut=volume("nut"),
            rAUf=surface("rAUf"),
            rho_rAU=volume("rhorAU"),
            rho_rAUf=surface("rhorAUf"),
            force=surface("gravityFaceForce"),
            phig=surface("phig"),
            p_flux=surface("pFlux"),
        )

    return [
        field(
            "U",
            lambda ctx: nfb.read_vector_volume_field(ctx["_neon_runtime"], "U"),
            depends_on=["_neon_runtime"],
            write=True,
        ),
        field("p_rgh", create_p_rgh, depends_on=["_neon_runtime"], write=True),
        field("gh", lambda ctx: nfb.create_gh(ctx["_neon_runtime"]), depends_on=["_neon_runtime"]),
        field(
            "ghf", lambda ctx: nfb.create_ghf(ctx["_neon_runtime"]), depends_on=["_neon_runtime"]
        ),
        field("p", create_p, depends_on=["fields.p_rgh", "fields.rho", "fields.gh"], write=True),
        model("pimple_controls", lambda _ctx: _read_pimple_controls()),
        model("workspace", create_workspace, depends_on=["_neon_runtime", "fields.phi"]),
        # gradSchemes/grad(U) as the case writes it (default when absent).
        model(
            "grad_op",
            lambda ctx: nfb.GradScheme(ctx["_neon_runtime"], "U"),
            depends_on=["_neon_runtime"],
        ),
    ]


@pimpleVoFNeoN.operation(operation_number="2.1")
@PimpleFvSchemes.add(
    ddt="ddt(rho,U)",
    div=["div(rhoPhi,U)", "div(((rho*nuEff)*dev2(T(grad(U)))))"],
    grad="grad(U)",
    laplacian="default",
)
@PimpleFvSolution.add("U")
def momentum(
    U: Any,
    phi: Any,
    rho: Any,
    nu: Any,
    rhoPhi: Any,
    workspace: Annotated[PimpleVoFWorkspace, "models"],
    grad_op: Annotated[Any, "models"],
    neon_runtime: Annotated[Any, "models"],
) -> FieldUpdates:
    """Assemble interFoam's UEqn (no predictor solve: momentumPredictor no)."""
    # The old times ddtCorr and ddt(rho, U) read: U and phi at the start of the step.
    nn.rotate_old_times(U)
    nn.rotate_old_times(phi)

    # divDevRhoReff of the laminar model: nuEff = nu (calcNu), the coefficient rho*nu.
    nfb.multiply(rho, nu, workspace.rho_nu)
    nfb.linear_interpolate(neon_runtime, workspace.rho_nu, out=workspace.rho_nu_f)
    # Kept alive through continuity: the stress operator holds a reference to it.
    grad_u = grad_op.grad_tensor(U)

    UEqn = nfb.PDESolverVec3(
        nn.imp.ddt(rho, U)
        + nn.imp.div(rhoPhi, U)
        - nn.imp.laplacian(workspace.rho_nu_f, U)
        + nfb.viscous_stress(workspace.rho_nu, workspace.nut, grad_u),
        U,
        neon_runtime,
    )
    # rhoPhi is formed for an Euler ddt(rho,U) (alphaEqn.H's first branch).
    if UEqn.ddt_scheme() != nfb.DdtScheme.BDF1:
        raise NotImplementedError("incompressibleVoFNeoN: only an Euler ddt(rho,U) is implemented")
    UEqn.set_final_iter(True)
    UEqn.relax()
    UEqn.assemble_and_relax()
    return FieldUpdates({"UEqn": UEqn, "grad_u": grad_u, "U": U})


def _predicted_flux(
    UEqn: Any, fields: tuple[Any, Any, Any, Any], workspace: PimpleVoFWorkspace, runtime: Any
) -> tuple[Any, Any, Any]:
    """pEqn.H up to constrainPressure: ``(rAU, HbyA, phiHbyA)``; rAUf and phig in workspace."""
    U, phi, rho, p_rgh = fields
    rAU, hByA = nfb.compute_rau_and_hbya(UEqn)
    nfb.linear_interpolate(runtime, rAU, out=workspace.rAUf)
    nfb.constrain_hbya(U, p_rgh, hByA)

    # The ddtCorr weight is interpolate(rho*rAU); phig and the laplacian take plain rAUf.
    nfb.multiply(rho, rAU, workspace.rho_rAU)
    nfb.linear_interpolate(runtime, workspace.rho_rAU, out=workspace.rho_rAUf)
    ddt_corr = nfb.ddt_flux_corr(U, phi, runtime.dt, UEqn.ddt_scheme())
    nfb.multiply(workspace.rho_rAUf, ddt_corr, ddt_corr)
    phi_hbya = nfb.flux(hByA)
    nfb.linear_combination(1.0, phi_hbya, 1.0, ddt_corr, phi_hbya)

    nfb.buoyancy_flux(workspace.force, workspace.rAUf, out=workspace.phig)
    nfb.linear_combination(1.0, phi_hbya, 1.0, workspace.phig, phi_hbya)
    # After phig: the fixedFluxPressure gradient balances the buoyancy flux at the walls.
    nfb.constrain_pressure(p_rgh, U, phi_hbya, workspace.rAUf)
    return rAU, hByA, phi_hbya


def _solve_pressure(
    p_rgh: Any, phi_hbya: Any, rAUf: Any, final_solves: list[bool], runtime: Any
) -> Any:
    """The non-orthogonal p_rgh solves, one per ``final_solves`` flag; the last equation."""
    for final_solve in final_solves:
        p_eqn = nfb.PDESolverScalar(
            nn.imp.laplacian(rAUf, p_rgh) - nn.exp.div(phi_hbya), p_rgh, runtime
        )
        p_eqn.set_final_iter(final_solve)
        p_eqn.solve()
        p_rgh.correct_boundary_conditions()
    return p_eqn


def _report_continuity_errors(phi: Any, controls: PimpleVoFControls, runtime: Any) -> None:
    """continuityErrs.H."""
    sum_local, global_err = nfb.compute_continuity_error(phi, runtime)
    controls.cumulative_continuity_error += global_err
    print(
        f"time step continuity errors : sum local = {sum_local}, "
        f"global = {global_err}, cumulative = {controls.cumulative_continuity_error}"
    )


@pimpleVoFNeoN.operation(operation_number="2.2", depends_on=["momentum"])
@PimpleFvSchemes.add(
    laplacian="laplacian(rAUf,p_rgh)",
    interpolation=["flux(U)", "flux(HbyA)", "interpolate(rho*rAU)"],
    snGrad=["snGrad(p_rgh)", "snGrad(rho)"],
)
@PimpleFvSolution.add("p_rgh")
def continuity(
    U: Any,
    phi: Any,
    rho: Any,
    p_rgh: Any,
    p: Any,
    gh: Any,
    ghf: Any,
    UEqn: Any,
    pimple_controls: Annotated[PimpleVoFControls, "models"],
    workspace: Annotated[PimpleVoFWorkspace, "models"],
    neon_runtime: Annotated[Any, "models"],
) -> FieldUpdates:
    """The p_rgh correctors of pEqn.H, then ``p = p_rgh + rho*gh``."""
    # -ghf*snGrad(rho): rho is fixed across the correctors, so once per step.
    nfb.gravity_face_force(neon_runtime, ghf, rho, out=workspace.force)
    n_correctors = pimple_controls.n_correctors
    n_non_orthogonal = pimple_controls.n_non_orthogonal_correctors
    for corrector in range(n_correctors):
        rAU, hByA, phi_hbya = _predicted_flux(UEqn, (U, phi, rho, p_rgh), workspace, neon_runtime)
        # p_rgh.select(pimple.finalInnerIter()): p_rghFinal on the very last solve only.
        final_solves = [
            corrector == n_correctors - 1 and non_orthogonal == n_non_orthogonal
            for non_orthogonal in range(n_non_orthogonal + 1)
        ]
        p_eqn = _solve_pressure(p_rgh, phi_hbya, workspace.rAUf, final_solves, neon_runtime)
        nfb.update_face_velocity(phi_hbya, p_eqn, phi)
        # p_rghEqn.flux() as phiHbyA - phi, the flux the face update just applied.
        nfb.linear_combination(1.0, phi_hbya, -1.0, phi, workspace.p_flux)
        nfb.update_velocity_buoyant(hByA, rAU, workspace.phig, workspace.p_flux, workspace.rAUf, U)
        U.correct_boundary_conditions()
        _report_continuity_errors(phi, pimple_controls, neon_runtime)
    nfb.update_static_pressure(p_rgh, rho, gh, p)
    return FieldUpdates({"U": U, "phi": phi, "p_rgh": p_rgh, "p": p})


def _sequential(func: Callable[..., Any], name: str, depends_on: list[str]) -> Operation:
    return Operation(
        func=SequentialOp(func),
        metadata=OperationMetadata(
            op_name=name, depends_on=depends_on, shape="box", color="lightblue"
        ),
    )


@pimpleVoFNeoN.operation_collection
def collected_operations(self: Any) -> Operations:
    """Expose ``momentum`` and ``continuity`` (one outer corrector: no inner loop)."""
    model_ops = Operations()
    model_ops.add(_sequential(pimpleVoFNeoN.wrap_operation(momentum, self), "momentum", []))
    model_ops.add(
        _sequential(pimpleVoFNeoN.wrap_operation(continuity, self), "continuity", ["momentum"])
    )
    return model_ops
