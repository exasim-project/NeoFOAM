# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""NeoN Spalart-Allmaras turbulence model for incompressibleFluidNeon solver."""

from typing import Annotated, Any

import neon._neon as nn
import pybFoam as pyf
from neofoam import neofoam_bindings as nfb

from neofoam.framework.context import FieldUpdates
from neofoam.framework.initialization import field, model
from neofoam.framework.model import Model
from neofoam.framework.operations import Operation, Operations, SequentialOp
from neofoam.framework.operation_wrapper import wrap_operation
from neofoam.framework.types import OperationMetadata
from neofoam.solver.incompressibleFluid.models.spalartAllmaras import (
    SpalartAllmarasConfig,
)


sa_neon = Model("spalart_allmaras_neon")


def detect_sa() -> Any | None:
    """Check turbulenceProperties for SA model. Returns model or None."""
    try:
        props = pyf.dictionary.read("constant/turbulenceProperties")
        if not props.found("RAS"):
            return None
        ras = props.subDict("RAS")
        if ras.found("RASModel") and ras.get_word("RASModel") == "SpalartAllmaras":
            return sa_neon
        return None
    except Exception:
        return None


# --- Field creation functions ---


def _create_nuTilda(context: dict[str, Any]) -> Any:
    """Read nuTilda field via NeoN."""
    rt: Any = context["models.neon_runtime"]
    return nfb.read_scalar_volume_field(rt, "nuTilda")


def _create_nut(context: dict[str, Any]) -> Any:
    """Read nut field via NeoN."""
    rt: Any = context["models.neon_runtime"]
    return nfb.read_scalar_volume_field(rt, "nut")


def _create_d(context: dict[str, Any]) -> Any:
    """Compute wall distance field via NeoN."""
    rt: Any = context["models.neon_runtime"]
    return nfb.compute_wall_distance(rt)


def _create_sa_config(context: dict[str, Any]) -> SpalartAllmarasConfig:
    """Create SA configuration with default constants."""
    return SpalartAllmarasConfig()


def _create_div_dev_reff_correction(context: dict[str, Any]) -> Any:
    """Create zero-initialized divDevReff correction field (Vec3)."""
    rt: Any = context["models.neon_runtime"]
    corr = nn.VectorVolumeField(rt.executor, "div_dev_reff_correction", rt.nf_mesh)
    nn.fill(corr.internal_vector(), nn.Vec3(0, 0, 0))
    return corr


# --- Build step ---


@sa_neon.build
def build(self: Any) -> list[Any]:
    return [
        field("nuTilda", _create_nuTilda, depends_on=["models.neon_runtime"]),
        field("nut", _create_nut, depends_on=["models.neon_runtime"]),
        field("d", _create_d, depends_on=["models.neon_runtime"]),
        field(
            "div_dev_reff_correction",
            _create_div_dev_reff_correction,
            depends_on=["models.neon_runtime"],
        ),
        model("sa_config", _create_sa_config, depends_on=[]),
    ]


# --- SA transport equation solver ---


def correct(
    cfg: SpalartAllmarasConfig,
    rt: Any,
    nuTilda: Any,
    nut: Any,
    U: Any,
    phi: Any,
    d: Any,
    nu_value: float,
) -> None:
    """Solve SA transport equation and update nut in-place."""
    # 0. Rotate old-time fields (required before ddt assembly, see neoIcoFoam.py)
    nn.rotate_old_times(nuTilda)

    # Safety floor for wall distance — boundary faces at walls have d=0,
    # which causes division by zero in d**2 denominators.
    SMALL = 1e-10
    d_safe = nn.field_max(d, SMALL)

    # 1. Auxiliary fields
    chi = nuTilda / nu_value
    fv1 = chi**3 / (chi**3 + cfg.Cv1**3)
    fv2 = 1.0 - chi / (1.0 + chi * fv1)

    # 2. Vorticity magnitude: sqrt(2)*mag(skew(gradU)) — matches OpenFOAM
    grad_U = nn.exp.grad_field(U)
    Omega = nn.field_max(2.0**0.5 * nn.mag(nn.skew(grad_U)), 0.0)

    # 3. Modified vorticity (Stilda)
    Stilda = nn.field_max(
        Omega + fv2 * nuTilda / (cfg.kappa**2 * d_safe**2),
        cfg.Cs * Omega,
    )

    # 4. fw function — matches OF: g*pow((1+Cw3^6)/(g^6+Cw3^6), 1/6)
    Stilda_safe = nn.field_max(Stilda, SMALL)
    r = nn.field_min(nuTilda / (Stilda_safe * cfg.kappa**2 * d_safe**2), 10.0)
    g = r + cfg.Cw2 * (r**6 - r)
    g6 = g**6
    fw_frac = (1.0 + cfg.Cw3**6) * (g6 + cfg.Cw3**6) ** (-1.0)
    fw = g * nn.field_pow(fw_frac, 1.0 / 6.0)

    # 5. Source terms — match OpenFOAM's treatment:
    #   Production: EXPLICIT on RHS → Cb1 * Stilda * nuTilda
    #   Destruction: IMPLICIT via Sp → fvm::Sp(Cw1*fw*nuTilda/d², nuTilda)
    production = cfg.Cb1 * Stilda * nuTilda  # explicit (current nuTilda values)
    Sn_coeff = cfg.Cw1 * fw * nuTilda / d_safe**2  # implicit destruction coefficient

    # 6. Diffusion coefficient → interpolate to faces
    DnuTildaEff = (nu_value + nuTilda) / cfg.sigma
    interp = nn.SurfaceInterpolationScalar(
        rt.executor, rt.nf_mesh, nn.TokenList(["linear"])
    )
    DnuTildaEff_f = interp.interpolate(DnuTildaEff)

    # 7. Non-conservative diffusion (explicit source)
    gradNuTilde = nn.exp.grad_field(nuTilda)
    nonConsDiff = (cfg.Cb2 / cfg.sigma) * nn.inner(gradNuTilde, gradNuTilde)

    # 8. Assemble and solve — matches OpenFOAM equation layout:
    #   LHS: ddt + div - laplacian + Sp(Sn_coeff, nuTilda)
    #   RHS: production + nonConsDiff
    # exp.source(coeff, field) computes coeff*field, so use a ones field
    # to avoid multiplying by nuTilda again.
    ones = nn.ScalarVolumeField(rt.executor, "ones", rt.nf_mesh)
    nn.fill(ones.internal_vector(), 1.0)
    rhs_source = production + nonConsDiff
    nuTildaEqn = nfb.PDESolverScalar(
        nn.imp.ddt(nuTilda)
        + nn.imp.div(phi, nuTilda)
        - nn.imp.laplacian(DnuTildaEff_f, nuTilda)
        + nn.imp.source(Sn_coeff, nuTilda)
        - nn.exp.source(rhs_source, ones),
        nuTilda,
        rt,
    )
    nuTildaEqn.solve()

    # 9. Bound and update
    nn.bound(nuTilda, 0.0)
    nuTilda.correct_boundary_conditions()

    # Recompute fv1 with updated nuTilda
    chi_new = nuTilda / nu_value
    fv1_new = chi_new**3 / (chi_new**3 + cfg.Cv1**3)

    # Update nut = nuTilda * fv1
    nut.assign(nuTilda * fv1_new)
    nut.correct_boundary_conditions()


def compute_nuEff_surface(
    rt: Any,
    nut: Any,
    nu_value: float,
) -> Any:
    """Compute effective viscosity and interpolate to faces."""
    nuEff = nu_value + nut
    interp = nn.SurfaceInterpolationScalar(
        rt.executor, rt.nf_mesh, nn.TokenList(["linear"])
    )
    return interp.interpolate(nuEff)


def compute_div_dev_reff_correction(
    nut: Any,
    nu_value: float,
    U: Any,
) -> Any:
    """Compute the explicit correction: -div(nuEff * dev2(T(grad(U)))).

    This is the term missing from the simplified -laplacian(nuEff, U)
    formulation. Together they form the full divDevReff(U).
    """
    grad_U = nn.exp.grad_field(U)
    T_grad_U = nn.transpose(grad_U)
    dev2_T_grad_U = nn.dev2(T_grad_U)
    nuEff = nu_value + nut
    nuEff_dev2 = nn.scalar_tensor_mul(nuEff, dev2_T_grad_U)
    return nn.exp.div_tensor(nuEff_dev2)


# --- Turbulence correction operation ---


@sa_neon.operation(operation_number="5", depends_on=["continuity"])
def turbulence_correction(
    nuTilda: Any,
    nut: Any,
    U: Any,
    phi: Any,
    d: Any,
    nuEff_surface: Any,
    sa_config: Annotated[SpalartAllmarasConfig, "models"],
    neon_runtime: Annotated[Any, "models"],
    nu_laminar_value: Annotated[float, "models"],
) -> FieldUpdates:
    """Solve SA transport equation and update turbulent viscosity."""
    rt = neon_runtime
    cfg = sa_config
    nu_value = nu_laminar_value

    correct(cfg, rt, nuTilda, nut, U, phi, d, nu_value)

    # Recompute nuEff surface field for next time step's momentum
    nuEff_surf = compute_nuEff_surface(rt, nut, nu_value)

    # NOTE: divDevReff correction (-div(nuEff * dev2(T(grad(U))))) is NOT
    # computed here. The explicit source treatment is unstable. The simplified
    # -laplacian(nuEff, U) formulation is used in momentum instead.
    # TODO: Implement as part of the implicit laplacian for stability.

    return FieldUpdates(
        {
            "nuTilda": nuTilda,
            "nut": nut,
            "nuEff_surface": nuEff_surf,
        }
    )


# --- Operation collection (returns ops for execution graph) ---


@sa_neon.operation_collection
def collected_operations(self: Any) -> Operations:
    model_ops = Operations()
    wrapped = wrap_operation(turbulence_correction, self, sa_neon._dependency_resolver)
    model_ops.add(
        Operation(
            func=SequentialOp(wrapped),
            metadata=OperationMetadata(
                op_name="turbulence_correction",
                depends_on=["continuity"],
                shape="box",
                color="lightyellow",
            ),
        )
    )
    return model_ops
