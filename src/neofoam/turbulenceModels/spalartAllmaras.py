# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""NeoN Spalart-Allmaras turbulence model for incompressibleFluidNeon solver."""

from typing import Any, Literal

import neon._neon as nn
import pybFoam as pyf
from pydantic import BaseModel

from neofoam import neofoam_bindings as nfb
from neofoam.framework.context import Context, FieldUpdates
from neofoam.framework.initialization import field, model
from neofoam.solver.incompressibleFluid.models.spalartAllmaras import (
    SpalartAllmarasConfig,
)

from .base import NeonTurbulenceModel


# --- Field creation functions ---


def _create_nuTilda(context: dict[str, Any]) -> Any:
    rt: Any = context["models.neon_runtime"]
    return nfb.read_scalar_volume_field(rt, "nuTilda")


def _create_nut(context: dict[str, Any]) -> Any:
    rt: Any = context["models.neon_runtime"]
    return nfb.read_scalar_volume_field(rt, "nut")


def _create_d(context: dict[str, Any]) -> Any:
    rt: Any = context["models.neon_runtime"]
    return nfb.compute_wall_distance(rt)


def _create_sa_config(context: dict[str, Any]) -> SpalartAllmarasConfig:
    return SpalartAllmarasConfig()


def _create_div_dev_reff_correction(context: dict[str, Any]) -> Any:
    rt: Any = context["models.neon_runtime"]
    corr = nn.VectorVolumeField(rt.executor, "div_dev_reff_correction", rt.nf_mesh)
    nn.fill(corr.internal_vector(), nn.Vec3(0, 0, 0))
    return corr


def _create_nuEff_surface(context: dict[str, Any]) -> Any:
    """Create effective viscosity surface field = nu + nut."""
    rt: Any = context["models.neon_runtime"]
    nu_value: float = context["models.nu_laminar_value"]
    return nfb.create_uniform_surface_field(rt, "nuEff", nu_value)


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
    nn.rotate_old_times(nuTilda)

    SMALL = 1e-10
    d_safe = nn.field_max(d, SMALL)

    chi = nuTilda / nu_value
    fv1 = chi**3 / (chi**3 + cfg.Cv1**3)
    fv2 = 1.0 - chi / (1.0 + chi * fv1)

    grad_U = nn.exp.grad_field(U)
    Omega = nn.field_max(2.0**0.5 * nn.mag(nn.skew(grad_U)), 0.0)

    Stilda = nn.field_max(
        Omega + fv2 * nuTilda / (cfg.kappa**2 * d_safe**2),
        cfg.Cs * Omega,
    )

    Stilda_safe = nn.field_max(Stilda, SMALL)
    r = nn.field_min(nuTilda / (Stilda_safe * cfg.kappa**2 * d_safe**2), 10.0)
    g = r + cfg.Cw2 * (r**6 - r)
    g6 = g**6
    fw_frac = (1.0 + cfg.Cw3**6) * (g6 + cfg.Cw3**6) ** (-1.0)
    fw = g * nn.field_pow(fw_frac, 1.0 / 6.0)

    production = cfg.Cb1 * Stilda * nuTilda
    Sn_coeff = cfg.Cw1 * fw * nuTilda / d_safe**2

    DnuTildaEff = (nu_value + nuTilda) / cfg.sigma
    interp = nn.SurfaceInterpolationScalar(
        rt.executor, rt.nf_mesh, nn.TokenList(["linear"])
    )
    DnuTildaEff_f = interp.interpolate(DnuTildaEff)

    gradNuTilde = nn.exp.grad_field(nuTilda)
    nonConsDiff = (cfg.Cb2 / cfg.sigma) * nn.inner(gradNuTilde, gradNuTilde)

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

    nn.bound(nuTilda, 0.0)
    nuTilda.correct_boundary_conditions()

    chi_new = nuTilda / nu_value
    fv1_new = chi_new**3 / (chi_new**3 + cfg.Cv1**3)

    nut.assign(nuTilda * fv1_new)
    nut.correct_boundary_conditions()


def compute_nuEff_surface(rt: Any, nut: Any, nu_value: float) -> Any:
    """Compute effective viscosity and interpolate to faces."""
    nuEff = nu_value + nut
    interp = nn.SurfaceInterpolationScalar(
        rt.executor, rt.nf_mesh, nn.TokenList(["linear"])
    )
    return interp.interpolate(nuEff)


def compute_div_dev_reff_correction(nut: Any, nu_value: float, U: Any) -> Any:
    """Compute the explicit correction: -div(nuEff * dev2(T(grad(U))))."""
    grad_U = nn.exp.grad_field(U)
    T_grad_U = nn.transpose(grad_U)
    dev2_T_grad_U = nn.dev2(T_grad_U)
    nuEff = nu_value + nut
    neg_nuEff = -1.0 * nuEff
    neg_nuEff_dev2 = nn.scalar_tensor_mul(neg_nuEff, dev2_T_grad_U)
    return nn.exp.div_tensor(neg_nuEff_dev2)


# --- PluginSystem-registered model ---


@NeonTurbulenceModel.register
class NeonSpalartAllmaras(BaseModel):
    """Spalart-Allmaras turbulence model using NeoN operators."""

    turbulence_type: Literal["spalart_allmaras"] = "spalart_allmaras"
    model_config = {"arbitrary_types_allowed": True}

    @staticmethod
    def detect_model() -> bool:
        """Check turbulenceProperties for SA model."""
        try:
            props = pyf.dictionary.read("constant/turbulenceProperties")
            if not props.found("RAS"):
                return False
            ras = props.subDict("RAS")
            return bool(
                ras.found("RASModel") and ras.get_word("RASModel") == "SpalartAllmaras"
            )
        except Exception:
            return False

    def build_steps(self) -> list[Any]:
        return [
            field("nuTilda", _create_nuTilda, depends_on=["models.neon_runtime"]),
            field("nut", _create_nut, depends_on=["models.neon_runtime"]),
            field("d", _create_d, depends_on=["models.neon_runtime"]),
            field(
                "div_dev_reff_correction",
                _create_div_dev_reff_correction,
                depends_on=["models.neon_runtime"],
            ),
            field(
                "nuEff_surface",
                _create_nuEff_surface,
                depends_on=["models.neon_runtime", "models.nu_laminar_value"],
            ),
            model("sa_config", _create_sa_config, depends_on=[]),
        ]

    def correct(self, ctx: Context) -> FieldUpdates:
        """Solve SA transport and update nuEff_surface."""
        nuTilda = ctx.fields["nuTilda"]
        nut = ctx.fields["nut"]
        U = ctx.fields["U"]
        phi = ctx.fields["phi"]
        d = ctx.fields["d"]
        rt = ctx.models["neon_runtime"]
        cfg = ctx.models["sa_config"]
        nu_value = ctx.models["nu_laminar_value"]

        correct(cfg, rt, nuTilda, nut, U, phi, d, nu_value)
        nuEff_surf = compute_nuEff_surface(rt, nut, nu_value)

        return FieldUpdates(
            {
                "nuTilda": nuTilda,
                "nut": nut,
                "nuEff_surface": nuEff_surf,
            }
        )
