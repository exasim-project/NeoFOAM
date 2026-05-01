# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""NeoN k-epsilon turbulence model."""

from dataclasses import dataclass
from typing import Any, Literal

import neon._neon as nn
import pybFoam as pyf
from pydantic import BaseModel

from neofoam import neofoam_bindings as nfb
from neofoam.framework.context import Context, FieldUpdates
from neofoam.framework.initialization import field, model

from .base import NeonTurbulenceModel


@dataclass
class KEpsilonConfig:
    """k-epsilon model constants."""

    Cmu: float = 0.09
    C1: float = 1.44
    C2: float = 1.92
    sigmaK: float = 1.0
    sigmaEps: float = 1.3


# --- Field creation functions ---


def _create_k(context: dict[str, Any]) -> Any:
    rt: Any = context["models.neon_runtime"]
    return nfb.read_scalar_volume_field(rt, "k")


def _create_epsilon(context: dict[str, Any]) -> Any:
    rt: Any = context["models.neon_runtime"]
    return nfb.read_scalar_volume_field(rt, "epsilon")


def _create_nut(context: dict[str, Any]) -> Any:
    rt: Any = context["models.neon_runtime"]
    return nfb.read_scalar_volume_field(rt, "nut")


def _create_ke_config(context: dict[str, Any]) -> KEpsilonConfig:
    return KEpsilonConfig()


def _create_nuEff_surface(context: dict[str, Any]) -> Any:
    """Create effective viscosity surface field = nu + nut."""
    rt: Any = context["models.neon_runtime"]
    nu_value: float = context["models.nu_laminar_value"]
    return nfb.create_uniform_surface_field(rt, "nuEff", nu_value)


# --- k-epsilon solver ---


def correct(
    cfg: KEpsilonConfig,
    rt: Any,
    k: Any,
    epsilon: Any,
    nut: Any,
    U: Any,
    phi: Any,
    nu_value: float,
) -> None:
    """Solve k-epsilon equations and update nut in-place."""
    SMALL = 1e-10

    nn.rotate_old_times(k)
    nn.rotate_old_times(epsilon)

    gradU = nn.exp.grad_field(U)
    GbyNu = nn.doubleInner(gradU, nn.devTwoSymm(gradU))
    G = nut * GbyNu

    divU = nn.exp.div_flux(phi)

    k.correct_boundary_conditions()
    epsilon.correct_boundary_conditions()
    nut.correct_boundary_conditions()

    interp = nn.SurfaceInterpolationScalar(
        rt.executor, rt.nf_mesh, nn.TokenList(["linear"])
    )
    DkEff = nut / cfg.sigmaK + nu_value
    DepsEff = nut / cfg.sigmaEps + nu_value
    DkEff_f = interp.interpolate(DkEff)
    DepsEff_f = interp.interpolate(DepsEff)

    k_safe = nn.field_max(k, SMALL)

    ones = nn.ScalarVolumeField(rt.executor, "ones", rt.nf_mesh)
    nn.fill(ones.internal_vector(), 1.0)

    # Epsilon equation
    eps_production = cfg.C1 * GbyNu * cfg.Cmu * k
    eps_destruction = cfg.C2 * epsilon / k_safe
    eps_susp_coeff = (2.0 / 3.0) * cfg.C1 * divU

    epsilonEqn = nfb.PDESolverScalar(
        nn.imp.ddt(epsilon)
        + nn.imp.div(phi, epsilon)
        - nn.imp.laplacian(DepsEff_f, epsilon)
        + nn.imp.source(eps_destruction, epsilon)
        + nn.imp.source(eps_susp_coeff, epsilon, susp=True)
        - nn.exp.source(eps_production, ones),
        epsilon,
        rt,
    )
    epsilonEqn.solve()
    nn.bound(epsilon, SMALL)
    epsilon.correct_boundary_conditions()

    # k equation
    k_destruction = epsilon / k_safe
    k_susp_coeff = (2.0 / 3.0) * divU

    kEqn = nfb.PDESolverScalar(
        nn.imp.ddt(k)
        + nn.imp.div(phi, k)
        - nn.imp.laplacian(DkEff_f, k)
        + nn.imp.source(k_destruction, k)
        + nn.imp.source(k_susp_coeff, k, susp=True)
        - nn.exp.source(G, ones),
        k,
        rt,
    )
    kEqn.solve()
    nn.bound(k, SMALL)
    k.correct_boundary_conditions()

    # Update nut = Cmu * k² / epsilon
    nut.assign(cfg.Cmu * k * k / epsilon)
    nut.correct_boundary_conditions()


# --- PluginSystem-registered model ---


@NeonTurbulenceModel.register
class NeonKEpsilon(BaseModel):
    """k-epsilon turbulence model using NeoN operators."""

    turbulence_type: Literal["k_epsilon"] = "k_epsilon"
    model_config = {"arbitrary_types_allowed": True}

    @staticmethod
    def detect_model() -> bool:
        """Check turbulenceProperties for k-epsilon model."""
        try:
            props = pyf.dictionary.read("constant/turbulenceProperties")
            if not props.found("RAS"):
                return False
            ras = props.subDict("RAS")
            return bool(
                ras.found("RASModel") and ras.get_word("RASModel") == "kEpsilon"
            )
        except Exception:
            return False

    def build_steps(self) -> list[Any]:
        return [
            field("k", _create_k, depends_on=["models.neon_runtime"]),
            field("epsilon", _create_epsilon, depends_on=["models.neon_runtime"]),
            field("nut", _create_nut, depends_on=["models.neon_runtime"]),
            field(
                "nuEff_surface",
                _create_nuEff_surface,
                depends_on=["models.neon_runtime", "models.nu_laminar_value"],
            ),
            model("ke_config", _create_ke_config, depends_on=[]),
        ]

    def correct(self, ctx: Context) -> FieldUpdates:
        """Solve k-epsilon transport and update nuEff_surface."""
        k = ctx.fields["k"]
        epsilon = ctx.fields["epsilon"]
        nut = ctx.fields["nut"]
        U = ctx.fields["U"]
        phi = ctx.fields["phi"]
        rt = ctx.models["neon_runtime"]
        cfg = ctx.models["ke_config"]
        nu_value = ctx.models["nu_laminar_value"]

        correct(cfg, rt, k, epsilon, nut, U, phi, nu_value)

        interp = nn.SurfaceInterpolationScalar(
            rt.executor, rt.nf_mesh, nn.TokenList(["linear"])
        )
        nut_f = interp.interpolate(nut)
        nuEff_surf = nut_f + nu_value

        return FieldUpdates({
            "k": k,
            "epsilon": epsilon,
            "nut": nut,
            "nuEff_surface": nuEff_surf,
        })
