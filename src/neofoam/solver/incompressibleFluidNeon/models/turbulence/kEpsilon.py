# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""k-epsilon turbulence model — NeoN implementation.

Solves k and epsilon transport equations using NeoN operators.
Mirrors OpenFOAM's kEpsilon::correct() from v2406.
Registered as a ModelSpec for auto-detection and solver integration.
"""

from dataclasses import dataclass
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


@dataclass
class KEpsilonConfig:
    """k-epsilon model constants."""

    Cmu: float = 0.09
    C1: float = 1.44
    C2: float = 1.92
    sigmaK: float = 1.0
    sigmaEps: float = 1.3


ke_neon = Model("k_epsilon_neon")


def detect_ke() -> Any | None:
    """Check turbulenceProperties for k-epsilon model. Returns model or None."""
    try:
        props = pyf.dictionary.read("constant/turbulenceProperties")
        if not props.found("RAS"):
            return None
        ras = props.subDict("RAS")
        if ras.found("RASModel") and ras.get_word("RASModel") == "kEpsilon":
            return ke_neon
        return None
    except Exception:
        return None


def _create_k(context: dict[str, Any]) -> Any:
    rt: Any = context["models.neon_runtime"]
    return nfb.read_scalar_volume_field(rt, "k")


def _create_epsilon(context: dict[str, Any]) -> Any:
    rt: Any = context["models.neon_runtime"]
    return nfb.read_scalar_volume_field(rt, "epsilon")


def _create_ke_config(context: dict[str, Any]) -> KEpsilonConfig:
    return KEpsilonConfig()


@ke_neon.build
def build(self: Any) -> list[Any]:
    return [
        field("k", _create_k, depends_on=["models.neon_runtime"]),
        field("epsilon", _create_epsilon, depends_on=["models.neon_runtime"]),
        model("ke_config", _create_ke_config, depends_on=[]),
    ]


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
    """Solve k-epsilon equations and update nut in-place.

    Mirrors OpenFOAM kEpsilon::correct():
    1. Compute production G = nut * (gradU && devTwoSymm(gradU))
    2. Solve epsilon equation (first)
    3. Solve k equation (second, with updated epsilon)
    4. Update nut = Cmu * k² / epsilon

    Args:
        cfg: Model constants
        rt: NeoN runtime
        k: Turbulent kinetic energy field (modified in-place)
        epsilon: Dissipation rate field (modified in-place)
        nut: Eddy viscosity field (modified in-place)
        U: Velocity field
        phi: Face flux field
        nu_value: Molecular viscosity (scalar)
    """
    SMALL = 1e-10

    # 0. Rotate old-time fields
    nn.rotate_old_times(k)
    nn.rotate_old_times(epsilon)

    # 1. Production term: G = nut * (gradU && devTwoSymm(gradU))
    gradU = nn.exp.grad_field(U)
    GbyNu = nn.doubleInner(gradU, nn.devTwoSymm(gradU))
    G = nut * GbyNu

    # 1b. divU = fvc::div(phi) for compressibility/dilatation terms
    divU = nn.exp.div_flux(phi)

    # 2. Correct boundary conditions before interpolation
    k.correct_boundary_conditions()
    epsilon.correct_boundary_conditions()
    nut.correct_boundary_conditions()

    # 3. Effective diffusivities → interpolate nut to faces first,
    #    then add nu. This avoids boundary data loss from volume arithmetic.
    interp = nn.SurfaceInterpolationScalar(
        rt.executor, rt.nf_mesh, nn.TokenList(["linear"])
    )
    nut_f = interp.interpolate(nut)
    DkEff_f = nut_f / cfg.sigmaK + nu_value
    DepsEff_f = nut_f / cfg.sigmaEps + nu_value

    # 3. Guard against division by zero
    k_safe = nn.field_max(k, SMALL)

    # 4. SuSp helper: split coeff into implicit (>0) and explicit (<0) parts
    #    SuSp(c, f) on LHS = imp.source(max(c,0), f) - exp.source(min(c,0)*f, ones)
    ones = nn.ScalarVolumeField(rt.executor, "ones", rt.nf_mesh)
    nn.fill(ones.internal_vector(), 1.0)

    # 5. Epsilon equation (solved FIRST)
    # OF: == C1*GbyNu*Cmu*k - SuSp((2/3)*C1*divU, eps) - Sp(C2*eps/k, eps)
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

    # 6. k equation (solved SECOND, with updated epsilon)
    # OF: == G - SuSp((2/3)*divU, k) - Sp(eps/k, k)
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

    # 7. Update nut = Cmu * k² / epsilon
    nut.assign(cfg.Cmu * k * k / epsilon)
    nut.correct_boundary_conditions()


def compute_nuEff_surface(rt: Any, nut: Any, nu_value: float) -> Any:
    """Compute effective viscosity and interpolate to faces."""
    interp = nn.SurfaceInterpolationScalar(
        rt.executor, rt.nf_mesh, nn.TokenList(["linear"])
    )
    nut_f = interp.interpolate(nut)
    return nut_f / 1.0 + nu_value  # nut + nu on faces


@ke_neon.operation(operation_number="5", depends_on=["continuity"])
def turbulence_correction(
    k: Any,
    epsilon: Any,
    nut: Any,
    U: Any,
    phi: Any,
    nuEff_surface: Any,
    ke_config: Annotated[KEpsilonConfig, "models"],
    neon_runtime: Annotated[Any, "models"],
    nu_laminar_value: Annotated[float, "models"],
) -> FieldUpdates:
    """Solve k-epsilon transport equations and update nuEff."""
    cfg = ke_config
    rt = neon_runtime
    nu_value = nu_laminar_value

    correct(cfg, rt, k, epsilon, nut, U, phi, nu_value)

    # Recompute nuEff surface field for momentum equation
    interp = nn.SurfaceInterpolationScalar(
        rt.executor, rt.nf_mesh, nn.TokenList(["linear"])
    )
    nut_f = interp.interpolate(nut)
    nuEff_surf = nut_f + nu_value

    return FieldUpdates(
        {
            "k": k,
            "epsilon": epsilon,
            "nut": nut,
            "nuEff_surface": nuEff_surf,
        }
    )


@ke_neon.operation_collection
def collected_operations(self: Any) -> Operations:
    model_ops = Operations()
    wrapped = wrap_operation(turbulence_correction, self, ke_neon._dependency_resolver)
    model_ops.add(
        Operation(
            func=SequentialOp(wrapped),
            metadata=OperationMetadata(
                op_name="turbulence_correction",
                depends_on=["continuity"],
                shape="box",
                color="lightgreen",
            ),
        )
    )
    return model_ops
