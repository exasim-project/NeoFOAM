# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Pure-Python standard ``kOmegaSST`` RAS closure on NeoN — a :class:`ModelSpec`.

Menter's two-equation ``k``/``omega`` SST closure, authored as **on-device field
maths** (CPU or GPU). Registered with the runtime-selectable
:class:`~neofoam.turbulence.momentumTransport.momentumTransportModel` family under the
``turbulenceProperties`` name ``kOmegaSST``.

It matches ``Foam::RASModels::kOmegaSSTBase::correct`` term for term for the
incompressible, divergence-free case (``alpha = rho = 1``, ``div U = 0`` so the
dilatation ``SuSp(divU, ...)`` terms vanish; ``omegaInf = kInf = 0``, no SAS/F3).
The blending functions ``F1``/``F2`` and all coefficients are chained NeoN
``ScalarVolumeField`` operators — ``pow``/``tanh``/``sqrt``, elementwise
``field_max``/``field_min`` — so nothing round-trips to the host. The one genuinely
implicit new primitive is :func:`neon._neon.imp.susp` (OpenFOAM ``fvm::SuSp``), which
carries the **cross-diffusion** term ``SuSp((F1-1)·CDkOmega/omega, omega)`` — nonzero
even for divergence-free flow, so unlike ``kEpsilon`` it cannot be dropped.

Mesh-aware quantities come from the small kernels
:func:`~neofoam_bindings.read_wall_distance` (``y``),
:func:`~neofoam_bindings.strain_production` (``GbyNu0 = gradU && devTwoSymm(gradU)``),
:func:`~neofoam_bindings.strain_magnitude_sqr` (``S2 = 2 magSqr(symm gradU)``) and
:func:`~neofoam_bindings.grad_dot_grad` (``grad(k)·grad(omega)`` for ``CDkOmega``).

The ``correct`` is split into the OpenFOAM stages, one ``@operation`` each: a
**blend** stage computes ``F1`` and the (frozen) production / source coefficients
and publishes them; the **omega** and **k** stages solve their transport PDEs
(omega first, so k reads the updated omega); the **nut** stage recomputes
``nut = a1 k / max(a1 omega, b1 F23 sqrt(S2))``.

Coefficients are the OpenFOAM ``kOmegaSST`` defaults.
"""

from typing import Annotated, Any

import neon._neon as nn
from neofoam import neofoam_bindings as nfb
from neofoam.framework.context import FieldUpdates
from neofoam.framework.initialization import InitStep
from neofoam.framework.initialization import field as init_field
from neofoam.framework.initialization import model as init_model

from ..config import TurbulencePropertiesConfig
from ..momentumTransport import Model, momentumTransportModel

__all__ = ["kOmegaSST"]

# OpenFOAM kOmegaSST default coefficients.
alphaK1 = 0.85
alphaK2 = 1.0
alphaOmega1 = 0.5
alphaOmega2 = 0.856
gamma1 = 5.0 / 9.0
gamma2 = 0.44
beta1 = 0.075
beta2 = 0.0828
betaStar = 0.09
a1 = 0.31
b1 = 1.0
c1 = 10.0
kMin = 1e-15
omegaMin = 1e-15


def _blend(f1: Any, psi1: float, psi2: float) -> Any:
    """OpenFOAM ``blend``: ``F1*(psi1 - psi2) + psi2`` (exact evaluation order)."""
    return f1 * (psi1 - psi2) + psi2


def _f2(k: Any, omega: Any, y: Any, nu: float) -> Any:
    """SST ``F2 = tanh(arg2^2)`` (= ``F23`` with the default ``F3 = false``)."""
    arg2 = nn.field_min(
        nn.field_max(
            (2.0 / betaStar) * nn.sqrt(k) / (omega * y),
            500.0 * nu / ((y * y) * omega),
        ),
        100.0,
    )
    return nn.tanh(arg2 * arg2)


def _f1(k: Any, omega: Any, y: Any, nu: float, cd_komega: Any) -> Any:
    """SST ``F1 = tanh(arg1^4)`` with the cross-diffusion-limited inner argument."""
    cd_plus = nn.field_max(cd_komega, 1.0e-10)
    arg1 = nn.field_min(
        nn.field_min(
            nn.field_max(
                (1.0 / betaStar) * nn.sqrt(k) / (omega * y),
                500.0 * nu / ((y * y) * omega),
            ),
            (4.0 * alphaOmega2) * k / (cd_plus * (y * y)),
        ),
        10.0,
    )
    return nn.tanh(arg1**4.0)


def _correct_nut(k: Any, omega: Any, f23: Any, s2: Any) -> Any:
    """correctNut: ``nut = a1 k / max(a1 omega, b1 F23 sqrt(S2))``."""
    return a1 * k / nn.field_max(a1 * omega, (b1 * f23) * nn.sqrt(s2))


kOmegaSST = Model("kOmegaSST").register_with(momentumTransportModel)
kOmegaSST.config(TurbulencePropertiesConfig)


@kOmegaSST.build
def build(config: TurbulencePropertiesConfig) -> list[InitStep]:
    """Read ``k`` / ``omega``, seed ``nut`` (correctNut), own the helper operators."""

    def read_k(ctx: dict[str, Any]) -> Any:
        return nfb.read_scalar_volume_field(ctx["models.neon_runtime"], "k")

    def read_omega(ctx: dict[str, Any]) -> Any:
        return nfb.read_scalar_volume_field(ctx["models.neon_runtime"], "omega")

    def create_surf(ctx: dict[str, Any]) -> Any:
        rt = ctx["models.neon_runtime"]
        return nn.SurfaceInterpolationScalar(
            rt.executor, rt.nf_mesh, nn.TokenList(["linear"])
        )

    def create_grad(ctx: dict[str, Any]) -> Any:
        return nfb.GaussGreenGrad(ctx["models.neon_runtime"])

    def read_wall_dist(ctx: dict[str, Any]) -> Any:
        return nfb.read_wall_distance(ctx["models.neon_runtime"])

    def create_near_wall_dist(ctx: dict[str, Any]) -> Any:
        # nearWallDist: boundary faces hold the owner-cell wall distance — the input
        # the omega/nutk wall functions read via the BoundaryContext.
        return nfb.build_near_wall_dist(ctx["models.neon_runtime"])

    def seed_nut(ctx: dict[str, Any]) -> Any:
        rt = ctx["models.neon_runtime"]
        nut = nfb.read_scalar_volume_field(rt, "nut")
        nu = nfb.read_transport_viscosity(rt)
        k = ctx["fields.k"]
        omega = ctx["fields.omega"]
        y = ctx["models.komega_wall_dist"]
        grad_u = ctx["models.komega_grad"].grad_tensor(ctx["fields.U"])
        s2 = nfb.strain_magnitude_sqr(grad_u)
        nut.assign(_correct_nut(k, omega, _f2(k, omega, y, nu), s2))
        # nutkWallFunction sets nut's wall faces from (k, nu, nearWallDist).
        nfb.correct_scalar_bc_ctx(
            nut, k, ctx["models.nu_vol"], ctx["models.komega_nearWallDist"]
        )
        return nut

    def create_nu_eff(ctx: dict[str, Any]) -> Any:
        # Effective (surface) viscosity for the momentum laplacian: nuEff = nut + nu.
        return ctx["models.komega_surf"].interpolate(
            ctx["fields.nut"] + ctx["models.nu_vol"]
        )

    return [
        init_field("k", read_k, depends_on=["models.neon_runtime"]),
        init_field("omega", read_omega, depends_on=["models.neon_runtime"]),
        init_model("komega_surf", create_surf, depends_on=["models.neon_runtime"]),
        init_model("komega_grad", create_grad, depends_on=["models.neon_runtime"]),
        init_model(
            "komega_wall_dist", read_wall_dist, depends_on=["models.neon_runtime"]
        ),
        init_model(
            "komega_nearWallDist",
            create_near_wall_dist,
            depends_on=["models.neon_runtime"],
        ),
        init_field(
            "nut",
            seed_nut,
            depends_on=[
                "models.neon_runtime",
                "fields.k",
                "fields.omega",
                "fields.U",
                "models.komega_grad",
                "models.komega_wall_dist",
                "models.komega_nearWallDist",
                "models.nu_vol",
            ],
        ),
        init_field(
            "nuEff",
            create_nu_eff,
            depends_on=["fields.nut", "models.komega_surf", "models.nu_vol"],
        ),
    ]


@kOmegaSST.operation(name="kOmegaSSTBlend")
def blend(
    self: Any,
    neon_runtime: Annotated[Any, "models"],
    nu_vol: Annotated[Any, "models"],
    komega_grad: Annotated[Any, "models"],
    komega_wall_dist: Annotated[Any, "models"],
    komega_nearWallDist: Annotated[Any, "models"],
    U: Annotated[Any, "fields"],
    k: Annotated[Any, "fields"],
    omega: Annotated[Any, "fields"],
    nut: Annotated[Any, "fields"],
) -> FieldUpdates:
    """Compute ``F1`` and the frozen production / Sp / SuSp coefficients, and publish them."""
    nu = nfb.read_transport_viscosity(neon_runtime)
    # omegaWallFunction::updateCoeffs writes the near-wall omega CELL from the current
    # k *before* CDkOmega/F1 (kOmegaSSTBase.C:541, "omegaWallFunctions change the cell
    # value!"). Refresh here so the blending coefficients read current-k near-wall
    # omega, removing the one-iteration lag (D2). A no-op without an omegaWallFunction
    # patch (e.g. the no-WF turbulentBox case).
    nfb.refresh_omega_wall_cells(omega, k, nu_vol, komega_nearWallDist, neon_runtime)
    y = komega_wall_dist
    grad_u = komega_grad.grad_tensor(U)
    s2 = nfb.strain_magnitude_sqr(grad_u)
    gbynu0 = nfb.strain_production(grad_u)  # gradU && devTwoSymm(gradU)

    cd_komega = (2.0 * alphaOmega2) * nfb.grad_dot_grad(neon_runtime, k, omega) / omega
    f1 = _f1(k, omega, y, nu, cd_komega)
    f23 = _f2(k, omega, y, nu)
    gamma = _blend(f1, gamma1, gamma2)
    beta = _blend(f1, beta1, beta2)

    # Production: omega uses the strain-limited GbyNu0; k uses the raw G = nut*GbyNu0
    # with the near-wall cells overridden by the log-law form. The k-equation cap
    # ``min(G, c1*betaStar*k*omega)`` is applied later in correct_k, with the
    # *post-solve* (pinned) omega — mirroring kOmegaSSTBase::Pk(G), which is
    # evaluated after the omega solve so the wall cap binds against the pinned
    # omega. (kEpsilon has no such cap, so its wall production is never limited.)
    gbynu0_lim = nn.field_min(
        gbynu0,
        (c1 / a1)
        * betaStar
        * omega
        * nn.field_max(a1 * omega, (b1 * f23) * nn.sqrt(s2)),
    )
    g = nut * gbynu0
    # omegaWallFunction overrides the near-wall G with the log-law form — identical
    # to epsilonWallFunction's (cmu=betaStar). Reuses the shared binding; scans
    # omega's boundary for the "omegaWallFunction" patch. A no-op when no
    # wall-function patch exists (e.g. the no-WF turbulentBox case).
    g_wall = nfb.epsilon_wall_production(
        g,
        omega,
        U,
        k,
        nu_vol,
        nut,
        komega_nearWallDist,
        neon_runtime,
        betaStar,
        0.41,
        "omegaWallFunction",
    )

    return FieldUpdates(
        {
            "omega": omega,  # near-wall cells refreshed in-place (updateCoeffs, D2)
            "komega_F1": f1,
            "komega_omega_prod": gamma * gbynu0_lim,
            "komega_omega_sp": beta * omega,  # Sp(beta*omega, omega), coeff frozen
            "komega_omega_susp": (f1 - 1.0) * cd_komega / omega,  # SuSp, coeff frozen
            "komega_G": g_wall,  # wall-overridden, uncapped G (Pk cap applied in k)
        }
    )


@kOmegaSST.operation(name="kOmegaSSTCorrectOmega")
def correct_omega(
    self: Any,
    neon_runtime: Annotated[Any, "models"],
    nu_vol: Annotated[Any, "models"],
    komega_surf: Annotated[Any, "models"],
    komega_nearWallDist: Annotated[Any, "models"],
    komega_F1: Annotated[Any, "fields"],
    komega_omega_prod: Annotated[Any, "fields"],
    komega_omega_sp: Annotated[Any, "fields"],
    komega_omega_susp: Annotated[Any, "fields"],
    phi: Annotated[Any, "fields"],
    k: Annotated[Any, "fields"],
    omega: Annotated[Any, "fields"],
    nut: Annotated[Any, "fields"],
) -> FieldUpdates:
    """omega transport: ddt + div - laplacian == gamma*GbyNu0 - Sp(beta*omega) - SuSp(cross)."""
    nn.rotate_old_times(omega)
    nu = nfb.read_transport_viscosity(neon_runtime)
    d_omega = komega_surf.interpolate(
        _blend(komega_F1, alphaOmega1, alphaOmega2) * nut + nu
    )
    eqn = nfb.PDESolverScalar(
        nn.imp.ddt(omega)
        + nn.imp.div(phi, omega)
        - nn.imp.laplacian(d_omega, omega)
        - nn.exp.source(komega_omega_prod)
        + nn.imp.source(komega_omega_sp, omega)
        + nn.imp.susp(komega_omega_susp, omega),
        omega,
        neon_runtime,
    )
    eqn.set_final_iter(False)
    # omegaWallFunction pins the near-wall CELL omega to the blended wall value
    # (OpenFOAM's matrix.setValues); apply it after assembly, before solve.
    nfb.pin_omega_wall_cells(eqn, omega, k, nu_vol, komega_nearWallDist, neon_runtime)
    eqn.solve()
    omega.assign(nn.field_max(omega, omegaMin))
    # OpenFOAM's fvMatrix::solve corrects the solved field's BCs; NeoN's does not.
    # The omegaWallFunction sets the wall face value from (k, nu, nearWallDist).
    nfb.correct_scalar_bc_ctx(omega, k, nu_vol, komega_nearWallDist)
    return FieldUpdates({"omega": omega})


@kOmegaSST.operation(name="kOmegaSSTCorrectK")
def correct_k(
    self: Any,
    neon_runtime: Annotated[Any, "models"],
    nu_vol: Annotated[Any, "models"],
    komega_surf: Annotated[Any, "models"],
    komega_nearWallDist: Annotated[Any, "models"],
    komega_F1: Annotated[Any, "fields"],
    komega_G: Annotated[Any, "fields"],
    phi: Annotated[Any, "fields"],
    k: Annotated[Any, "fields"],
    omega: Annotated[Any, "fields"],
    nut: Annotated[Any, "fields"],
) -> FieldUpdates:
    """k transport: ddt + div - laplacian == Pk - Sp(betaStar*omega, k) (uses the new omega)."""
    nn.rotate_old_times(k)
    nu = nfb.read_transport_viscosity(neon_runtime)
    d_k = komega_surf.interpolate(_blend(komega_F1, alphaK1, alphaK2) * nut + nu)
    eps_by_k = betaStar * omega  # epsilonByk, uses the updated omega
    # Pk = min(G, c1*betaStar*k*omega) — kOmegaSSTBase::Pk(G), evaluated here (after
    # the omega solve) so the near-wall log-law G is capped against the *pinned*
    # omega. omega is huge at the wall, so the cap normally only binds at high-shear
    # wall cells (e.g. the inlet/wall corner), which is exactly where the raw
    # log-law G overshoots.
    pk = nn.field_min(komega_G, (c1 * betaStar) * k * omega)
    eqn = nfb.PDESolverScalar(
        nn.imp.ddt(k)
        + nn.imp.div(phi, k)
        - nn.imp.laplacian(d_k, k)
        - nn.exp.source(pk)
        + nn.imp.source(eps_by_k, k),
        k,
        neon_runtime,
    )
    eqn.set_final_iter(False)
    eqn.solve()
    k.assign(nn.field_max(k, kMin))
    # OpenFOAM's fvMatrix::solve corrects the solved field's BCs; NeoN's does not.
    # kqRWallFunction (zero-gradient) applies through the context correction too.
    nfb.correct_scalar_bc_ctx(k, k, nu_vol, komega_nearWallDist)
    return FieldUpdates({"k": k})


@kOmegaSST.operation(name="kOmegaSSTCorrectNut")
def correct_nut(
    self: Any,
    neon_runtime: Annotated[Any, "models"],
    nu_vol: Annotated[Any, "models"],
    komega_surf: Annotated[Any, "models"],
    komega_grad: Annotated[Any, "models"],
    komega_wall_dist: Annotated[Any, "models"],
    komega_nearWallDist: Annotated[Any, "models"],
    U: Annotated[Any, "fields"],
    k: Annotated[Any, "fields"],
    omega: Annotated[Any, "fields"],
    nut: Annotated[Any, "fields"],
) -> FieldUpdates:
    """correctNut from the updated ``k`` / ``omega``, then refresh ``nuEff``."""
    nu = nfb.read_transport_viscosity(neon_runtime)
    y = komega_wall_dist
    s2 = nfb.strain_magnitude_sqr(komega_grad.grad_tensor(U))
    nut.assign(_correct_nut(k, omega, _f2(k, omega, y, nu), s2))
    # nutkWallFunction sets nut's wall faces from (k, nu, nearWallDist).
    nfb.correct_scalar_bc_ctx(nut, k, nu_vol, komega_nearWallDist)
    return FieldUpdates({"nut": nut, "nuEff": komega_surf.interpolate(nut + nu_vol)})


@kOmegaSST.operation(name="kOmegaSSTCorrect", fallback=True)
def correct(
    self: Any,
    turbulence: Annotated[Any, "models"],  # the wrapped pybFoam handle
) -> FieldUpdates:
    """Fallback path (incompressibleFluid): advance OpenFOAM's own kOmegaSST.

    Scheduled only when a solver selects fallback=True; the native NeoN
    transport @operations above are skipped. The pybFoam handle owns nut
    and its stress. Resolved from the Context, never captured in the closure
    (see [[project_pybfoam_op_closure_cycle]]).
    """
    turbulence.correct()
    return FieldUpdates({})
