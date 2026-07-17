# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Pure-Python standard ``SpalartAllmaras`` RAS closure on NeoN — a :class:`ModelSpec`.

A one-equation eddy-viscosity closure: it owns the transport unknown ``nuTilda``
and defines ``nut = nuTilda * fv1(chi)``. Registered with the runtime-selectable
:class:`~neofoam.turbulence.momentumTransport.momentumTransportModel` family under the
``turbulenceProperties`` name ``SpalartAllmaras``.

It matches ``Foam::RASModels::SpalartAllmaras::correct`` term for term for the
incompressible case (``alpha = rho = 1``) with the default ``ft2 = false`` (so the
trip term drops out) and ``dTilda = y`` (wall distance — the RAS model uses no DES
length-scale limiter).

The closure is authored as **readable field maths that runs entirely on-device**
(CPU or GPU): the nonlinear Spalart-Allmaras functions (``chi``, ``fv1``, ``fv2``,
``Stilda``, ``r``, ``g``, ``fw``) are chained NeoN ``ScalarVolumeField`` operators —
``**`` (pow), ``field_max``/``field_min`` (elementwise against a field or a scalar),
and the reflected ``scalar - field`` / ``scalar / field`` — so no host round-trip
happens. Only the quantities needing mesh differencing / tensor algebra come from
small bindings — :func:`~neofoam_bindings.read_wall_distance` (``y``),
:func:`~neofoam_bindings.vorticity_magnitude` (``Omega = sqrt(2) mag(skew(gradU))``)
and :func:`~neofoam_bindings.mag_sqr_grad` (``magSqr(grad(nuTilda))``) — and the
implicit transport operators (``imp.ddt/div/laplacian/source``) drive the solve.

Coefficients are the OpenFOAM ``SpalartAllmaras`` defaults.
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

__all__ = ["spalartAllmaras"]

# OpenFOAM SpalartAllmaras default coefficients.
sigmaNut = 0.66666
kappa = 0.41
Cb1 = 0.1355
Cb2 = 0.622
Cw1 = Cb1 / kappa**2 + (1.0 + Cb2) / sigmaNut
Cw2 = 0.3
Cw3 = 2.0
Cv1 = 7.1
Cs = 0.3
# ft2_ defaults to false in OpenFOAM, so the trip term vanishes and is omitted.
nuTildaMin = 0.0  # OpenFOAM bound(nuTilda, 0)
SMALL = 1e-15  # Foam SMALL, the floor in r (Stilda >> SMALL in practice)


def _fv1(chi: Any) -> Any:
    """``fv1 = chi^3 / (chi^3 + Cv1^3)`` — chained field ops (runs on-device)."""
    chi3 = chi**3.0
    return chi3 / (chi3 + Cv1**3)


spalartAllmaras = Model("SpalartAllmaras").register_with(momentumTransportModel)
spalartAllmaras.config(TurbulencePropertiesConfig)


@spalartAllmaras.build
def build(config: TurbulencePropertiesConfig) -> list[InitStep]:
    """Read ``nuTilda``, seed ``nut = nuTilda fv1``, and own the helper operators.

    ``nut`` is seeded from the initial ``nuTilda`` (OpenFOAM's construction-time
    ``correctNut``). The wall-distance field, surface-interpolation and Gauss-Green
    gradient operators are owned as models so the per-step ``correct`` reuses them.
    """

    def read_nutilda(ctx: dict[str, Any]) -> Any:
        return nfb.read_scalar_volume_field(ctx["models.neon_runtime"], "nuTilda")

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
        # the nutUSpaldingWallFunction reads via the BoundaryContext.
        return nfb.build_near_wall_dist(ctx["models.neon_runtime"])

    def seed_nut(ctx: dict[str, Any]) -> Any:
        # nut = nuTilda * fv1(chi) — OpenFOAM correctNut in on-device field maths.
        rt = ctx["models.neon_runtime"]
        nut = nfb.read_scalar_volume_field(rt, "nut")
        nu = nfb.read_transport_viscosity(rt)
        nuTilda = ctx["fields.nuTilda"]
        nut.assign(nuTilda * _fv1(nuTilda / nu))
        # nutUSpaldingWallFunction sets nut's wall faces from (U, nu, nearWallDist).
        nfb.correct_scalar_bc_ctx_u(
            nut, ctx["fields.U"], ctx["models.nu_vol"], ctx["models.sa_nearWallDist"]
        )
        return nut

    def create_nu_eff(ctx: dict[str, Any]) -> Any:
        # Effective (surface) viscosity for the momentum laplacian: nuEff = nut + nu.
        return ctx["models.sa_surf"].interpolate(
            ctx["fields.nut"] + ctx["models.nu_vol"]
        )

    return [
        init_field("nuTilda", read_nutilda, depends_on=["models.neon_runtime"]),
        init_model("sa_surf", create_surf, depends_on=["models.neon_runtime"]),
        init_model("sa_grad", create_grad, depends_on=["models.neon_runtime"]),
        init_model("sa_wall_dist", read_wall_dist, depends_on=["models.neon_runtime"]),
        init_model(
            "sa_nearWallDist",
            create_near_wall_dist,
            depends_on=["models.neon_runtime"],
        ),
        init_field(
            "nut",
            seed_nut,
            depends_on=[
                "models.neon_runtime",
                "fields.nuTilda",
                "fields.U",
                "models.nu_vol",
                "models.sa_nearWallDist",
            ],
        ),
        init_field(
            "nuEff",
            create_nu_eff,
            depends_on=["fields.nut", "models.sa_surf", "models.nu_vol"],
        ),
    ]


@spalartAllmaras.operation(name="spalartAllmarasCorrectNuTilda")
def correct_nutilda(
    self: Any,
    neon_runtime: Annotated[Any, "models"],
    nu_vol: Annotated[Any, "models"],
    sa_surf: Annotated[Any, "models"],
    sa_grad: Annotated[Any, "models"],
    sa_wall_dist: Annotated[Any, "models"],
    U: Annotated[Any, "fields"],
    phi: Annotated[Any, "fields"],
    nuTilda: Annotated[Any, "fields"],
) -> FieldUpdates:
    """Advance the ``nuTilda`` transport PDE one step (OpenFOAM SA ``correct``, ft2=0).

    Every coefficient is a chained ``ScalarVolumeField`` expression, so the whole
    step runs on the executor the fields live on — no host transfer.
    """
    nn.rotate_old_times(nuTilda)
    rt = neon_runtime
    nu = nfb.read_transport_viscosity(rt)
    y = sa_wall_dist

    # Mesh-aware quantities (own kernels); everything else is chained field maths.
    grad_u = sa_grad.grad_tensor(U)
    omega = nfb.vorticity_magnitude(grad_u)  # sqrt(2) mag(skew(gradU))
    mag_sqr_grad = nfb.mag_sqr_grad(rt, nuTilda)  # magSqr(grad(nuTilda))

    chi = nuTilda / nu
    fv1 = _fv1(chi)
    fv2 = 1.0 - chi / (1.0 + chi * fv1)
    kd2 = (kappa * y) ** 2.0
    Stilda = nn.field_max(omega + fv2 * nuTilda / kd2, Cs * omega)
    r = nn.field_min(nuTilda / (nn.field_max(Stilda, SMALL) * kd2), 10.0)
    g = r + Cw2 * (r**6.0 - r)
    fw = g * ((1.0 + Cw3**6) / (g**6.0 + Cw3**6)) ** (1.0 / 6.0)

    # nuTildaEqn == Cb1*Stilda*nuTilda + (Cb2/sigma)*magSqr(grad(nuTilda))  (explicit)
    #             - Sp(Cw1*fw*nuTilda/y^2, nuTilda)  (implicit sink; coeff frozen)
    explicit = (Cb2 / sigmaNut) * mag_sqr_grad + Cb1 * Stilda * nuTilda
    sink_coeff = Cw1 * fw * nuTilda / (y**2.0)
    d_eff = sa_surf.interpolate((nuTilda + nu_vol) / sigmaNut)

    eqn = nfb.PDESolverScalar(
        nn.imp.ddt(nuTilda)
        + nn.imp.div(phi, nuTilda)
        - nn.imp.laplacian(d_eff, nuTilda)
        - nn.exp.source(explicit)
        + nn.imp.source(sink_coeff, nuTilda),
        nuTilda,
        rt,
    )
    eqn.set_final_iter(False)
    eqn.solve()
    nuTilda.assign(nn.field_max(nuTilda, nuTildaMin))
    # OpenFOAM's fvMatrix::solve corrects the solved field's BCs; NeoN's does not.
    nuTilda.correct_boundary_conditions()
    return FieldUpdates({"nuTilda": nuTilda})


@spalartAllmaras.operation(name="spalartAllmarasCorrectNut")
def correct_nut(
    self: Any,
    neon_runtime: Annotated[Any, "models"],
    nu_vol: Annotated[Any, "models"],
    sa_surf: Annotated[Any, "models"],
    sa_nearWallDist: Annotated[Any, "models"],
    U: Annotated[Any, "fields"],
    nuTilda: Annotated[Any, "fields"],
    nut: Annotated[Any, "fields"],
) -> FieldUpdates:
    """correctNut: ``nut = nuTilda fv1(chi)``, then refresh the effective viscosity ``nuEff``."""
    nu = nfb.read_transport_viscosity(neon_runtime)
    nut.assign(nuTilda * _fv1(nuTilda / nu))
    # nutUSpaldingWallFunction sets nut's wall faces from (U, nu, nearWallDist).
    nfb.correct_scalar_bc_ctx_u(nut, U, nu_vol, sa_nearWallDist)
    return FieldUpdates({"nut": nut, "nuEff": sa_surf.interpolate(nut + nu_vol)})


@spalartAllmaras.operation(name="spalartAllmarasCorrect", fallback=True)
def correct(
    self: Any,
    turbulence: Annotated[Any, "models"],  # the wrapped pybFoam handle
) -> FieldUpdates:
    """Fallback path (incompressibleFluid): advance OpenFOAM's own spalartAllmaras.

    Scheduled only when a solver selects fallback=True; the native NeoN
    transport @operations above are skipped. The pybFoam handle owns nut
    and its stress. Resolved from the Context, never captured in the closure
    (see [[project_pybfoam_op_closure_cycle]]).
    """
    turbulence.correct()
    return FieldUpdates({})
