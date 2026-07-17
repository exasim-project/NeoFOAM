# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Pure-Python standard ``kEpsilon`` RAS closure on NeoN — a :class:`ModelSpec`.

The NeoN mirror of :mod:`neofoam.turbulence.models.laminar`, but a real
two-equation closure: registered with the runtime-selectable
:class:`~neofoam.turbulence.momentumTransport.momentumTransportModel` family under the
``turbulenceProperties`` name ``kEpsilon``, it owns the transport fields ``k`` and
``epsilon`` and defines an eddy viscosity ``nut = Cmu k^2 / epsilon``.

The point of the exercise is that the closure is authored as **readable field
maths in Python** — no C++ turbulence model, no nanobind rebuild per closure:

* ``@build`` reads ``k`` / ``epsilon`` / ``nut`` from the case (carrying their
  on-disk BCs), seeds ``nut = Cmu k^2 / epsilon`` (OpenFOAM ``kEpsilon``'s
  construction-time ``correctNut``) and ``nuEff``, and creates the
  surface-interpolation / gradient helper operators the transport solve reuses.
* the ``correct`` ``@operation``\\ s advance the ``k`` and ``epsilon`` transport PDEs
  one step with the same NeoN DSL the pressure solve uses (``imp.ddt/div/laplacian``
  + ``imp/exp.source``), the production term ``G = nut (dev(twoSymm(gradU)) &&
  gradU)`` from :func:`neofoam_bindings.strain_production`, bound ``k`` / ``epsilon``
  with :func:`neon._neon.field_max`, and recompute ``nut`` / ``nuEff``.

This matches ``Foam::RASModels::kEpsilon::correct`` term for term for the
incompressible, divergence-free case (``alpha = rho = 1``, ``div U = 0`` so the
dilatation ``SuSp(divU, …)`` terms vanish). Coefficients are the OpenFOAM defaults.

Each transport ``@operation`` mirrors the NeoN solver's PDE lifecycle exactly:
``rotate_old_times`` seeds the field's previous-time value for ``imp.ddt``; every
intermediate coefficient is bound to a local because the implicit operators hold
*references* (not ownership) to their operands, so an inline temporary would be
freed before ``solve()`` and read as garbage.
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

__all__ = ["kEpsilon"]

# OpenFOAM kEpsilon default coefficients.
Cmu = 0.09
C1 = 1.44
C2 = 1.92
sigmak = 1.0
sigmaEps = 1.3
# Lower bounds for k and epsilon (OpenFOAM bound() floors), kept tiny/positive.
kMin = 1e-15
epsilonMin = 1e-15

kEpsilon = Model("kEpsilon").register_with(momentumTransportModel)
kEpsilon.config(TurbulencePropertiesConfig)


@kEpsilon.build
def build(config: TurbulencePropertiesConfig) -> list[InitStep]:
    """Read + seed ``k`` / ``epsilon`` / ``nut`` / ``nuEff`` and the helper operators.

    ``nut`` is seeded from the initial ``k`` / ``epsilon`` (OpenFOAM ``correctNut``);
    the surface-interpolation and Gauss-Green gradient operators are owned as models
    so the per-step ``correct`` operation reuses them.
    """

    def read_k(ctx: dict[str, Any]) -> Any:
        return nfb.read_scalar_volume_field(ctx["models.neon_runtime"], "k")

    def read_epsilon(ctx: dict[str, Any]) -> Any:
        return nfb.read_scalar_volume_field(ctx["models.neon_runtime"], "epsilon")

    def create_surf(ctx: dict[str, Any]) -> Any:
        rt = ctx["models.neon_runtime"]
        return nn.SurfaceInterpolationScalar(
            rt.executor, rt.nf_mesh, nn.TokenList(["linear"])
        )

    def create_grad(ctx: dict[str, Any]) -> Any:
        return nfb.GaussGreenGrad(ctx["models.neon_runtime"])

    def create_near_wall_dist(ctx: dict[str, Any]) -> Any:
        # nearWallDist: boundary faces hold the owner-cell wall distance — the input
        # the epsilon/nutk wall functions read via the BoundaryContext.
        return nfb.build_near_wall_dist(ctx["models.neon_runtime"])

    def seed_nut(ctx: dict[str, Any]) -> Any:
        # nut = Cmu k^2 / epsilon — OpenFOAM ``correctNut`` in Python field maths.
        # Parenthesise sqr(k) to match OpenFOAM's Cmu*sqr(k)/epsilon evaluation order.
        nut = nfb.read_scalar_volume_field(ctx["models.neon_runtime"], "nut")
        k = ctx["fields.k"]
        epsilon = ctx["fields.epsilon"]
        nut.assign(Cmu * (k * k) / epsilon)
        # nutkWallFunction sets nut's wall faces from (k, nu, nearWallDist).
        nfb.correct_scalar_bc_ctx(
            nut, k, ctx["models.nu_vol"], ctx["models.kEpsilon_nearWallDist"]
        )
        return nut

    def create_nu_eff(ctx: dict[str, Any]) -> Any:
        return ctx["models.kEpsilon_surf"].interpolate(
            ctx["fields.nut"] + ctx["models.nu_vol"]
        )

    return [
        init_field("k", read_k, depends_on=["models.neon_runtime"]),
        init_field("epsilon", read_epsilon, depends_on=["models.neon_runtime"]),
        init_model("kEpsilon_surf", create_surf, depends_on=["models.neon_runtime"]),
        init_model("kEpsilon_grad", create_grad, depends_on=["models.neon_runtime"]),
        init_model(
            "kEpsilon_nearWallDist",
            create_near_wall_dist,
            depends_on=["models.neon_runtime"],
        ),
        init_field(
            "nut",
            seed_nut,
            depends_on=[
                "models.neon_runtime",
                "fields.k",
                "fields.epsilon",
                "models.nu_vol",
                "models.kEpsilon_nearWallDist",
            ],
        ),
        init_field(
            "nuEff",
            create_nu_eff,
            depends_on=["fields.nut", "models.kEpsilon_surf", "models.nu_vol"],
        ),
    ]


# ``correct`` is split into the stages of OpenFOAM ``kEpsilon::correct`` for the
# incompressible case (``alpha = rho = 1``, ``div(U) ~ 0``), one ``@operation`` each
# — the closure "exposes several" operations the wrapper steps in order (production
# → epsilon → k → nut). Each stage returns the fields it updated as a
# :class:`FieldUpdates`, so the Context (not in-place mutation) carries state
# between stages: ``production`` publishes ``G``; the transport stages read it.


@kEpsilon.operation(name="kEpsilonProduction")
def production(
    self: Any,
    kEpsilon_grad: Annotated[Any, "models"],
    neon_runtime: Annotated[Any, "models"],
    nu_vol: Annotated[Any, "models"],
    kEpsilon_nearWallDist: Annotated[Any, "models"],
    U: Annotated[Any, "fields"],
    nut: Annotated[Any, "fields"],
    k: Annotated[Any, "fields"],
    epsilon: Annotated[Any, "fields"],
) -> FieldUpdates:
    """Turbulent production ``G = nut (dev(twoSymm(gradU)) && gradU)``.

    Publishes two fields: the bulk ``G`` the epsilon equation consumes, and ``Gk``
    — the same production with the near-wall cells overwritten by the
    epsilonWallFunction log-law form ``(nut+nu)|dU/dn| Cmu^0.25 sqrt(k)/(kappa y)``
    — which the k equation consumes. The override is the model-side half of the
    wall function (the BC only sets the wall *face* value); without it the k
    equation lacks the source that balances the wall-function epsilon.
    """
    grad_u = kEpsilon_grad.grad_tensor(U)
    G = nut * nfb.strain_production(grad_u)
    Gk = nfb.epsilon_wall_production(
        G, epsilon, U, k, nu_vol, nut, kEpsilon_nearWallDist, neon_runtime
    )
    return FieldUpdates({"G": G, "Gk": Gk})


@kEpsilon.operation(name="kEpsilonCorrectEpsilon")
def correct_epsilon(
    self: Any,
    neon_runtime: Annotated[Any, "models"],
    nu_vol: Annotated[Any, "models"],
    kEpsilon_surf: Annotated[Any, "models"],
    kEpsilon_nearWallDist: Annotated[Any, "models"],
    G: Annotated[Any, "fields"],
    phi: Annotated[Any, "fields"],
    k: Annotated[Any, "fields"],
    epsilon: Annotated[Any, "fields"],
    nut: Annotated[Any, "fields"],
) -> FieldUpdates:
    """epsilon transport: ddt + div - laplacian == C1 G eps/k - Sp(C2 eps/k, eps); bound."""
    # ``ddt`` reads epsilon's previous-time value; seed it (old := current) so the
    # first step's ddt is (epsilon - epsilon_0)/dt, matching OpenFOAM's oldTime.
    nn.rotate_old_times(epsilon)
    # Bind every intermediate to a local: the NeoN implicit operators hold
    # references (not ownership) to their operand fields, so a temporary consumed
    # inline would be GC'd before solve() and read freed memory.
    eps_over_k = epsilon / k
    diffusivity = nut / sigmaEps + nu_vol
    d_eps = kEpsilon_surf.interpolate(diffusivity)
    production = C1 * G * eps_over_k
    sink_coeff = C2 * eps_over_k
    eps_eqn = nfb.PDESolverScalar(
        nn.imp.ddt(epsilon)
        + nn.imp.div(phi, epsilon)
        - nn.imp.laplacian(d_eps, epsilon)
        - nn.exp.source(production)
        + nn.imp.source(sink_coeff, epsilon),
        epsilon,
        neon_runtime,
    )
    eps_eqn.set_final_iter(False)
    # epsilonWallFunction pins the near-wall CELL epsilon to the log-law value
    # (OpenFOAM's matrix.setValues); apply it after assembly, before solve.
    nfb.pin_epsilon_wall_cells(eps_eqn, epsilon, k, kEpsilon_nearWallDist, neon_runtime)
    eps_eqn.solve()
    epsilon.assign(nn.field_max(epsilon, epsilonMin))
    # OpenFOAM's fvMatrix::solve corrects the solved field's BCs; NeoN's does not.
    # The epsilonWallFunction sets the wall face value from (k, nu, nearWallDist).
    nfb.correct_scalar_bc_ctx(epsilon, k, nu_vol, kEpsilon_nearWallDist)
    return FieldUpdates({"epsilon": epsilon})


@kEpsilon.operation(name="kEpsilonCorrectK")
def correct_k(
    self: Any,
    neon_runtime: Annotated[Any, "models"],
    nu_vol: Annotated[Any, "models"],
    kEpsilon_surf: Annotated[Any, "models"],
    kEpsilon_nearWallDist: Annotated[Any, "models"],
    Gk: Annotated[Any, "fields"],
    phi: Annotated[Any, "fields"],
    k: Annotated[Any, "fields"],
    epsilon: Annotated[Any, "fields"],
    nut: Annotated[Any, "fields"],
) -> FieldUpdates:
    """k transport: ddt + div - laplacian == Gk - Sp(eps/k, k); bound (uses the new epsilon).

    Uses ``Gk`` — the production with the near-wall cells overwritten by the
    epsilonWallFunction G form — not the bulk ``G`` the epsilon equation uses.
    """
    nn.rotate_old_times(k)  # old := current k (still initial — k solved after epsilon)
    # Bind operands to locals (operator-operand lifetime — see correct_epsilon).
    eps_over_k = epsilon / k
    diffusivity = nut / sigmak + nu_vol
    d_k = kEpsilon_surf.interpolate(diffusivity)
    k_eqn = nfb.PDESolverScalar(
        nn.imp.ddt(k)
        + nn.imp.div(phi, k)
        - nn.imp.laplacian(d_k, k)
        - nn.exp.source(Gk)
        + nn.imp.source(eps_over_k, k),
        k,
        neon_runtime,
    )
    k_eqn.set_final_iter(False)
    k_eqn.solve()
    k.assign(nn.field_max(k, kMin))
    # OpenFOAM's fvMatrix::solve corrects the solved field's BCs; NeoN's does not.
    # kqRWallFunction (zero-gradient) applies through the context correction too.
    nfb.correct_scalar_bc_ctx(k, k, nu_vol, kEpsilon_nearWallDist)
    return FieldUpdates({"k": k})


@kEpsilon.operation(name="kEpsilonCorrectNut")
def correct_nut(
    self: Any,
    nu_vol: Annotated[Any, "models"],
    kEpsilon_surf: Annotated[Any, "models"],
    kEpsilon_nearWallDist: Annotated[Any, "models"],
    k: Annotated[Any, "fields"],
    epsilon: Annotated[Any, "fields"],
    nut: Annotated[Any, "fields"],
) -> FieldUpdates:
    """correctNut: ``nut = Cmu k^2 / epsilon``, then refresh the effective viscosity ``nuEff``."""
    nut.assign(Cmu * (k * k) / epsilon)
    # nutkWallFunction sets nut's wall faces from (k, nu, nearWallDist).
    nfb.correct_scalar_bc_ctx(nut, k, nu_vol, kEpsilon_nearWallDist)
    return FieldUpdates({"nut": nut, "nuEff": kEpsilon_surf.interpolate(nut + nu_vol)})


@kEpsilon.operation(name="kEpsilonCorrect", fallback=True)
def correct(
    self: Any,
    turbulence: Annotated[Any, "models"],  # the wrapped pybFoam handle
) -> FieldUpdates:
    """Fallback path (incompressibleFluid): advance OpenFOAM's own kEpsilon.

    Scheduled only when a solver selects fallback=True; the native NeoN
    transport @operations above are skipped. The pybFoam handle owns nut
    and its stress. Resolved from the Context, never captured in the closure
    (see [[project_pybfoam_op_closure_cycle]]).
    """
    turbulence.correct()
    return FieldUpdates({})
