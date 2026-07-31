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
  with :func:`neofoam_bindings.bound`, and recompute ``nut`` / ``nuEff``.

This matches ``Foam::RASModels::kEpsilon::correct`` term for term for the
incompressible case (``alpha = rho = 1``), *including* the dilatation
``SuSp(divU, …)`` terms: ``divU = div(phi)`` only vanishes at convergence, and
during SIMPLE/PIMPLE iterations the continuity defect is O(1) exactly where
production peaks, so dropping them measurably shifts ``k``. Coefficients default
to OpenFOAM's and are overridden per case by the ``RAS`` sub-dictionary's
``kEpsilonCoeffs`` entry (:func:`~neofoam.turbulence.config.model_coefficients`).

Each transport ``@operation`` mirrors the NeoN solver's PDE lifecycle exactly:
``rotate_old_times`` seeds the field's previous-time value for ``imp.ddt``; every
intermediate coefficient is bound to a local because the implicit operators hold
*references* (not ownership) to their operands, so an inline temporary would be
freed before ``solve()`` and read as garbage.
"""

from typing import Annotated, Any

import neon._neon as nn
import pybFoam as pyf

from neofoam import neofoam_bindings as nfb
from neofoam.framework.context import FieldUpdates
from neofoam.framework.initialization import InitStep
from neofoam.framework.initialization import field as init_field
from neofoam.framework.initialization import model as init_model

from ..config import TurbulencePropertiesConfig, model_coefficients
from ..momentumTransport import Model, register_momentum_transport

__all__ = ["kEpsilon"]

#: OpenFOAM kEpsilon coefficient defaults; a case's ``kEpsilonCoeffs`` overrides them.
DEFAULT_COEFFS = {"Cmu": 0.09, "C1": 1.44, "C2": 1.92, "C3": 0.0, "sigmak": 1.0, "sigmaEps": 1.3}
# Lower bounds for k and epsilon (OpenFOAM bound() floors), kept tiny/positive.
kMin = 1e-15
epsilonMin = 1e-15

kEpsilon = register_momentum_transport(Model("kEpsilon"), family="RAS")
kEpsilon.config(TurbulencePropertiesConfig)


@kEpsilon.build
def build(config: TurbulencePropertiesConfig) -> list[InitStep]:
    """Read + seed ``k`` / ``epsilon`` / ``nut`` / ``nuEff`` and the helper operators.

    ``nut`` is seeded from the initial ``k`` / ``epsilon`` (OpenFOAM ``correctNut``);
    the surface-interpolation and ``gradSchemes``-selected gradient operators are owned as models
    so the per-step ``correct`` operation reuses them. The resolved coefficients are
    owned as a model too, so the operations read the case's overrides.
    """
    coeffs = model_coefficients(config, "kEpsilon", DEFAULT_COEFFS)

    def read_k(ctx: dict[str, Any]) -> Any:
        # OpenFOAM's kEpsilon constructor bounds both transport fields as read
        # (kEpsilon.C: bound(k_, kMin_); bound(epsilon_, epsilonMin_)), before the
        # nut seeding below divides by them.
        k = nfb.read_scalar_volume_field(ctx["models.neon_runtime"], "k")
        nfb.bound(k, kMin)
        return k

    def read_epsilon(ctx: dict[str, Any]) -> Any:
        epsilon = nfb.read_scalar_volume_field(ctx["models.neon_runtime"], "epsilon")
        nfb.bound(epsilon, epsilonMin)
        return epsilon

    def create_surf(ctx: dict[str, Any]) -> Any:
        rt = ctx["models.neon_runtime"]
        return nn.SurfaceInterpolationScalar(rt.executor, rt.nf_mesh, nn.TokenList(["linear"]))

    def create_grad(ctx: dict[str, Any]) -> Any:
        # OpenFOAM's kEpsilon takes its production gradient from fvc::grad(U), i.e.
        # the case's ``gradSchemes`` ``grad(U)`` entry (a cellLimited scheme there
        # must limit here too); GradScheme falls back to Gauss linear when absent.
        return nfb.GradScheme(ctx["models.neon_runtime"], "U")

    def create_near_wall_dist(ctx: dict[str, Any]) -> Any:
        # nearWallDist: boundary faces hold the owner-cell wall distance — the input
        # the epsilon/nutk wall functions read via the BoundaryContext. pybFoam's
        # nearWallDist is purely geometric, so (unlike Foam::wallDist) it needs no
        # `wallDist { method … }` entry in fvSchemes — which kEpsilon cases do not ship.
        rt = ctx["models.neon_runtime"]
        return nfb.build_near_wall_dist(rt, pyf.nearWallDist(rt.mesh))

    def seed_nut(ctx: dict[str, Any]) -> Any:
        # nut = Cmu k^2 / epsilon — OpenFOAM ``correctNut`` in Python field maths.
        # Parenthesise sqr(k) to match OpenFOAM's Cmu*sqr(k)/epsilon evaluation order.
        nut = nfb.read_scalar_volume_field(ctx["models.neon_runtime"], "nut")
        k = ctx["fields.k"]
        epsilon = ctx["fields.epsilon"]
        nut.assign(coeffs["Cmu"] * (k * k) / epsilon)
        # nutkWallFunction sets nut's wall faces from (k, nu, nearWallDist).
        nfb.correct_scalar_bc_ctx(nut, k, ctx["models.nu_vol"], ctx["models.kEpsilon_nearWallDist"])
        return nut

    def create_nu_eff(ctx: dict[str, Any]) -> Any:
        return ctx["models.kEpsilon_surf"].interpolate(ctx["fields.nut"] + ctx["models.nu_vol"])

    return [
        init_field("k", read_k, depends_on=["models.neon_runtime"]),
        init_field("epsilon", read_epsilon, depends_on=["models.neon_runtime"]),
        init_model("kEpsilon_coeffs", lambda _ctx: coeffs),
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
# incompressible case (``alpha = rho = 1``), one ``@operation`` each
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
    kEpsilon_coeffs: Annotated[Any, "models"],
    U: Annotated[Any, "fields"],
    phi: Annotated[Any, "fields"],
    nut: Annotated[Any, "fields"],
    k: Annotated[Any, "fields"],
    epsilon: Annotated[Any, "fields"],
) -> FieldUpdates:
    """Turbulent production ``G = nut GbyNu``, ``GbyNu = dev(twoSymm(gradU)) && gradU``.

    Publishes four fields: ``GbyNu`` — production per unit eddy viscosity, which the
    epsilon equation consumes as OpenFOAM v2406 does (``C1 GbyNu Cmu k``, a form
    that never divides by ``k``) — ``Gk``, the eddy-viscosity production with
    the near-wall cells overwritten by the epsilonWallFunction log-law form
    ``(nut+nu)|dU/dn| Cmu^0.25 sqrt(k)/(kappa y)``, which the k equation consumes,
    and the two dilatation ``SuSp`` coefficients ``eps_dilatation`` /
    ``k_dilatation`` built from ``divU = div(phi)`` (kEpsilon.C computes ``divU``
    once at the top of ``correct``; it is nonzero until the pressure solve has
    converged). The ``Gk`` override is the model-side half of the wall function
    (the BC only sets the wall *face* value); without it the k equation lacks the
    source that balances the wall-function epsilon.
    """
    grad_u = kEpsilon_grad.grad_tensor(U)
    GbyNu = nfb.strain_production(grad_u)
    G = nut * GbyNu
    Gk = nfb.epsilon_wall_production(
        G, epsilon, U, k, nu_vol, nut, kEpsilon_nearWallDist, neon_runtime
    )
    # fvc::div of a flux is the plain surface integral — no scheme is involved,
    # but evaluate_explicit's reader wants tokens, so pass the Gauss default.
    divU = nfb.create_uniform_volume_field(neon_runtime, "divU", 0.0)
    flux_divergence = nn.exp.div(phi)
    nfb.evaluate_explicit(
        flux_divergence, nn.TokenList(["Gauss", "linear"]), divU.internal_vector()
    )
    eps_dilatation = (2.0 / 3.0 * kEpsilon_coeffs["C1"] - kEpsilon_coeffs["C3"]) * divU
    k_dilatation = (2.0 / 3.0) * divU
    return FieldUpdates(
        {"GbyNu": GbyNu, "Gk": Gk, "eps_dilatation": eps_dilatation, "k_dilatation": k_dilatation}
    )


@kEpsilon.operation(name="kEpsilonCorrectEpsilon")
def correct_epsilon(
    self: Any,
    neon_runtime: Annotated[Any, "models"],
    nu_vol: Annotated[Any, "models"],
    kEpsilon_surf: Annotated[Any, "models"],
    kEpsilon_nearWallDist: Annotated[Any, "models"],
    kEpsilon_coeffs: Annotated[Any, "models"],
    GbyNu: Annotated[Any, "fields"],
    eps_dilatation: Annotated[Any, "fields"],
    phi: Annotated[Any, "fields"],
    k: Annotated[Any, "fields"],
    epsilon: Annotated[Any, "fields"],
    nut: Annotated[Any, "fields"],
) -> FieldUpdates:
    """epsilon transport: ddt + div - laplacian == C1 GbyNu Cmu k - SuSp - Sp; bound."""
    # ``ddt`` reads epsilon's previous-time value; seed it (old := current) so the
    # first step's ddt is (epsilon - epsilon_0)/dt, matching OpenFOAM's oldTime.
    nn.rotate_old_times(epsilon)
    # Bind every intermediate to a local: the NeoN implicit operators hold
    # references (not ownership) to their operand fields, so a temporary consumed
    # inline would be GC'd before solve() and read freed memory.
    diffusivity = nut / kEpsilon_coeffs["sigmaEps"] + nu_vol
    d_eps = kEpsilon_surf.interpolate(diffusivity)
    # v2406 writes the production as C1 GbyNu Cmu k, not C1 G eps/k: algebraically
    # the same while nut == Cmu k^2/eps, but with no division by a bounded k.
    production = kEpsilon_coeffs["C1"] * GbyNu * kEpsilon_coeffs["Cmu"] * k
    sink_coeff = kEpsilon_coeffs["C2"] * (epsilon / k)
    eps_eqn = nfb.PDESolverScalar(
        nn.imp.ddt(epsilon)
        + nn.imp.div(phi, epsilon)
        - nn.imp.laplacian(d_eps, epsilon)
        - nn.exp.source(production)
        + nn.imp.susp(eps_dilatation, epsilon)
        + nn.imp.source(sink_coeff, epsilon),
        epsilon,
        neon_runtime,
    )
    eps_eqn.set_final_iter(False)
    eps_eqn.relax()  # OpenFOAM kEpsilon.C: epsEqn.ref().relax()
    # epsilonWallFunction pins the near-wall CELL epsilon to the log-law value
    # (OpenFOAM's matrix.setValues); apply it after assembly, before solve.
    nfb.pin_epsilon_wall_cells(eps_eqn, epsilon, k, kEpsilon_nearWallDist, neon_runtime)
    eps_eqn.solve()
    nfb.bound(epsilon, epsilonMin)
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
    kEpsilon_coeffs: Annotated[Any, "models"],
    Gk: Annotated[Any, "fields"],
    k_dilatation: Annotated[Any, "fields"],
    phi: Annotated[Any, "fields"],
    k: Annotated[Any, "fields"],
    epsilon: Annotated[Any, "fields"],
    nut: Annotated[Any, "fields"],
) -> FieldUpdates:
    """k transport: ddt + div - laplacian == Gk - SuSp - Sp(eps/k, k); bound (uses the new epsilon).

    Uses ``Gk`` — the production with the near-wall cells overwritten by the
    epsilonWallFunction G form — not the bulk production the epsilon equation uses.
    """
    nn.rotate_old_times(k)  # old := current k (still initial — k solved after epsilon)
    # Bind operands to locals (operator-operand lifetime — see correct_epsilon).
    eps_over_k = epsilon / k
    diffusivity = nut / kEpsilon_coeffs["sigmak"] + nu_vol
    d_k = kEpsilon_surf.interpolate(diffusivity)
    k_eqn = nfb.PDESolverScalar(
        nn.imp.ddt(k)
        + nn.imp.div(phi, k)
        - nn.imp.laplacian(d_k, k)
        - nn.exp.source(Gk)
        + nn.imp.susp(k_dilatation, k)
        + nn.imp.source(eps_over_k, k),
        k,
        neon_runtime,
    )
    k_eqn.set_final_iter(False)
    k_eqn.relax()  # OpenFOAM kEpsilon.C: kEqn.ref().relax()
    k_eqn.solve()
    nfb.bound(k, kMin)
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
    kEpsilon_coeffs: Annotated[Any, "models"],
    k: Annotated[Any, "fields"],
    epsilon: Annotated[Any, "fields"],
    nut: Annotated[Any, "fields"],
) -> FieldUpdates:
    """correctNut: ``nut = Cmu k^2 / epsilon``, then refresh the effective viscosity ``nuEff``."""
    nut.assign(kEpsilon_coeffs["Cmu"] * (k * k) / epsilon)
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
