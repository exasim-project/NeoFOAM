# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

# NB: no ``from __future__ import annotations`` — the dependency_resolver matches
# ``Annotated[..., "models"]`` / ``param.annotation is Context`` live.

"""Chorin fractional-step projection built from ``blockamr`` building blocks.

One member of the projection family. The model **owns its configs** — the
physical viscosity, the discretisation schemes and the two per-field
linear-solver blocks are registered on the spec and loaded by ``instantiate``,
so ``@build`` receives them directly instead of fishing them out of the solver's
init work dict.

``@build`` creates ``U`` / ``p`` / ``phi`` explicitly on the ``blockamr`` mesh
and registers them as Context fields, alongside the values the equations need
(``nu`` / ``schemes`` / ``sol_U`` / ``sol_p`` / ``ibm``). There is no solver
state object: the framework Context holds the fields and the framework
``LoopState`` (``ctx.time``) holds the clock.

The fractional step is exposed as two operations mirroring PIMPLE's ``momentum``
/ ``continuity`` split, each writing its own equation at the point of solve:
``momentum`` assembles ``UEqn`` and does interpolate → MAC-project φ → momentum
predictor; ``continuity`` assembles ``pEqn`` and does pressure Poisson → velocity
correct → IBM apply, leaving ``U`` divergence-free. Fields and values are
resolved from the Context at run time (``ctx.models`` / ``ctx.fields``), never
captured in the operation closure — mirroring the op-closure-cycle rule.

NB: importing this module pulls ``blockamr`` (and therefore jax + the native
AMReX extension) eagerly — the imports are at the top so the reader sees what the
solver is built from. The CLI imports solver packages lazily per command, so this
only costs the blockAMR solver itself.
"""

from typing import Annotated, Any, Callable

from blockamr.bc import pressure_domain_bc
from blockamr.dsl import Equation, exp, imp
from blockamr.field import CellField, FaceField
from blockamr.fillpatch import FillPatchCellConservative, FillPatchWithBC
from blockamr.ibm import IBM
from blockamr.operators.correct import correct
from blockamr.operators.interpolate import interpolate
from blockamr.operators.mac_project import mac_project
from blockamr.schemes.div_schemes import Upwind
from blockamr.schemes.laplacian_schemes import CentralDiffLaplacian
from blockamr.schemes.registry import lookup_scheme

from neofoam.framework.context import Context
from neofoam.framework.dependency_resolver import wrap_with_dependency_resolution
from neofoam.framework.initialization import field, lazy, model as init_model
from neofoam.framework.operations import Operation, Operations, SequentialOp
from neofoam.framework.types import OperationMetadata

from ...configs import (
    BlockAMRSolutionConfig,
    FvSchemesConfig,
    PSolutionConfig,
    USolutionConfig,
)
from ..incompressibleFluidBlockAMRModel import Model
from ..bc_mapping import build_vector_bc

chorinProjection = Model("ChorinProjection")

# The configs this model owns. ``ModelSpec.instantiate`` loads each through its
# ``@IOStrategy`` binding and hands them to ``@build`` as a namespace. The mesh
# dict stays with the solver (``create_fields`` builds the mesh from it) and is
# read from the init work dict as ``_mesh_cfg``.
chorinProjection.config(BlockAMRSolutionConfig)
chorinProjection.config(FvSchemesConfig)
chorinProjection.config(USolutionConfig)
chorinProjection.config(PSolutionConfig)


def _ngrow(schemes: dict[str, Any]) -> int:
    """Ghost width: the widest stencil across the operators. Not hardcoded."""
    div_sw = lookup_scheme(schemes, ["div(phi,U)"], "div", Upwind()).stencil_width
    return int(max(div_sw, CentralDiffLaplacian().stencil_width))


@chorinProjection.build
def build(config: Any) -> list[Any]:
    """Create U/p/phi on the mesh and register them + the equation inputs.

    ``config`` is the namespace of this model's own configs; the mesh and the
    mesh dict come from ``create_fields`` as ``_blockamr_mesh`` / ``_mesh_cfg``.
    """
    nu: float = config.block_amr_solution_config.nu
    # fvSchemes: discretisation scheme NAMES — the engine's registry resolves
    # them when the operations assemble UEqn / pEqn.
    schemes: dict[str, Any] = config.fv_schemes_config.resolve()
    # Per-field fvSolution.solvers[<field>] blocks. ``sol_U`` carries the
    # velocity field's ``ibm`` method (empty on non-cylinder cases); ``sol_p``
    # the pressure MLMG rtol/atol/maxIter/bottomSolver/verbosity.
    sol_U: dict[str, Any] = config.u_solution_config.resolve()
    sol_p: dict[str, Any] = config.p_solution_config.resolve()

    def make_u_bc(context: dict[str, Any]) -> Any:
        """The velocity BC — None when the domain is fully periodic."""
        mesh_cfg = context["_mesh_cfg"]
        if all(mesh_cfg.periodicity):
            return None
        return build_vector_bc(mesh_cfg.boundary or {})

    def make_p_bc(context: dict[str, Any]) -> Any:
        """Per-face pressure BC for the MAC + nodal Poisson solves.

        Derived from the velocity BC (outflow face -> Dirichlet p, inlet/wall ->
        Neumann p); None -> the periodic/all-Neumann default. Stashed on p / phi
        so the free-function solves (dsl.solve, mac_project) can read it.
        """
        u_bc = context["_u_bc"]
        if u_bc is None:
            return None
        return pressure_domain_bc(u_bc, context["_blockamr_mesh"].geom(0))

    def make_u(context: dict[str, Any]) -> Any:
        u_bc = context["_u_bc"]
        # Fully periodic: no domain BCs — use the conservative fill-patch.
        fill_patch = (
            FillPatchCellConservative() if u_bc is None else FillPatchWithBC(u_bc)
        )
        return CellField(
            context["_blockamr_mesh"],
            ncomp=3,
            ngrow=_ngrow(schemes),
            name="U",
            fill_patch=fill_patch,
        )

    def make_p(context: dict[str, Any]) -> Any:
        p = CellField(context["_blockamr_mesh"], ncomp=1, ngrow=0, name="p")
        p.pressure_bc = context["_p_bc"]
        return p

    def make_phi(context: dict[str, Any]) -> Any:
        phi = FaceField(
            context["_blockamr_mesh"], ncomp=1, ngrow=_ngrow(schemes), name="phi"
        )
        phi.pressure_bc = context["_p_bc"]
        return phi

    def make_ibm(context: dict[str, Any]) -> Any:
        """Precompute every distinct IBM method's data; return ``U``'s method.

        Immersed body: the geometry lives on ``mesh.body`` (set by the mesh
        factory); the method is chosen per field via ``solution["ibm"]``. The
        data is precomputed eagerly here, ready before the first solve.
        """
        methods: list[Any] = []
        for sol in (sol_U, sol_p):
            ibm_name = sol.get("ibm")
            if ibm_name is not None:
                method = IBM.lookup(ibm_name)
                if method not in methods:
                    methods.append(method)
        if methods:
            context["_blockamr_mesh"].build_ibm(methods)

        u_ibm = sol_U.get("ibm")
        return IBM.lookup(u_ibm) if u_ibm is not None else None

    return [
        lazy("_u_bc", make_u_bc, depends_on=["_mesh_cfg"]),
        lazy("_p_bc", make_p_bc, depends_on=["_u_bc", "_blockamr_mesh"]),
        field("U", make_u, depends_on=["_blockamr_mesh", "_u_bc"], write=True),
        field("p", make_p, depends_on=["_blockamr_mesh", "_p_bc"], write=True),
        field("phi", make_phi, depends_on=["_blockamr_mesh", "_p_bc"]),
        init_model("nu", lambda _ctx: nu),
        init_model("schemes", lambda _ctx: schemes),
        init_model("sol_U", lambda _ctx: sol_U),
        init_model("sol_p", lambda _ctx: sol_p),
        init_model(
            "ibm",
            make_ibm,
            depends_on=["_blockamr_mesh", "fields.U", "fields.p", "fields.phi"],
        ),
    ]


@chorinProjection.operation(operation_number="2.1")
def momentum(
    U: Annotated[Any, "fields"],
    phi: Annotated[Any, "fields"],
    nu: Annotated[Any, "models"],
    schemes: Annotated[Any, "models"],
    sol_U: Annotated[Any, "models"],
    sol_p: Annotated[Any, "models"],
    ctx: Context,
) -> None:
    """Momentum predictor — first half of the Chorin fractional step.

    Mirrors PIMPLE's ``momentum``: fill BCs, write the momentum equation,
    interpolate ``U`` to the faces, MAC-project the face flux ``phi``
    divergence-free, solve for the predicted (non-solenoidal) velocity, and
    refill BCs ready for the pressure solve. Mutates ``U`` / ``phi`` in place —
    the same objects registered in ``ctx.fields`` — so no ``FieldUpdates`` are
    needed.

    ``nu`` is passed as a numeric constant (not a lambda): the jax Laplacian
    collapses a provably-constant callable gamma to exactly ``coeff * nu``, and
    the cpp explicit backend rejects callables outright.
    """
    dt = ctx.time.delta_t
    # ``increment_time`` has already advanced the loop, so ``value`` is the END
    # of this step; the BCs and both solves are evaluated at its start.
    t = ctx.time.value - dt
    n_levels = U.mesh.n_levels()

    for lev in range(n_levels):
        U.fill_patch(lev, t)

    UEqn = Equation(
        exp.ddt(U) + exp.div(phi, U) - exp.laplacian(nu, U),
        schemes=schemes,
    )

    interpolate(U, phi)
    mac_project(phi, sol_p)

    UEqn.solve(dt=dt, t=t, solution=sol_U)

    for lev in range(n_levels):
        U.fill_patch(lev, t)


@chorinProjection.operation(operation_number="2.2", depends_on=["momentum"])
def continuity(
    U: Annotated[Any, "fields"],
    p: Annotated[Any, "fields"],
    schemes: Annotated[Any, "models"],
    sol_p: Annotated[Any, "models"],
    ibm: Annotated[Any, "models"],
    ctx: Context,
) -> None:
    """Pressure projection — second half of the Chorin fractional step.

    Mirrors PIMPLE's ``continuity``: write and solve the pressure Poisson
    equation, correct the predicted velocity by ``-dt·grad(p)`` so ``U`` is
    divergence-free, and apply the immersed-boundary forcing. Mutates ``U`` /
    ``p`` in place. ``pEqn`` is assembled here with the *current* ``dt``, so the
    implicit Laplacian's sigma tracks a changing time step without being patched
    after the fact. The clock is the framework loop's — nothing to advance here.
    """
    dt = ctx.time.delta_t
    t = ctx.time.value - dt

    pEqn = Equation(
        imp.laplacian(dt, p) == exp.div(U),
        schemes=schemes,
    )
    pEqn.solve(dt=dt, t=t, solution=sol_p)

    correct(U, -dt * exp.grad(p))

    # Carry from phase 04: solve() does NOT consume solution["ibm"] — direct
    # forcing is applied once per step AFTER correct() (order matches the
    # pre-refactor engine exactly, the Cd/Cl/St acceptance oracle).
    if ibm is not None:
        ibm.apply(U, dt, t, U.mesh.ibm_data(ibm))


def _alias_operation(
    op_func: Callable[..., Any],
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


@chorinProjection.operation_collection
def collected_operations(self: Any) -> Operations:
    """Expose the momentum + continuity ops for the solver's execution graph."""
    model_ops = Operations()
    wrapped_momentum = wrap_with_dependency_resolution(
        momentum, self, chorinProjection._dependency_resolver
    )
    wrapped_continuity = wrap_with_dependency_resolution(
        continuity, self, chorinProjection._dependency_resolver
    )
    model_ops.add(
        _alias_operation(wrapped_momentum, operation_name="momentum", depends_on=[])
    )
    model_ops.add(
        _alias_operation(
            wrapped_continuity, operation_name="continuity", depends_on=["momentum"]
        )
    )
    return model_ops
