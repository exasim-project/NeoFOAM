# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

# NB: no ``from __future__ import annotations`` — the dependency_resolver matches
# ``Annotated[..., "models"]`` / ``param.annotation is Context`` live.

"""Chorin fractional-step projection backed by ``neon.blockamr``.

One member of the projection family. ``@build`` constructs the fields
(``U`` / ``p`` / ``phi``) and the two ``Equation``s directly (via
``neon.blockamr.incompressible.build_incompressible``) from the mesh + configs
assembled in ``create_fields``, registers the fields into the Context, and
registers the projection **state** (fields + equations + solve settings) as
``models.projection_state``.

The single ``project`` operation is the API doc's worked example: interpolate →
MAC-project φ → momentum predictor → pressure Poisson → velocity correct → IBM
apply, leaving ``U`` divergence-free. State + fields are resolved from the
Context at run time (``ctx.models`` / ``ctx.fields``), never captured in the
operation closure — mirroring the op-closure-cycle rule.
"""

from typing import Annotated, Any, Callable

from neofoam.framework.dependency_resolver import wrap_with_dependency_resolution
from neofoam.framework.initialization import field, lazy, model as init_model
from neofoam.framework.operations import Operation, Operations, SequentialOp
from neofoam.framework.types import OperationMetadata

from ..incompressibleFluidBlockAMRModel import Model
from ..bc_mapping import build_vector_bc

chorinProjection = Model("ChorinProjection")


@chorinProjection.build
def build(self: Any) -> list[Any]:
    """Construct the projection state and register U/p/phi + the state model.

    Reads the resources ``create_fields`` provides: ``_blockamr_mesh`` and the
    validated ``_mesh_cfg`` / ``_solution_cfg`` / ``_control_cfg`` /
    ``_fvschemes_cfg`` / ``_sol_u_cfg`` / ``_sol_p_cfg`` configs.
    """

    def create_state(context: dict[str, Any]) -> Any:
        from neon.blockamr.fillpatch import FillPatchCellConservative
        from neon.blockamr.incompressible import build_incompressible

        mesh = context["_blockamr_mesh"]
        mesh_cfg = context["_mesh_cfg"]
        sol_cfg = context["_solution_cfg"]
        ctrl_cfg = context["_control_cfg"]

        # fvSchemes: discretisation scheme NAMES (the engine's registry resolves
        # them), bound to UEqn/pEqn at construction.
        schemes: dict[str, Any] = context["_fvschemes_cfg"].resolve()
        # Per-field fvSolution.solvers[<field>] blocks. ``sol_U`` carries the
        # velocity field's ``ibm`` method (empty on non-cylinder cases);
        # ``sol_p`` the pressure MLMG rtol/atol/maxIter/bottomSolver/verbosity.
        sol_U: dict[str, Any] = context["_sol_u_cfg"].resolve()
        sol_p: dict[str, Any] = context["_sol_p_cfg"].resolve()

        if all(mesh_cfg.periodicity):
            # Fully periodic: no domain BCs — use the conservative fill-patch.
            return build_incompressible(
                mesh,
                sol_cfg.nu,
                ctrl_cfg.deltaT,
                fill_patch=FillPatchCellConservative(),
                schemes=schemes,
                sol_U=sol_U,
                sol_p=sol_p,
            )
        # Walled/open domain: map the per-face boundary spec to a VectorBC.
        u_bc = build_vector_bc(mesh_cfg.boundary or {})
        return build_incompressible(
            mesh,
            sol_cfg.nu,
            ctrl_cfg.deltaT,
            U_bc=u_bc,
            schemes=schemes,
            sol_U=sol_U,
            sol_p=sol_p,
        )

    def alias_state(context: dict[str, Any]) -> Any:
        return context["_projection_state"]

    def get_u(context: dict[str, Any]) -> Any:
        return context["_projection_state"].U

    def get_p(context: dict[str, Any]) -> Any:
        return context["_projection_state"].p

    def get_phi(context: dict[str, Any]) -> Any:
        return context["_projection_state"].phi

    return [
        lazy(
            "_projection_state",
            create_state,
            depends_on=[
                "_blockamr_mesh",
                "_mesh_cfg",
                "_solution_cfg",
                "_control_cfg",
                "_fvschemes_cfg",
                "_sol_u_cfg",
                "_sol_p_cfg",
            ],
        ),
        init_model("projection_state", alias_state, depends_on=["_projection_state"]),
        field("U", get_u, depends_on=["_projection_state"], write=True),
        field("p", get_p, depends_on=["_projection_state"], write=True),
        field("phi", get_phi, depends_on=["_projection_state"]),
    ]


@chorinProjection.operation(operation_number="2.0")
def project(
    projection_state: Annotated[Any, "models"],
    U: Annotated[Any, "fields"],
    p: Annotated[Any, "fields"],
    phi: Annotated[Any, "fields"],
) -> None:
    """Advance one fractional step (projection keeps U divergence-free).

    The API doc's worked example, written in the DSL. Mutates ``U`` / ``p`` /
    ``phi`` in place — the same objects registered in ``ctx.fields`` — so no
    ``FieldUpdates`` are needed. Faithfully reproduces the pre-refactor
    ``step()`` order (BC fills, ``pEqn.sigma = dt``, post-``correct`` IBM apply);
    the numerics oracle is identical.
    """
    from neon.blockamr.dsl import exp
    from neon.blockamr.ibm import IBM
    from neon.blockamr.operators.correct import correct
    from neon.blockamr.operators.interpolate import interpolate
    from neon.blockamr.operators.mac_project import mac_project

    st = projection_state
    dt = st.dt
    t = st.t
    mesh = U.mesh
    n_levels = mesh.n_levels()

    for lev in range(n_levels):
        U.fill_patch(lev, t)

    interpolate(U, phi)
    mac_project(phi, st.sol_p)

    st.UEqn.solve(dt=dt, t=t, solution=st.sol_U)

    for lev in range(n_levels):
        U.fill_patch(lev, t)

    st.pEqn.implicit_lhs.sigma = dt
    st.pEqn.implicit_lhs.coefficient = dt
    st.pEqn.solve(dt=dt, t=t, solution=st.sol_p)

    correct(U, -dt * exp.grad(p))

    # Carry from phase 04: solve() does NOT consume solution["ibm"] — direct
    # forcing is applied once per step AFTER correct() (order matches the
    # pre-refactor engine exactly, the Cd/Cl/St acceptance oracle).
    ibm_name = st.sol_U.get("ibm")
    if ibm_name is not None:
        method = IBM.lookup(ibm_name)
        method.apply(U, dt, t, mesh.ibm_data(method))

    st.t += dt


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
    """Expose the single ``project`` op for the solver's execution graph."""
    model_ops = Operations()
    wrapped_project = wrap_with_dependency_resolution(
        project, self, chorinProjection._dependency_resolver
    )
    model_ops.add(
        _alias_operation(wrapped_project, operation_name="project", depends_on=[])
    )
    return model_ops
