# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

# NB: no ``from __future__ import annotations`` — the dependency_resolver matches
# ``Annotated[..., "models"]`` / ``param.annotation is Context`` live.

"""Chorin fractional-step projection backed by ``neon.blockamr``.

One member of the projection family. ``@build`` constructs the
``DSLIncompressibleSolver`` engine (MAC projection + nodal pressure solve) from
the mesh + configs assembled in ``create_fields`` and registers the engine's own
``U`` / ``p`` / ``phi`` fields into the Context. The single ``project``
operation advances the engine one fractional step: interpolate → MAC-project φ →
momentum predictor → pressure Poisson → velocity correct (all inside
``engine.step()``), leaving ``U`` divergence-free.

The engine is resolved from the Context at run time (``ctx.models``), never
captured in the operation closure — mirroring the op-closure-cycle rule.
"""

from typing import Annotated, Any, Callable, Optional

from neofoam.framework.dependency_resolver import wrap_with_dependency_resolution
from neofoam.framework.initialization import field, lazy, model as init_model
from neofoam.framework.operations import Operation, Operations, SequentialOp
from neofoam.framework.types import OperationMetadata

from ..incompressibleFluidBlockAMRModel import Model
from ..bc_mapping import build_vector_bc

chorinProjection = Model("ChorinProjection")


def _make_div_scheme(name: str) -> Optional[Any]:
    """Map a ``divScheme`` keyword to a ``neon.blockamr.schemes`` object.

    ``None`` lets ``DSLIncompressibleSolver`` fall back to its default (Upwind).
    """
    from neon.blockamr.schemes import QUICK, Linear, Upwind, VanLeer

    table = {
        "upwind": Upwind,
        "linear": Linear,
        "vanleer": VanLeer,
        "quick": QUICK,
    }
    cls = table.get(name.strip().lower())
    return cls() if cls is not None else None


@chorinProjection.build
def build(self: Any) -> list[Any]:
    """Construct the engine and register U/p/phi + the engine model.

    Reads the resources ``create_fields`` provides: ``_blockamr_mesh`` and the
    validated ``_mesh_cfg`` / ``_solution_cfg`` / ``_control_cfg`` configs.
    """

    def create_engine(context: dict[str, Any]) -> Any:
        from neon.blockamr.dsl_solver import DSLIncompressibleSolver
        from neon.blockamr.fillpatch import FillPatchCellConservative

        mesh = context["_blockamr_mesh"]
        mesh_cfg = context["_mesh_cfg"]
        sol_cfg = context["_solution_cfg"]
        ctrl_cfg = context["_control_cfg"]

        # MLMG solve controls from the fvSolution ``blockAMR`` subdict, threaded
        # to the engine as the ``sol_p`` (fvSolution.solvers['p']) parameter.
        # ``verbose`` / ``bottomVerbose`` set the AMReX residual-trace level
        # (0 = quiet).
        sol_p: dict[str, Any] = {
            "rtol": sol_cfg.rtol,
            "atol": sol_cfg.atol,
            "maxIter": sol_cfg.maxIter,
            "verbose": sol_cfg.verbose,
            "bottomVerbose": sol_cfg.bottomVerbose,
        }
        # optional explicit MLMG bottom solver (empty → AMReX default)
        if sol_cfg.bottomSolver:
            sol_p["bottomSolver"] = sol_cfg.bottomSolver

        # fvSchemes: discretisation scheme names, bound to UEqn/pEqn at
        # construction. Omitting the key lets the engine fall back to its
        # default (Upwind).
        schemes: dict[str, Any] = {}
        div_scheme = _make_div_scheme(sol_cfg.divScheme)
        if div_scheme is not None:
            schemes["div(phi,U)"] = div_scheme

        # Immersed cylinder body → direct-forcing IBM spec for the engine.
        eb = None
        if mesh_cfg.eb.type == "cylinder":
            eb = {
                "center": mesh_cfg.eb.center,
                "radius": mesh_cfg.eb.radius,
                "axis": mesh_cfg.eb.axis,
            }

        if all(mesh_cfg.periodicity):
            # Fully periodic: no domain BCs — use the conservative fill-patch.
            return DSLIncompressibleSolver(
                mesh,
                sol_cfg.nu,
                ctrl_cfg.deltaT,
                fill_patch=FillPatchCellConservative(),
                schemes=schemes,
                sol_p=sol_p,
                eb=eb,
            )
        # Walled/open domain: map the per-face boundary spec to a VectorBC.
        u_bc = build_vector_bc(mesh_cfg.boundary or {})
        return DSLIncompressibleSolver(
            mesh,
            sol_cfg.nu,
            ctrl_cfg.deltaT,
            U_bc=u_bc,
            schemes=schemes,
            sol_p=sol_p,
            eb=eb,
        )

    def alias_engine(context: dict[str, Any]) -> Any:
        return context["_blockamr_engine"]

    def get_u(context: dict[str, Any]) -> Any:
        return context["_blockamr_engine"].U

    def get_p(context: dict[str, Any]) -> Any:
        return context["_blockamr_engine"].p

    def get_phi(context: dict[str, Any]) -> Any:
        return context["_blockamr_engine"].phi

    return [
        lazy(
            "_blockamr_engine",
            create_engine,
            depends_on=[
                "_blockamr_mesh",
                "_mesh_cfg",
                "_solution_cfg",
                "_control_cfg",
            ],
        ),
        init_model("blockamr_engine", alias_engine, depends_on=["_blockamr_engine"]),
        field("U", get_u, depends_on=["_blockamr_engine"], write=True),
        field("p", get_p, depends_on=["_blockamr_engine"], write=True),
        field("phi", get_phi, depends_on=["_blockamr_engine"]),
    ]


@chorinProjection.operation(operation_number="2.0")
def project(blockamr_engine: Annotated[Any, "models"]) -> None:
    """Advance the engine one fractional step (projection keeps U divergence-free).

    ``engine.step()`` mutates ``U`` / ``p`` / ``phi`` in place — the same objects
    registered in ``ctx.fields`` — so no ``FieldUpdates`` are needed.
    """
    blockamr_engine.step()


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
