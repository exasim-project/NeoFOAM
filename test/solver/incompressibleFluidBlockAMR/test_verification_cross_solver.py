# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""VER-4/5/6 cross-solver variant — per-solver L2 error table (V9).

The analytic solutions are mesh-topology-agnostic, so the same field can be pushed
through more than one solver and compared at the error-norm level. This module builds
a per-solver L2 table for the Taylor-Green field: it always runs the
``incompressibleFluidBlockAMR`` engine leg, and is structured so other solvers
(``incompressibleFluid``, a NeoN stand-in) plug in as they gain a periodic
Taylor-Green case — each such leg is gated independently and reported as *skipped*
when its solver or case is absent (never a hard failure), per the spec's
"skips absent solvers" rule.

Only the blockAMR leg runs today: the OpenFOAM ``incompressibleFluid`` solver has no
periodic / cyclic Taylor-Green case in the suite, and the Kovasznay / Poiseuille
cross-solver legs are deferred with their base slices (see
``loop/02-blockAMR-code-verification/decisions.md`` D4/D5). The table makes those
gaps explicit rather than silent.
"""

import math

import numpy as np
import pytest

pytest.importorskip("neon")

import jax.numpy as jnp  # noqa: E402
import neon.blockamr as blockamr  # noqa: E402
from neon.blockamr.dsl_solver import DSLIncompressibleSolver  # noqa: E402
from neon.blockamr.fillpatch import FillPatchCellConservative  # noqa: E402
from neon.blockamr.schemes.div_schemes import Linear  # noqa: E402

from incompressibleFluidBlockAMR.verification_helpers import (  # noqa: E402
    l2_error,
    run_at_resolution,
    taylor_green,
)
from neofoam.solver.incompressibleFluidBlockAMR.configs import (  # noqa: E402
    MeshDictConfig,
)
from neofoam.solver.incompressibleFluidBlockAMR.models.mesh_factory import (  # noqa: E402
    build_mesh,
)

TWO_PI = 2.0 * math.pi
NU = 0.05
DT = 1.5e-3
NSTEPS = 5
NZ = 4


# --- blockAMR leg ------------------------------------------------------------


def _blockamr_tg_case(n):
    mesh = build_mesh(
        MeshDictConfig(
            domain=[[0.0, 0.0, 0.0], [TWO_PI, TWO_PI, TWO_PI * NZ / n]],
            nCell=[n, n, NZ],
            periodicity=[True, True, True],
        )
    )
    solver = DSLIncompressibleSolver(
        mesh,
        NU,
        DT,
        fill_patch=FillPatchCellConservative(),
        schemes={"div(phi,U)": Linear()},
        sol_p={"rtol": 1e-12, "atol": 1e-14, "maxIter": 400, "verbose": 0},
    )
    geom = mesh.geom(0)
    dx = geom.cell_size()
    mf = solver.U.mf[0]
    for mfi in blockamr.MFIterator(mf):
        bx = mfi.valid_box()
        lo, hi = bx.small_end(), bx.big_end()
        nx, ny, nz = (hi[i] - lo[i] + 1 for i in range(3))
        xs = np.array([(lo[0] + i + 0.5) * dx[0] for i in range(nx)])
        ys = np.array([(lo[1] + j + 0.5) * dx[1] for j in range(ny)])
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        u, v, _ = taylor_green(X, Y, t=0.0, nu=NU)
        vals = np.zeros((nx, ny, nz, 3))
        vals[:, :, :, 0] = u[:, :, None]
        vals[:, :, :, 1] = v[:, :, None]
        mf.copy_from(mfi, jnp.asarray(vals, dtype=float))
    solver.U.fill_patch(0, 0.0)
    for _ in range(NSTEPS):
        solver.step()
    return solver, mesh


def _blockamr_tg_l2(n=64):
    fields = run_at_resolution(_blockamr_tg_case, n)
    t_end = NSTEPS * DT
    u_exact, v_exact, _ = taylor_green(fields["x"], fields["y"], t=t_end, nu=NU)
    cell_vol = float(fields["cell_vol"])
    return math.hypot(
        l2_error(fields["u"], u_exact, cell_vol),
        l2_error(fields["v"], v_exact, cell_vol),
    )


# --- other legs (absent today; gated so they self-report as skipped) ---------


def _incompressible_fluid_tg_l2(n=64):
    """incompressibleFluid (OpenFOAM) Taylor-Green leg — returns L2 error or None.

    Requires ``pybFoam`` *and* a periodic/cyclic Taylor-Green case, which the suite
    does not ship yet. Returns ``None`` (leg skipped) whenever either is missing,
    instead of failing the table.
    """
    try:
        import pybFoam  # noqa: F401
    except ImportError:
        return None
    # No periodic Taylor-Green case exists for incompressibleFluid yet (see D4/D5);
    # the leg is intentionally not run until such a case is added.
    return None


def _neon_standin_tg_l2(n=64):
    """NeoN stand-in Taylor-Green leg — absent in this repo; always skipped."""
    return None


CROSS_SOLVER_LEGS = {
    "incompressibleFluidBlockAMR": _blockamr_tg_l2,
    "incompressibleFluid": _incompressible_fluid_tg_l2,
    "neon-standin": _neon_standin_tg_l2,
}


def test_cross_solver_taylor_green_l2_table(blockamr_session, capsys):
    """Build a per-solver Taylor-Green L2 table; assert every leg that ran is accurate."""
    table = {name: leg() for name, leg in CROSS_SOLVER_LEGS.items()}

    lines = ["", "Taylor-Green cross-solver L2 error table:"]
    for name, err in table.items():
        lines.append(
            f"  {name:32s} {'skipped (absent)' if err is None else f'{err:.4e}'}"
        )
    with capsys.disabled():
        print("\n".join(lines))

    ran = {name: err for name, err in table.items() if err is not None}
    # At least the blockAMR leg must run, and every leg that ran must be accurate.
    assert "incompressibleFluidBlockAMR" in ran, "blockAMR leg failed to run"
    for name, err in ran.items():
        assert err < 5e-2, f"{name}: Taylor-Green L2 error {err:.3e} exceeds tol"
