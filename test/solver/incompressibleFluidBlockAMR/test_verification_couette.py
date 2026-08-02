# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""VER-6 (Couette leg) — plane Couette flow: wall boundary-condition correctness.

Plane Couette flow between a stationary no-slip wall (``y=0``) and a wall moving at
``U`` (``y=h``), streamwise/spanwise periodic. The exact steady solution is the
*linear* profile ``u(y) = U y / h`` (``v=w=0``, uniform pressure): the advective term
vanishes (``u`` depends only on ``y``) and the viscous steady state is ``laplacian u =
0``. Because the discrete gradient/laplacian are exact on a linear field, the solver
must reproduce this profile to the transient/solve tolerance at *any* resolution — so
this is a profile-match test, the direct check of the no-slip + moving-wall Dirichlet
fills and the viscous term (requirement V6, wall-BC correctness).

The pressure-gradient-driven **Poiseuille** leg of V6 is deferred: it needs an additive
body force to drive a stationary-wall channel, the same engine seam missing for VER-7
(see ``loop/02-blockAMR-code-verification/decisions.md`` D5/D3).
"""

import numpy as np
import pytest

pytest.importorskip("neon")

import blockamr  # noqa: E402
from blockamr.bc import VectorBC, fixedValue, noSlip  # noqa: E402
from blockamr.incompressible import build_incompressible, step  # noqa: E402
from blockamr.mesh import Mesh  # noqa: E402

WALL_SPEED = 1.0
HEIGHT = 1.0
NU = 0.1  # profile is Reynolds-independent; nu only sets the approach-to-steady rate


def _couette_profile(n, nsteps):
    """Run plane Couette to steady at n cells; return (y_centres, u, exact)."""
    dt = 0.2 / n
    box = blockamr.Box([0, 0, 0], [n - 1, n - 1, 3])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, HEIGHT, 4.0 / n])
    geom = blockamr.Geometry(box, rb, 0, [1, 0, 1])  # x,z periodic; y walls
    box_array = blockamr.BoxArray(box)
    box_array.max_size(n)
    dm = blockamr.DistributionMapping(box_array)
    mesh = Mesh(box_array, dm, geom)

    u_bc = VectorBC(ylo=noSlip(), yhi=fixedValue([WALL_SPEED, 0.0, 0.0]))
    solver = build_incompressible(mesh, NU, dt, U_bc=u_bc)
    for _ in range(nsteps):
        step(solver)

    mf = solver.U.mf[0]
    ng = mf.n_grow()
    dx = geom.cell_size()
    arr = np.asarray(mf.arrays()[0])
    ix, iz = n // 2, 2
    u = np.asarray(arr[ng + ix, ng : ng + n, ng + iz, 0])
    y = np.array([(j + 0.5) * dx[1] for j in range(n)])
    exact = WALL_SPEED * y / HEIGHT
    return y, u, exact


def test_couette_profile_matches_linear_exact(blockamr_session):
    """Steady Couette profile matches u(y)=U y/h at two resolutions (wall-BC correctness).

    The full-profile match includes the near-wall cells, so it also proves the no-slip
    (``y=0``) and moving-wall (``y=h``) Dirichlet fills are correct.
    """
    for n, nsteps in ((8, 800), (16, 1400)):
        _, u, exact = _couette_profile(n, nsteps)
        max_err = float(np.max(np.abs(u - exact)))
        assert max_err < 1e-5, f"N={n}: max|u - U y/h| = {max_err:.3e}"
        # Near-wall cells (no-slip below, moving wall above) sit on the exact line.
        assert abs(u[0] - exact[0]) < 1e-5, f"N={n} no-slip wall: {u[0]:.3e}"
        assert abs(u[-1] - exact[-1]) < 1e-5, f"N={n} moving wall: {u[-1]:.3e}"
