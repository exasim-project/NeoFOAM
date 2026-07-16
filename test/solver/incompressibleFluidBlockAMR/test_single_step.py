# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""INT-6 / C6 — one projection step: finite, bounded, divergence-free.

The engine is stepped directly (the framework wiring is exercised by
``test_solution_loop_integration``). IC = a Taylor-Green vortex (analytically
divergence-free); after one ``step()`` the face flux ``phi`` must be
divergence-free to the MLMG tolerance, ``U`` finite, and ``max|U|`` bounded.
"""

import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("neon")

import blockamr  # noqa: E402
from blockamr.incompressible import build_incompressible, step  # noqa: E402
from blockamr.fillpatch import FillPatchCellConservative  # noqa: E402
from blockamr.mesh import Mesh  # noqa: E402

TWO_PI = 2.0 * np.pi


def _taylor_green_ic(mf, geom):
    """u = sin x cos y, v = -cos x sin y, w = 0  (divergence-free, periodic)."""
    dx = geom.cell_size()
    for mfi in blockamr.MFIterator(mf):
        bx = mfi.valid_box()
        lo, hi = bx.small_end(), bx.big_end()
        nx, ny, nz = (hi[i] - lo[i] + 1 for i in range(3))
        xs = jnp.array([(lo[0] + i + 0.5) * dx[0] for i in range(nx)])
        ys = jnp.array([(lo[1] + j + 0.5) * dx[1] for j in range(ny)])
        x2d = xs[:, None] * jnp.ones((1, ny))
        y2d = jnp.ones((nx, 1)) * ys[None, :]
        u = jnp.sin(x2d) * jnp.cos(y2d)
        v = -jnp.cos(x2d) * jnp.sin(y2d)
        w = jnp.zeros_like(u)
        vals = jnp.stack([u, v, w], axis=-1)[:, :, None, :] * jnp.ones((1, 1, nz, 1))
        mf.copy_from(mfi, vals)


def _max_face_divergence(phi, mesh):
    """max|div(phi)| over the domain from the face flux field."""
    dx = mesh.geom(0).cell_size()
    max_abs = 0.0
    face_arrs = [phi[0][d].mf.arrays() for d in range(3)]
    for bi in range(len(face_arrs[0])):
        div_val = None
        for d in range(3):
            f = face_arrs[d][bi][:, :, :, 0]
            ng = phi[0][d].mf.n_grow()
            nc = [int(f.shape[ax]) - 2 * ng - (1 if ax == d else 0) for ax in range(3)]
            sl_hi = [slice(ng, ng + nc[ax]) for ax in range(3)]
            sl_lo = [slice(ng, ng + nc[ax]) for ax in range(3)]
            sl_hi[d] = slice(ng + 1, ng + 1 + nc[d])
            sl_lo[d] = slice(ng, ng + nc[d])
            contrib = (f[tuple(sl_hi)] - f[tuple(sl_lo)]) / dx[d]
            div_val = contrib if div_val is None else div_val + contrib
        max_abs = max(max_abs, float(jnp.max(jnp.abs(div_val))))
    return max_abs


def _max_velocity(solver):
    m = 0.0
    for a in solver.U.mf[0].arrays():
        m = max(m, float(jnp.max(jnp.abs(jnp.asarray(a)))))
    return m


def _make_periodic_tg_solver(n=16, nz=4, re=100.0, cfl=0.2):
    nu = 1.0 / re
    dt = cfl / n
    box = blockamr.Box([0, 0, 0], [n - 1, n - 1, nz - 1])
    real_box = blockamr.RealBox([0.0, 0.0, 0.0], [TWO_PI, TWO_PI, TWO_PI * nz / n])
    geom = blockamr.Geometry(box, real_box, 0, [1, 1, 1])
    ba = blockamr.BoxArray(box)
    ba.max_size(n)
    dm = blockamr.DistributionMapping(ba)
    mesh = Mesh(ba, dm, geom)
    solver = build_incompressible(
        mesh,
        nu,
        dt,
        fill_patch=FillPatchCellConservative(),
        sol_p={"rtol": 1e-12, "atol": 1e-14, "maxIter": 400, "verbose": 0},
    )
    _taylor_green_ic(solver.U.mf[0], geom)
    return solver, mesh


def test_single_step_bounded_and_divergence_free(blockamr_session):
    solver, mesh = _make_periodic_tg_solver()
    u0 = _max_velocity(solver)

    step(solver)

    # No NaNs / Infs anywhere in U.
    for a in solver.U.mf[0].arrays():
        assert np.isfinite(np.asarray(a)).all()

    # max|U| stays within a factor of the IC (no blow-up).
    u1 = _max_velocity(solver)
    assert u1 < 2.0 * u0
    assert u1 > 0.1 * u0

    # Projection worked: the face flux is divergence-free to the solve tolerance.
    assert _max_face_divergence(solver.phi, mesh) < 1e-6
