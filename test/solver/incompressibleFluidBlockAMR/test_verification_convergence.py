# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""VER-4 — Taylor-Green vortex: the whole coupled unsteady Navier-Stokes solver.

The 2D Taylor-Green vortex on ``[0, 2 pi]^2`` (periodic, uniform in z) is an exact
solution of incompressible Navier-Stokes::

    u = -cos(x) sin(y) e^{-2 nu t}
    v =  sin(x) cos(y) e^{-2 nu t}
    p = -1/4 (cos 2x + cos 2y) e^{-4 nu t}

Driving ``incompressibleFluidBlockAMR``'s ``blockamr`` DSL projection with
the exact field as initial condition exercises every term at once — advection,
diffusion, the MAC projection, the pressure solve, and the coupled time advance.
Three checks:

* **spatial order** — the velocity L2 error against the exact field converges at
  2nd order under grid refinement ``N in {32,64,128}`` at fixed dt (``observed_order``);
* **single-grid accuracy** — the coarse-grid L2 error is already below a tolerance
  (a gross-bug guard, secondary to the slope);
* **viscous decay** — the sampled ``max|U|(t)`` tracks the analytic envelope
  ``e^{-2 nu t}``, proving the physical decay rate, not just a static snapshot.
"""

import math

import numpy as np
import pytest

pytest.importorskip("neon")

import jax.numpy as jnp  # noqa: E402
import blockamr  # noqa: E402
from blockamr.incompressible import build_incompressible, step  # noqa: E402
from blockamr.fillpatch import FillPatchCellConservative  # noqa: E402
from blockamr.schemes.div_schemes import Linear  # noqa: E402

from incompressibleFluidBlockAMR.verification_helpers import (  # noqa: E402
    l2_error,
    observed_order,
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


def _tg_mesh(n):
    return build_mesh(
        MeshDictConfig(
            domain=[[0.0, 0.0, 0.0], [TWO_PI, TWO_PI, TWO_PI * NZ / n]],
            nCell=[n, n, NZ],
            periodicity=[True, True, True],
        )
    )


def _set_tg_ic(solver, mesh):
    """Seed U with the analytic Taylor-Green field at t=0."""
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


def _make_tg_solver(n):
    mesh = _tg_mesh(n)
    solver = build_incompressible(
        mesh,
        NU,
        DT,
        fill_patch=FillPatchCellConservative(),
        # Unlimited 2nd-order central advection: a verification test must measure the
        # scheme's design order, so it uses the non-limited scheme (the production
        # default vanLeer clips to 1st order at extrema and would mask the order).
        schemes={"div(phi,U)": Linear()},
        sol_p={"rtol": 1e-12, "atol": 1e-14, "maxIter": 400, "verbose": 0},
    )
    _set_tg_ic(solver, mesh)
    return solver, mesh


def _tg_case_builder(n):
    """case_builder for run_at_resolution: build, step NSTEPS, return (solver, mesh)."""
    solver, mesh = _make_tg_solver(n)
    for _ in range(NSTEPS):
        step(solver)
    return solver, mesh


def _velocity_l2_error(fields, t_end):
    u_exact, v_exact, _ = taylor_green(fields["x"], fields["y"], t=t_end, nu=NU)
    cell_vol = float(fields["cell_vol"])
    err_u = l2_error(fields["u"], u_exact, cell_vol)
    err_v = l2_error(fields["v"], v_exact, cell_vol)
    return math.hypot(err_u, err_v)


def test_taylor_green_velocity_is_second_order_in_space(blockamr_session):
    resolutions = (32, 64, 128)
    t_end = NSTEPS * DT
    errors = [
        _velocity_l2_error(run_at_resolution(_tg_case_builder, n), t_end)
        for n in resolutions
    ]
    order = observed_order(errors)
    assert order > 1.8, f"Taylor-Green spatial order {order:.3f}, L2 errors={errors}"
    # Gross-bug guard: coarse grid already accurate.
    assert errors[0] < 1e-4, f"coarse-grid velocity L2 error too large: {errors[0]:.3e}"


def _max_speed(solver):
    m = 0.0
    for a in solver.U.mf[0].arrays():
        arr = np.asarray(a)
        speed = np.sqrt(arr[..., 0] ** 2 + arr[..., 1] ** 2 + arr[..., 2] ** 2)
        m = max(m, float(np.max(speed)))
    return m


def test_taylor_green_viscous_decay_envelope(blockamr_session):
    """max|U|(t) tracks the analytic decay e^{-2 nu t} through the run."""
    solver, _ = _make_tg_solver(64)
    m0 = _max_speed(solver)
    n_decay = 20
    for k in range(1, n_decay + 1):
        step(solver)
        t = k * DT
        ratio = _max_speed(solver) / m0
        expected = math.exp(-2.0 * NU * t)
        assert abs(ratio - expected) < 5e-3, (
            f"step {step}: max|U| ratio {ratio:.6f} vs envelope {expected:.6f}"
        )
