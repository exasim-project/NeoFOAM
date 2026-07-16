# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""VER-2 — Poisson / MLMG order of accuracy against a manufactured solution.

Manufactured solution on ``[0,1]^3``::

    phi*(x,y,z) = sin(2 pi x) sin(2 pi y) sin(2 pi z)
    f(x,y,z)    = laplacian(phi*) = -12 pi^2 phi*

``phi*`` vanishes on every face, so homogeneous Dirichlet BCs are exact. We solve
``laplacian(phi) = f`` with the same AMReX MLMG machinery the pressure step drives
(``MLPoisson`` + ``MLMG``), refine the grid over ``N in {32,64,128}``, and assert the
discrete L2 error of ``phi - phi*`` converges at the design 2nd order — the
observed order, not a single-grid threshold. The MLMG residual is also asserted to
reach its tolerance (the solver actually converged, it did not stall).
"""

import math

import numpy as np
import pytest

pytest.importorskip("neon")

import blockamr  # noqa: E402
from blockamr.mesh import Mesh  # noqa: E402

from incompressibleFluidBlockAMR.verification_helpers import (  # noqa: E402
    l2_error,
    observed_order,
)

TWO_PI = 2.0 * math.pi


def _unit_cube(n):
    box = blockamr.Box([0, 0, 0], [n - 1, n - 1, n - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [0, 0, 0])
    box_array = blockamr.BoxArray(box)
    box_array.max_size(n)
    dm = blockamr.DistributionMapping(box_array)
    return Mesh(box_array, dm, geom), geom, box_array, dm


def _valid_cell_centres(arr, lo, dx):
    """Cell-centre coordinate arrays for a valid (ngrow-stripped) box array."""
    nx, ny, nz = arr.shape[:3]
    xs = np.array([(lo[0] + i + 0.5) * dx[0] for i in range(nx)])
    ys = np.array([(lo[1] + j + 0.5) * dx[1] for j in range(ny)])
    zs = np.array([(lo[2] + k + 0.5) * dx[2] for k in range(nz)])
    return np.meshgrid(xs, ys, zs, indexing="ij")


def _sin3d(X, Y, Z):
    return np.sin(TWO_PI * X) * np.sin(TWO_PI * Y) * np.sin(TWO_PI * Z)


def _poisson_mms(n):
    """Solve laplacian(phi)=f at resolution n; return (L2 error, MLMG residual)."""
    _, geom, box_array, dm = _unit_cube(n)
    dx = geom.cell_size()
    cell_vol = float(dx[0] * dx[1] * dx[2])

    lp = blockamr.MLPoisson(geom, box_array, dm)
    lp.set_domain_bc(
        [blockamr.LinOpBCType.Dirichlet] * 3,
        [blockamr.LinOpBCType.Dirichlet] * 3,
    )
    lp.set_level_bc(0, None)

    sol = blockamr.MultiFab(box_array, dm, 1, 1)  # initial guess 0
    rhs = blockamr.MultiFab(box_array, dm, 1, 0)

    for mfi in blockamr.MFIterator(rhs):
        arr = rhs.copy_to_host(mfi)
        lo = mfi.valid_box().small_end()
        X, Y, Z = _valid_cell_centres(arr, lo, dx)
        arr[:, :, :, 0] = -12.0 * math.pi**2 * _sin3d(X, Y, Z)
        rhs.copy_from(mfi, arr)

    mlmg = blockamr.MLMG(lp)
    mlmg.set_verbose(0)
    mlmg.set_max_iter(200)
    mlmg.set_bottom_verbose(0)
    residual = mlmg.solve(sol, rhs, 1e-11, 1e-13)

    numeric, exact = [], []
    for mfi in blockamr.MFIterator(sol):
        arr = sol.copy_to_host(mfi)
        lo = mfi.valid_box().small_end()
        X, Y, Z = _valid_cell_centres(arr, lo, dx)
        numeric.append(np.asarray(arr[:, :, :, 0]).ravel())
        exact.append(_sin3d(X, Y, Z).ravel())

    err = l2_error(np.concatenate(numeric), np.concatenate(exact), cell_vol)
    return err, float(residual)


def test_poisson_mms_is_second_order(blockamr_session):
    resolutions = (32, 64, 128)
    results = [_poisson_mms(n) for n in resolutions]
    errors = [e for e, _ in results]
    residuals = [r for _, r in results]

    # The solver actually converged on every grid (did not stall at max_iter).
    assert all(r < 1e-9 for r in residuals), f"MLMG residuals {residuals}"

    order = observed_order(errors)
    assert order > 1.8, f"Poisson order {order:.3f}, L2 errors={errors}"
    # Secondary gross-bug guard: coarse-grid error is already small.
    assert errors[0] < 1e-2, f"coarse-grid L2 error too large: {errors[0]:.3e}"
