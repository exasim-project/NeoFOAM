# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Pure-Python verification harness + analytic solutions (Spec 02).

Grid-refinement helpers (``l2_error``, ``observed_order``) and the closed-form
Navier-Stokes solutions used by the blockAMR order-of-accuracy tests. These are
plain ``numpy`` building blocks — **no** ``neon``/``jax`` import — so they unit-test
GPU-free. The engine-driving ``run_at_resolution`` body is added in a later slice
(VER-4); it is stubbed here.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np


def l2_error(numeric: np.ndarray, exact: np.ndarray, cell_vol: float) -> float:
    """Discrete L2 norm of ``numeric - exact``: ``sqrt(sum(diff**2) * cell_vol)``."""
    diff = np.asarray(numeric, dtype=float) - np.asarray(exact, dtype=float)
    return float(np.sqrt(np.sum(diff**2) * cell_vol))


def observed_order(errors: list[float], refine_ratio: int = 2) -> float:
    """Least-squares slope of ``log(err)`` vs ``log(h)`` over a refinement sweep.

    ``errors[i]`` is the error on the ``i``-th grid; each successive grid refines
    the mesh spacing by ``refine_ratio`` (so ``h_i`` is proportional to
    ``refine_ratio**(-i)``). For ``err = C * h**p`` this returns ``p``.
    """
    err = np.asarray(errors, dtype=float)
    if err.size < 2:
        raise ValueError("observed_order needs >= 2 error values")
    if np.any(err <= 0.0):
        raise ValueError("observed_order needs strictly positive errors")
    log_h = -np.arange(err.size) * np.log(float(refine_ratio))
    slope, _ = np.polyfit(log_h, np.log(err), 1)
    return float(slope)


def run_at_resolution(case_builder: Callable[[int], Any], N: int) -> dict[str, np.ndarray]:
    """Build + run a blockAMR case at resolution ``N``, return final fields as numpy.

    ``case_builder(N)`` builds a single-level blockAMR case at ``N`` cells per axis,
    advances the engine to its final state (fixed steps for an unsteady problem, to
    steady state otherwise), and returns ``(solver, mesh)`` where ``solver.U`` is the
    cell-centred velocity ``CellField``. This helper marshals that final velocity to
    flat numpy arrays over all valid cells, tagged with their cell-centre
    coordinates, so a caller can build an analytic reference and call
    :func:`l2_error`.

    Returns a dict with flat, index-aligned arrays ``u``/``v``/``w`` (velocity
    components) and ``x``/``y``/``z`` (cell-centre coordinates), plus the scalar
    ``cell_vol``. ``neon`` is imported lazily here so importing this module stays
    GPU-free (the pure-math helpers above never touch the engine).
    """
    import blockamr  # noqa: PLC0415 — lazy: keep module import GPU-free

    solver, mesh = case_builder(N)
    geom = mesh.geom(0)
    dx = geom.cell_size()
    prob_lo = geom.prob_lo()

    cols: dict[str, list[np.ndarray]] = {k: [] for k in ("u", "v", "w", "x", "y", "z")}
    mf = solver.U.mf[0]
    for mfi in blockamr.MFIterator(mf):
        arr = np.asarray(mf.copy_to_host(mfi))  # (nx, ny, nz, 3) valid region
        lo = mfi.valid_box().small_end()
        nx, ny, nz = arr.shape[:3]
        xs = prob_lo[0] + (np.arange(nx) + lo[0] + 0.5) * dx[0]
        ys = prob_lo[1] + (np.arange(ny) + lo[1] + 0.5) * dx[1]
        zs = prob_lo[2] + (np.arange(nz) + lo[2] + 0.5) * dx[2]
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
        cols["u"].append(arr[:, :, :, 0].ravel())
        cols["v"].append(arr[:, :, :, 1].ravel())
        cols["w"].append(arr[:, :, :, 2].ravel())
        cols["x"].append(X.ravel())
        cols["y"].append(Y.ravel())
        cols["z"].append(Z.ravel())

    out: dict[str, np.ndarray] = {k: np.concatenate(v) for k, v in cols.items()}
    out["cell_vol"] = np.asarray(float(dx[0] * dx[1] * dx[2]))
    return out


def taylor_green(
    x: np.ndarray, y: np.ndarray, t: float, nu: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """2D Taylor-Green vortex on ``[0, 2*pi]**2`` (periodic), viscosity ``nu``.

    ``u = -cos(x) sin(y) e^{-2 nu t}``, ``v = sin(x) cos(y) e^{-2 nu t}``,
    ``p = -1/4 (cos 2x + cos 2y) e^{-4 nu t}``. Divergence-free for all ``t``.
    """
    decay = np.exp(-2.0 * nu * t)
    u = -np.cos(x) * np.sin(y) * decay
    v = np.sin(x) * np.cos(y) * decay
    p = -0.25 * (np.cos(2.0 * x) + np.cos(2.0 * y)) * decay**2
    return u, v, p


def kovasznay(x: np.ndarray, y: np.ndarray, Re: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Steady Kovasznay flow at Reynolds number ``Re``.

    ``lam = Re/2 - sqrt(Re**2/4 + 4 pi**2)``;
    ``u = 1 - e^{lam x} cos(2 pi y)``, ``v = (lam/2pi) e^{lam x} sin(2 pi y)``,
    ``p = 1/2 (1 - e^{2 lam x})``.
    """
    lam = Re / 2.0 - np.sqrt(Re**2 / 4.0 + 4.0 * np.pi**2)
    ex = np.exp(lam * x)
    u = 1.0 - ex * np.cos(2.0 * np.pi * y)
    v = (lam / (2.0 * np.pi)) * ex * np.sin(2.0 * np.pi * y)
    p = 0.5 * (1.0 - np.exp(2.0 * lam * x))
    return u, v, p


def poiseuille(y: np.ndarray, G: float, mu: float, h: float) -> np.ndarray:
    """Plane Poiseuille profile ``u(y) = G/(2 mu) (h**2 - y**2)`` (no-slip at +/- h)."""
    return G / (2.0 * mu) * (h**2 - np.asarray(y, dtype=float) ** 2)
