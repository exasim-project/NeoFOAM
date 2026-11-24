# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""INT-8 — observable post-processing on canned fields (Spec 03, P2/P3).

Locks the observable *math* deterministically and GPU-free, so an E2E band
failure points at the physics, not the post-processing. Every test builds a
synthetic field (or series) with a known answer and checks the extractor
recovers it:

* :func:`momentum_deficit_forces` — a uniform stream with a wake deficit yields a
  positive drag of the right order; a pure uniform stream yields ~zero.
* :func:`strouhal` — a synthetic sinusoidal ``Cl(t)`` yields the injected
  ``St = f D / U∞`` and ignores a DC offset.
* :func:`recirculation_length` — a centreline profile with a controlled ``u = 0``
  up-crossing yields the injected ``Lr/D``.
* :func:`separation_angle` — an azimuthal field reversing at a known angle yields
  that ``θs``.

No ``neon`` / GPU import — these are pure numpy.
"""

import numpy as np
import pytest

from neofoam.solver.incompressibleFluidBlockAMR import postpro

# ---------------------------------------------------------------------------
# helpers — build canned snapshots on a regular grid
# ---------------------------------------------------------------------------


def _grid(nx, ny, nz, lx, ly, lz):
    dx, dy, dz = lx / nx, ly / ny, lz / nz
    x = (np.arange(nx) + 0.5) * dx
    y = (np.arange(ny) + 0.5) * dy
    z = (np.arange(nz) + 0.5) * dz
    return x, y, z, dx, dy, dz


def _disc_mask(x, y, z, cx, cy, r):
    X, Y, _ = np.meshgrid(x, y, z, indexing="ij")
    return (X - cx) ** 2 + (Y - cy) ** 2 < r * r


# ---------------------------------------------------------------------------
# P2 — force coefficients (momentum-deficit CV survey)
# ---------------------------------------------------------------------------


def test_momentum_deficit_drag_is_positive_and_right_order():
    """A uniform stream with a wake velocity deficit → drag Fx > 0, O(deficit)."""
    nx, ny = 200, 100
    x, y, _, dx, dy, _ = _grid(nx, ny, 1, 4.0, 2.0, 1.0)
    U0, rho = 1.0, 1.0

    # uniform inflow; a Gaussian streamwise-velocity deficit centred in the wake,
    # with a compensating outflow through the top/bottom so mass roughly balances.
    X, Y = np.meshgrid(x, y, indexing="ij")
    deficit = 0.6 * np.exp(-((Y - 1.0) ** 2) / (2 * 0.15**2))
    wake = (X > 2.0).astype(float)
    u = U0 - deficit * wake
    v = np.zeros_like(u)
    p = np.zeros_like(u)

    cv = (5, nx - 5, 5, ny - 5)
    Fx, Fy = postpro.momentum_deficit_forces(u, v, p, x, y, dx, dy, cv, rho=rho)

    assert Fx > 0.0  # net drag
    # order: ρ ∫ u (U0 - u) dy  ~  ρ U0 * deficit_peak * width  ~ O(0.1)
    assert 0.01 < Fx < 2.0
    assert abs(Fy) < 0.05 * Fx  # symmetric wake → ~no lift


def test_momentum_deficit_uniform_stream_has_no_force():
    """A perfectly uniform stream (no body) → Fx ≈ Fy ≈ 0."""
    nx, ny = 120, 80
    x, y, _, dx, dy, _ = _grid(nx, ny, 1, 3.0, 2.0, 1.0)
    u = np.ones((nx, ny))
    v = np.zeros((nx, ny))
    p = np.zeros((nx, ny))
    Fx, Fy = postpro.momentum_deficit_forces(u, v, p, x, y, dx, dy, (5, nx - 5, 5, ny - 5))
    assert abs(Fx) < 1e-9
    assert abs(Fy) < 1e-9


# ---------------------------------------------------------------------------
# P3 — Strouhal from a synthetic lift series
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("f_true", [0.8, 1.4, 2.5])
def test_strouhal_recovers_injected_frequency(f_true):
    """FFT peak of a sinusoidal Cl(t) → St = f D / U∞."""
    dt, D, U0 = 0.005, 0.2, 1.0
    t = np.arange(0.0, 40.0, dt)
    cl = 0.3 * np.sin(2 * np.pi * f_true * t)
    St = postpro.strouhal(cl, dt, D, U0)
    assert St == pytest.approx(f_true * D / U0, rel=0.02)


def test_strouhal_ignores_dc_offset():
    """A constant offset (mean drag-like bias) must not shift the peak."""
    dt, D, U0, f_true = 0.005, 0.2, 1.0, 1.4
    t = np.arange(0.0, 40.0, dt)
    cl = 5.0 + 0.3 * np.sin(2 * np.pi * f_true * t)
    St = postpro.strouhal(cl, dt, D, U0)
    assert St == pytest.approx(f_true * D / U0, rel=0.02)


# ---------------------------------------------------------------------------
# P3 — recirculation length from a canned centreline profile
# ---------------------------------------------------------------------------


def test_recirculation_length_matches_injected_crossing():
    """Centreline u(x) that dips negative then recovers → Lr/D at the crossing."""
    nx, ny, nz = 200, 100, 4
    x, y, z, dx, dy, dz = _grid(nx, ny, nz, 4.0, 2.0, 0.25)
    D, cx, cy, r = 0.2, 1.0, 1.0, 0.1
    solid = _disc_mask(x, y, z, cx, cy, r)

    x_rear = cx + r  # ≈ back of the body
    x_reattach = x_rear + 0.93 * D  # inject Lr/D = 0.93
    X = x[:, None, None]
    # negative between rear and reattachment, positive elsewhere downstream
    u = np.tanh((X - x_reattach) / 0.05) * np.ones((nx, ny, nz))
    u[x < x_rear] = 1.0  # upstream: free stream
    v = np.zeros_like(u)
    w = np.zeros_like(u)
    p = np.zeros_like(u)
    snap = postpro.FieldSnapshot(u, v, w, p, x, y, z, dx, dy, dz, solid)

    lr = postpro.recirculation_length(snap, D, center=(cx, cy))
    assert lr == pytest.approx(0.93, rel=0.12)


def test_recirculation_length_zero_without_reversal():
    """No negative centreline velocity → no bubble → Lr = 0."""
    nx, ny, nz = 100, 60, 4
    x, y, z, dx, dy, dz = _grid(nx, ny, nz, 4.0, 2.0, 0.25)
    solid = _disc_mask(x, y, z, 1.0, 1.0, 0.1)
    u = np.ones((nx, ny, nz))
    z0 = np.zeros((nx, ny, nz))
    snap = postpro.FieldSnapshot(u, z0, z0, z0, x, y, z, dx, dy, dz, solid)
    assert postpro.recirculation_length(snap, 0.2, center=(1.0, 1.0)) == 0.0


# ---------------------------------------------------------------------------
# P3 — separation angle from a canned azimuthal field
# ---------------------------------------------------------------------------


def test_separation_angle_matches_injected_reversal():
    """Azimuthal velocity reversing at φ=45° from the rear → θs ≈ 45°."""
    nx, ny, nz = 240, 240, 4
    x, y, z, dx, dy, dz = _grid(nx, ny, nz, 2.0, 2.0, 0.25)
    cx, cy, r = 1.0, 1.0, 0.1
    solid = _disc_mask(x, y, z, cx, cy, r)

    X, Y, _ = np.meshgrid(x, y, z, indexing="ij")
    phi = np.arctan2(Y - cy, X - cx)  # angle from +x (rear) axis
    theta_s = np.deg2rad(45.0)
    g = theta_s - np.abs(phi)  # >0 inside |φ|<45°, <0 outside
    # purely azimuthal field of magnitude g: U = g * (−sinφ, cosφ) → u_t = g
    u = -g * np.sin(phi)
    v = g * np.cos(phi)
    w = np.zeros_like(u)
    p = np.zeros_like(u)
    snap = postpro.FieldSnapshot(u, v, w, p, x, y, z, dx, dy, dz, solid)

    theta = postpro.separation_angle(snap, center=(cx, cy), radius=r)
    assert theta == pytest.approx(45.0, abs=5.0)
