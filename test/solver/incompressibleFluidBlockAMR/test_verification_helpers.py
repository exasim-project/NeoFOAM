# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Unit tests for the pure-Python verification harness + analytic solutions.

No GPU, no ``neon``/``jax`` import — these are plain-numpy building blocks that the
GPU convergence slices build on.
"""

import numpy as np

from .verification_helpers import (
    kovasznay,
    l2_error,
    observed_order,
    poiseuille,
    taylor_green,
)


def test_observed_order_recovers_second_order():
    # err = C * h**2, grid refined by 2 -> err drops by 4 each level.
    errors = [4.0, 1.0, 0.25]
    assert observed_order(errors) == pytest_approx(2.0)


def test_observed_order_recovers_first_order():
    errors = [4.0, 2.0, 1.0]  # err = C * h
    assert observed_order(errors) == pytest_approx(1.0)


def test_observed_order_honours_refine_ratio():
    # refine by 3: err = C*h**2 -> drops by 9 each level.
    errors = [9.0, 1.0, 1.0 / 9.0]
    assert observed_order(errors, refine_ratio=3) == pytest_approx(2.0)


def test_l2_error_zero_when_equal():
    a = np.linspace(0.0, 1.0, 50)
    assert l2_error(a, a.copy(), cell_vol=0.02) == pytest_approx(0.0)


def test_l2_error_constant_offset():
    n, delta, vol = 64, 0.1, 0.001
    num = np.zeros(n)
    exact = np.full(n, delta)
    expected = np.sqrt(n * delta**2 * vol)
    assert l2_error(num, exact, cell_vol=vol) == pytest_approx(expected)


def test_taylor_green_divergence_free():
    # Central-difference divergence of the analytic TG field ~ 0.
    n = 64
    coord = (np.arange(n) + 0.5) * (2 * np.pi / n)
    x, y = np.meshgrid(coord, coord, indexing="ij")
    u, v, _ = taylor_green(x, y, t=0.3, nu=0.05)
    h = 2 * np.pi / n
    dudx = (np.roll(u, -1, 0) - np.roll(u, 1, 0)) / (2 * h)
    dvdy = (np.roll(v, -1, 1) - np.roll(v, 1, 1)) / (2 * h)
    assert np.max(np.abs(dudx + dvdy)) < 1e-3


def test_taylor_green_decay_envelope():
    x = np.array([0.0])
    y = np.array([np.pi / 2])  # sin(y)=1 so |u| peaks
    nu = 0.1
    u0, _, _ = taylor_green(x, y, t=0.0, nu=nu)
    ut, _, _ = taylor_green(x, y, t=1.0, nu=nu)
    assert ut[0] / u0[0] == pytest_approx(np.exp(-2 * nu * 1.0))


def test_poiseuille_noslip_and_peak():
    h, G, mu = 1.0, 2.0, 0.5
    y = np.array([-h, 0.0, h])
    u = poiseuille(y, G=G, mu=mu, h=h)
    assert u[0] == pytest_approx(0.0)
    assert u[2] == pytest_approx(0.0)
    assert u[1] == pytest_approx(G / (2 * mu) * h**2)  # centreline peak


def test_kovasznay_matches_closed_form():
    Re = 40.0
    lam = Re / 2 - np.sqrt(Re**2 / 4 + 4 * np.pi**2)
    x = np.array([0.5])
    y = np.array([0.25])
    u, v, p = kovasznay(x, y, Re=Re)
    assert u[0] == pytest_approx(1 - np.exp(lam * 0.5) * np.cos(2 * np.pi * 0.25))
    assert v[0] == pytest_approx((lam / (2 * np.pi)) * np.exp(lam * 0.5) * np.sin(2 * np.pi * 0.25))
    assert p[0] == pytest_approx(0.5 * (1 - np.exp(2 * lam * 0.5)))


# local approx helper avoids a pytest import just for tolerance comparisons
def pytest_approx(expected, rel=1e-6, abs_=1e-9):
    import pytest  # noqa: PLC0415

    return pytest.approx(expected, rel=rel, abs=abs_)
