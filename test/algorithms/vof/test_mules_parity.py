# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Cross-backend parity of MULES explicit solve and correct (pybFoam vs NeoFOAM bindings).

The phase fraction is the ``sharpSphere`` case's sphere in a prescribed velocity,
``phi = fvc::flux(U)``: a solid-body rotation with a vertical drift or a uniform
translation. Two fields cover the boundary treatment of MULES: ``alpha`` has
zeroGradient walls; ``alphaFixed`` has fixedValue 1 walls, which with the
zeroGradient ``U`` carry alpha inflow and outflow and so exercise OpenFOAM's
widening of the local extrema at fixed-value patches. ``mules.correct`` with a
base field whose sphere is shifted (``alphaFixedShifted``, the ``alpha-alphaFixed``
case) gives a nonzero boundary correction flux, which exercises the outlet
limiting of ``limiterCorr``.

The tolerance is a relative L2 norm, as in the steady simpleFoam parity tests, and
the same on every executor. Serial follows OpenFOAM's arithmetic order, but
pybFoam and the prebuilt OpenFOAM may differ in FMA use (as on the ubuntu-latest
CI runner), which moves the last bits. On CPU (Kokkos threads) and GPU the
per-cell sums are atomic and add up in a different order (the GPU also fuses
multiply-adds), with the same effect. The interface normal
``gradAlphaf/(|gradAlphaf| + deltaN)`` with ``deltaN = 1e-8/h`` amplifies that to
~1e-8 of the compression flux peak, and the nonlinear limiter steps grow it in
alpha (seen: 5e-7 relative L2 on the GPU, 1.4e-6 peak difference per cell in CI).
"""

from __future__ import annotations

from typing import Callable

import pytest

from ._parity_backends import _assert_relative_l2, _Simulation, _use_velocity, nb, pyb
from .conftest import EXECUTORS, MESH_NAMES

FIELDS = ["alpha", "alphaFixed"]

DELTA_T = 0.02  # Courant number below 0.3 on all meshes for |U| <= 0.75


_ALPHA_L2_TOL = 5e-6


@pytest.mark.parametrize("executor", EXECUTORS)
@pytest.mark.parametrize("mesh", MESH_NAMES)
@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("n_steps", [1, 20])
@pytest.mark.parametrize("rotate", [True, False], ids=["rotation", "translation"])
@pytest.mark.parametrize("n_corr", [1, 2], ids=["nAlphaCorr1", "nAlphaCorr2"])
def test_mules_explicit_solve_matches_openfoam(
    parity_simulation: Callable[[str, str], _Simulation],
    mesh: str,
    executor: str,
    field: str,
    n_steps: int,
    rotate: bool,
    n_corr: int,
) -> None:
    """With nAlphaCorr 2 the second corrector limits around the corrected alpha but
    advances from the step's start (psi.oldTime()), as interFoam's alphaEqn does."""
    sim = parity_simulation(mesh, executor)
    _use_velocity(sim, rotate)
    alpha = sim.field(field)

    expected = pyb.mules.explicit_solve(alpha, DELTA_T, 1.0, n_steps, n_corr)
    result = nb.mules.explicit_solve(alpha, DELTA_T, 1.0, n_steps, n_corr)

    _assert_relative_l2(
        result,
        expected,
        _ALPHA_L2_TOL,
        f"{mesh}/{executor}/{field} {n_steps} steps x {n_corr} correctors",
    )


@pytest.mark.parametrize("executor", EXECUTORS)
@pytest.mark.parametrize("mesh", MESH_NAMES)
@pytest.mark.parametrize(
    ("field", "base"),
    [
        ("alpha", "alpha"),
        ("alphaFixed", "alphaFixed"),
        pytest.param("alpha", "alphaFixedShifted", id="alpha-alphaFixed"),
    ],
)
def test_mules_correct_matches_openfoam(
    parity_simulation: Callable[[str, str], _Simulation],
    mesh: str,
    executor: str,
    field: str,
    base: str,
) -> None:
    sim = parity_simulation(mesh, executor)
    _use_velocity(sim)
    alpha = sim.field(field)

    expected = pyb.mules.correct(alpha, sim.field(base), DELTA_T, 1.0)
    result = nb.mules.correct(alpha, sim.field(base), DELTA_T, 1.0)

    _assert_relative_l2(result, expected, _ALPHA_L2_TOL, f"{mesh}/{executor}/{field}<-{base}")
