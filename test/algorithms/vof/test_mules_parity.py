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

Tolerances: Serial follows OpenFOAM's arithmetic order face by face and matches
it bit for bit; ``rtol=1e-12`` only guards against compiler contraction. On CPU
(Kokkos threads) and GPU the per-cell sums are atomic and add up in a different
order (the GPU also fuses multiply-adds), which moves the last bits. The
interface normal ``gradAlphaf/(|gradAlphaf| + deltaN)`` with ``deltaN = 1e-8/h``
amplifies that to ~1e-8 of the compression flux peak, and through 20 nonlinear
limiter steps that grows to ~1e-6 in alpha (seen: 8e-7); everything else differs
by ~1e-15.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pytest

from ._parity_backends import _Simulation, _use_velocity, nb, pyb
from .conftest import EXECUTORS, MESH_NAMES

FIELDS = ["alpha", "alphaFixed"]

DELTA_T = 0.02  # Courant number below 0.3 on all meshes for |U| <= 0.75


def _alpha_tolerances(executor: str) -> tuple[float, float]:
    """(rtol, atol) for alpha, which is O(1)."""
    return (1e-12, 1e-14) if executor == "Serial" else (1e-8, 5e-6)


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

    rtol, atol = _alpha_tolerances(executor)
    np.testing.assert_allclose(
        result,
        expected,
        rtol=rtol,
        atol=atol,
        err_msg=f"{mesh}/{executor}/{field} {n_steps} steps x {n_corr} correctors",
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

    rtol, atol = _alpha_tolerances(executor)
    np.testing.assert_allclose(
        result, expected, rtol=rtol, atol=atol, err_msg=f"{mesh}/{executor}/{field}<-{base}"
    )
