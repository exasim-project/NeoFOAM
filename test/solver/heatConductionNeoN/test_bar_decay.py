# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""heatConductionNeoN on one region: a bar decaying to the linear profile.

``cases/bar``: [0, 1], 8 cells, 400 K at x = 0, 300 K at x = 1, adiabatic walls,
``alpha = 1``, from ``400 - 100 x + 50 sin(pi x)``. The linear part is exact
on the mesh and the sine mode an eigenvector of the discrete laplacian
(``lambda_h = 4/dx^2 sin^2(pi dx/2)``), so after 50 implicit Euler steps of
1e-3 s the amplitude is exactly ``50 / (1 + dt lambda_h)^50 = 30.7906...``,
up to the linear solver (1e-12 per step; 1e-8 K). The continuous amplitude
``50 exp(-pi^2 t) = 30.5249...`` is within the mesh and time-step error, 0.6 K
(2 % of the mode). ``barSteady`` (``steadyState``) is the linear profile in one
step. The expected values are at the cell centres.
"""

from pathlib import Path

import numpy as np

from solver.heatConductionNeoN._run_case import mesh_copy, result

DISCRETE_T = [
    399.75695177, 398.356351355, 394.351464016, 386.448985859,
    373.948985859, 356.851464016, 335.856351355, 312.25695177,
]  # fmt: skip
CONTINUOUS_T = [
    399.705112817, 398.208726508, 394.130527819, 386.188373845,
    373.688373845, 356.630527819, 335.708726508, 312.205112817,
]  # fmt: skip
LINEAR_T = [393.75, 381.25, 368.75, 356.25, 343.75, 331.25, 318.75, 306.25]


def _T(case_dir: Path) -> list[float]:
    return list(result(case_dir, "single")["regions"]["case"]["T"])


def test_bar_decays_as_the_discrete_sine_mode(tmp_path: Path) -> None:
    T = _T(mesh_copy("bar", tmp_path / "bar"))

    np.testing.assert_allclose(T, DISCRETE_T, rtol=0, atol=1e-8, err_msg="bar, discrete")
    np.testing.assert_allclose(T, CONTINUOUS_T, rtol=0, atol=0.6, err_msg="bar, continuous")


def test_bar_steady_is_the_linear_profile(tmp_path: Path) -> None:
    T = _T(mesh_copy("bar", tmp_path / "bar", "barSteady"))

    np.testing.assert_allclose(T, LINEAR_T, rtol=0, atol=1e-8, err_msg="barSteady")
