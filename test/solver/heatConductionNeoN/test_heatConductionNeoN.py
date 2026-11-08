# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""heatConductionNeoN on one region: the 1-D analytic solutions, and its errors.

``cases/bar`` is a bar on [0, 1], 8 cells: 400 K at x = 0, 300 K at x = 1,
adiabatic walls, ``alpha = k / (rho cp) = 1``. It starts from the steady profile plus the first
sine mode, ``T0 = 400 - 100 x + 50 sin(pi x)``, so

    T(x, t) = 400 - 100 x + 50 exp(-pi^2 alpha t) sin(pi x).

The linear part is exact on the mesh; the sine mode is an eigenvector of the
discrete laplacian (eigenvalue ``4/dx^2 sin^2(pi dx/2)``), so each implicit
Euler step multiplies its amplitude by ``1 / (1 + dt lambda_h)`` exactly. With
``steadyState`` in place of Euler it is the linear profile after one step.
"""

import math
import re
from pathlib import Path

import numpy as np
import pytest

from solver.heatConductionNeoN._run_case import cell_centres, mesh_copy, result, run_worker

ALPHA, END_TIME, DELTA_T, CELLS, AMPLITUDE = 1.0, 0.05, 0.001, 8, 50.0


def _linear(x: np.ndarray) -> np.ndarray:
    return 400.0 - 100.0 * x


def _run(case_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    region = result(case_dir, "single")["regions"]["case"]
    return cell_centres(case_dir)[:, 0], np.asarray(region["T"])


def test_transient_bar_decays_as_the_analytic_sine_mode(tmp_path: Path) -> None:
    x, T = _run(mesh_copy("bar", tmp_path / "bar"))
    mode = np.sin(math.pi * x)
    amplitude = (T - _linear(x)) @ mode / (mode @ mode)
    # what is left of the mode is the mode: the linear part is exact
    np.testing.assert_allclose(T, _linear(x) + amplitude * mode, rtol=0, atol=1e-8)
    dx, steps = 1.0 / CELLS, round(END_TIME / DELTA_T)
    lambda_h = 4.0 / dx**2 * math.sin(math.pi * dx / 2) ** 2
    assert amplitude == pytest.approx(
        AMPLITUDE / (1 + DELTA_T * ALPHA * lambda_h) ** steps, rel=1e-9
    )
    # against the continuous solution: within the error of the mesh and of the time step
    assert amplitude == pytest.approx(
        AMPLITUDE * math.exp(-(math.pi**2) * ALPHA * END_TIME), rel=0.02
    )


def test_steady_bar_is_the_linear_profile(tmp_path: Path) -> None:
    case_dir = mesh_copy("bar", tmp_path / "bar")
    schemes = case_dir / "system" / "fvSchemes"
    schemes.write_text(
        re.sub(r"default\s+Euler;", "default         steadyState;", schemes.read_text())
    )
    x, T = _run(case_dir)
    np.testing.assert_allclose(T, _linear(x), rtol=0, atol=1e-8)


def _case_with_coupled(tmp_path: Path, coupled: str) -> Path:
    """The left block of ``cases/coupledBlocks`` run alone, its interface coupled as ``coupled``."""
    case_dir = mesh_copy("coupledBlocks", tmp_path / "blocks") / "left"
    field = case_dir / "0" / "T"
    field.write_text(field.read_text().replace("coupledHeatFlux", coupled))
    return case_dir


@pytest.mark.parametrize(
    ("coupled", "message"),
    [
        # the case couples no region: the coupled patch has no exchange
        ("coupledHeatFlux", "the case does not couple its region"),
        ("coupledTemperature", "the case does not couple its region"),
        ("coupledPressure", "coupled as 'coupledPressure'; known: "),
    ],
)
def test_a_coupled_patch_the_case_does_not_couple_raises(
    tmp_path: Path, coupled: str, message: str
) -> None:
    done = run_worker(_case_with_coupled(tmp_path, coupled), "single")
    assert done.returncode != 0
    assert "ValueError" in done.stderr and message in done.stderr, done.stderr[-3000:]
