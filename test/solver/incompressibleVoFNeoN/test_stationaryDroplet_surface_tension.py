# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""incompressibleVoFNeoN's surface tension against native interFoam on a stationary 2-D droplet.

The case (``cases/stationaryDroplet``) is a 2 mm water cylinder in air at the centre of an
8 mm box, 40 x 40 cells (R/dx = 10), without gravity, ``sigma 0.07``. The walls are noSlip /
fixedFluxPressure; the top is the open ``atmosphere`` with patch types NeoN and OpenFOAM
evaluate identically (``p_rgh`` fixedValue 0, ``U`` zeroGradient), which also gives p_rgh its
reference level (closed domains are not ported). ``setAlphaField`` cuts the initial volume
fractions exactly, so the interface starts sharp and the droplet should stay at rest: the
pressure jump balances the surface tension and the remaining velocity is the spurious current
of the CSF model. ``deltaT 2e-5`` is a fifth of the capillary time-step limit
``sqrt(rho_avg dx^3/(2 pi sigma))``; 200 steps reach ``t = 4e-3``.

Measures, both solvers, at the end time: the Laplace pressure ``mean p_rgh(r < R/2) -
mean p_rgh(r > 3R/2)`` against ``sigma/R`` with ``R`` the radius of the water volume, and the
peak spurious velocity. The CSF curvature ``-div(nHatf)`` of a two-cell interface is too small
at R/dx = 10: interFoam's jump is 14.7% under ``sigma/R`` (9.6% at R/dx = 20), so the physical
check is loose (20%) and the strict one is the agreement with interFoam. Measured, the same on
Serial, CPU and GPU: jump 29.9007 vs interFoam's 29.9009 Pa (sigma/R 35.04), peak |U| 0.19897
vs 0.19927 m/s, relL2 alpha.water 2.8e-6, p_rgh 9.4e-6, U 1.7e-3. The ceilings are about 10x
the measured jump and peak differences and 3x the relL2 (as in ``test_damBreak_comparison``).
The fields cannot agree to round-off: the linear solvers stop at different iterates (Ginkgo vs
OpenFOAM), and the spurious currents are the velocity's whole content.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from neofoam.io import DictFile

from ..incompressibleFluid.comparison_helpers import read_internal_fields

os.environ["FOAM_SIGFPE"] = ""

_CASE = Path(__file__).parent / "cases" / "stationaryDroplet"
_END_TIME = "0.004"
_SIGMA = 0.07
_THICKNESS = 2e-4  # the one-cell z extent of the blockMeshDict
_CENTRE = np.array([0.004, 0.004])

# The droplet's Laplace pressure may fall short of sigma/R by interFoam's -14.7% and some.
_LAPLACE_RTOL = 0.2
# NeoN vs interFoam (measured: 7e-6, 1.5e-3; relL2 x3).
_JUMP_RTOL = 1e-4
_SPURIOUS_RTOL = 1.5e-2
_CEILINGS = {"alpha.water": 8.5e-6, "p_rgh": 2.9e-5, "U": 5.2e-3}


def _run(command: list[str], case: Path, log_name: str) -> None:
    log_path = case / f"log.{log_name}"
    with log_path.open("w") as log:
        completed = subprocess.run(
            command, cwd=case, stdout=log, stderr=subprocess.STDOUT, timeout=1200
        )
    assert completed.returncode == 0, (
        f"{log_name} failed in {case}:\n{log_path.read_text()[-4000:]}"
    )


def _staged(tmp_path: Path) -> Path:
    """The meshed case with the cut initial alpha.water and the cell centres in 0/C."""
    case = tmp_path / "base"
    shutil.copytree(_CASE, case)
    shutil.copytree(case / "0.orig", case / "0")
    for command in (["blockMesh"], ["setAlphaField"], ["postProcess", "-func", "writeCellCentres"]):
        _run(command, case, command[0])
    return case


def _droplet_measures(case: Path, centres: np.ndarray) -> tuple[float, float, float]:
    """(Laplace pressure, sigma/R of the water volume, peak |U|) at the end time."""
    fields = read_internal_fields(case, case / _END_TIME, ["alpha.water", "p_rgh", "U"])
    alpha, p_rgh, velocity = fields["alpha.water"], fields["p_rgh"], fields["U"]
    cell_volume = 8e-3 * 8e-3 * _THICKNESS / alpha.size
    radius = np.sqrt(alpha.sum() * cell_volume / (np.pi * _THICKNESS))
    distance = np.linalg.norm(centres[:, :2] - _CENTRE, axis=1)
    jump = p_rgh[distance < 0.5 * radius].mean() - p_rgh[distance > 1.5 * radius].mean()
    return float(jump), _SIGMA / radius, float(np.linalg.norm(velocity, axis=1).max())


def _rel_l2(case: Path, native: Path) -> dict[str, float]:
    names = list(_CEILINGS)
    ours = read_internal_fields(case, case / _END_TIME, names)
    theirs = read_internal_fields(native, native / _END_TIME, names)
    return {
        name: float(np.linalg.norm(ours[name] - theirs[name]) / np.linalg.norm(theirs[name]))
        for name in names
    }


@pytest.mark.parametrize("executor", ["Serial", "CPU", "GPU"])
def test_stationary_droplet_surface_tension_matches_interFoam(
    tmp_path: Path, executor: str
) -> None:
    base = _staged(tmp_path)
    neon, native = tmp_path / "neon", tmp_path / "interFoam"
    shutil.copytree(base, neon)
    shutil.copytree(base, native)
    control = DictFile(neon / "system" / "controlDict")
    control.set("executor", executor)
    control.write()
    centres = read_internal_fields(base, base / "0", ["C"])["C"]

    _run([sys.executable, "-m", "neofoam.cli.app", "solver", "incompressiblevofneon"], neon, "neon")
    _run(["interFoam"], native, "interFoam")

    jump, laplace, spurious = _droplet_measures(neon, centres)
    native_jump, _, native_spurious = _droplet_measures(native, centres)
    rel_l2 = _rel_l2(neon, native)
    print(
        f"{executor}: jump {jump:.4f} (interFoam {native_jump:.4f}, sigma/R {laplace:.4f}), "
        f"peak |U| {spurious:.4e} (interFoam {native_spurious:.4e}), relL2 {rel_l2}"
    )
    np.testing.assert_allclose(jump, laplace, rtol=_LAPLACE_RTOL, err_msg=f"droplet/{executor}")
    np.testing.assert_allclose(jump, native_jump, rtol=_JUMP_RTOL, err_msg=f"droplet/{executor}")
    np.testing.assert_allclose(
        spurious, native_spurious, rtol=_SPURIOUS_RTOL, err_msg=f"droplet/{executor}"
    )
    exceeded = {name: value for name, value in rel_l2.items() if value > _CEILINGS[name]}
    assert not exceeded, f"droplet/{executor}: relL2 above ceiling {exceeded}"
