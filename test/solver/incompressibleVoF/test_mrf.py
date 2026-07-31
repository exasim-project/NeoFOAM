# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""MRF (rotating reference frame) zones in the VoF momentum equation.

``interFoam``'s ``UEqn.H`` carries ``+ MRF.DDt(rho, U)`` — the *mass-weighted*
frame acceleration, which is where the VoF hook differs from the single-phase
one (``test/solver/incompressibleFluid/test_mrf.py``, which also pins detection
and the wall-velocity correction). This module pins the weighting: drop the
``rho`` factor and the term is wrong by three orders of magnitude on the water
cells of the case below.

**Case.** ``cases/vofRow4Rotor`` — ``cases/vofRow4`` (four cells across a unit
cube, ``0/U`` a uniform ``(2 0 0)``, ``0/alpha.water`` stepping
``0, 0.25, 0.75, 1``) with its block named ``rotor`` so blockMesh puts every
cell in a cellZone, plus a ``constant/MRFProperties`` spinning that zone at
``omega 2`` about ``(0 0 1)``.

Hand-derived expectation, per cell: ``Omega x U = (0, 4, 0)``, the mixture
density is ``alpha*1000 + (1-alpha)*1`` (``constant/transportProperties``), the
cells have volume ``0.25``, and an ``fvMatrix`` takes an explicit vector field
into its source as ``source -= V*field`` — so the source must shift by
``(0, -rho, 0)``.

Tolerance is ``atol`` only: the shift is a difference of two O(1e3) sources, so
a few ulp (~1e-13) is the error floor, while the failure this guards against is
a factor of ~1000.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).parent
_CASE = _HERE / "cases" / "vofRow4Rotor"
_WORKER = _HERE / "_mrf_worker.py"

#: ``constant/MRFProperties``: ``axis (0 0 1)``, ``omega 2``.
_OMEGA = np.array([0.0, 0.0, 2.0])

#: ``0/U`` internal field.
_U = np.array([2.0, 0.0, 0.0])

#: ``0/alpha.water`` internal field, and the two phase densities from
#: ``constant/transportProperties``.
_ALPHA = np.array([0.0, 0.25, 0.75, 1.0])
_RHO_WATER = 1000.0
_RHO_AIR = 1.0

#: Far above the ulp floor of an O(1e3) source, far below a missing rho factor.
_ATOL = 1e-9


@pytest.fixture(scope="module")
def vof_row4_rotor(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Mesh ``cases/vofRow4Rotor`` and assemble its momentum equation both ways."""
    case = tmp_path_factory.mktemp("vofRow4Rotor") / "case"
    shutil.copytree(_CASE, case)
    subprocess.run(["blockMesh", "-case", str(case)], check=True, capture_output=True, timeout=300)
    subprocess.run(
        [sys.executable, str(_WORKER), str(case)], check=True, capture_output=True, timeout=600
    )
    return json.loads((case / "mrf.json").read_text())


def test_the_case_builds_one_rotating_zone(vof_row4_rotor: dict) -> None:
    assert vof_row4_rotor["zones"] == 1


def test_the_momentum_source_gains_the_mass_weighted_frame_acceleration(
    vof_row4_rotor: dict,
) -> None:
    volumes = np.asarray(vof_row4_rotor["cell_volumes"])[:, None]
    rho = (_ALPHA * _RHO_WATER + (1.0 - _ALPHA) * _RHO_AIR)[:, None]

    # Against the assembly on the *same* corrected boundary state, so the shift
    # is the frame acceleration and nothing else.
    shift = np.asarray(vof_row4_rotor["source_with_mrf"]) - np.asarray(
        vof_row4_rotor["source_without_mrf_corrected_walls"]
    )
    np.testing.assert_allclose(
        shift,
        -volumes * rho * np.cross(_OMEGA, _U),
        rtol=0,
        atol=_ATOL,
        err_msg="vofRow4Rotor: the momentum source does not carry -V*rho*(Omega x U)",
    )
