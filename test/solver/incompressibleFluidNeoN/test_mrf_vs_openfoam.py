# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""``nfb.MRFNeoN`` against ``Foam::IOMRFZoneList``, operation by operation.

The NeoN rotating-frame handle does not re-implement ``MRFZone``: it probes the
OpenFOAM zone list once and carries the results as constant NeoN fields
(``src/bindings/mrf.cpp``). This is the check that the probing is faithful — the
end-to-end case cannot serve as one, because a wrong Coriolis sign or a
misassigned zone face still produces a plausible-looking flow.

``cases/mrfBox`` is a two-block box whose first block is the ``rotor`` cellZone,
with ``yMin`` under ``nonRotatingPatches``, so every face class ``MRFZone``
distinguishes is exercised: zone-internal faces, *included* patch faces (which
makeRelative assigns to zero and correctBoundaryVelocity overwrites) and
*excluded* patch faces (which are only shifted by the frame flux).

Both backends act on the velocity read from ``0/U`` and on its flux. That field
is uniform, so ``flux(U)`` is ``U & Sf`` exactly on both sides (linear
interpolation of a uniform field is that field) and the two fluxes entering the
comparison are the same input rather than two approximations of one.

``mrf_worker.py`` explains why each backend needs its own process.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

_HERE = Path(__file__).parent
_CASE = _HERE / "cases" / "mrfBox"
_WORKER = _HERE / "mrf_worker.py"
_RESULT_PREFIX = "#RESULT "

#: The mesh's patches in polyMesh order — the order both backends flatten their
#: boundary values in. Written out rather than queried so the comparison cannot
#: agree by asking one backend how to read the other.
PATCHES = ("xMin", "xMax", "yMin", "yMax", "zMin", "zMax")

#: Bit-identity is not expected: the two backends reduce the same arithmetic in
#: their own order (Foam ``Field`` loops vs NeoN kernels over the transferred
#: values). Every quantity here is a difference or product of O(1) values, so a
#: faithful transfer agrees to a few ULP.
RTOL = 1e-12
ATOL = 1e-14

QUANTITIES = (
    "acceleration",
    "zero_filter",
    "zero_filter_boundary",
    "relative_flux",
    "relative_flux_boundary",
    # The Python-composed form of makeRelative that the NeoN MRF model contributes:
    # keep * (phi - frameFlux), compared against native's own makeRelative.
    "composed_relative_flux",
    "composed_relative_flux_boundary",
    "boundary_velocity",
)

#: Omega x U on a rotor cell: omega = (0 0 5) about z, U = (1 2 3) — the one
#: quantity with a closed form, pinned so a sign or component swap that both
#: backends could never disagree on still fails.
ROTOR_ACCELERATION = (-10.0, 5.0, 0.0)


def _run(case: Path, backend: str) -> dict[str, np.ndarray]:
    env = {**os.environ, "FOAM_SIGFPE": "false"}
    result = subprocess.run(
        [sys.executable, str(_WORKER), str(case), backend, *PATCHES],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, (
        f"mrf_worker {backend} failed (rc={result.returncode}):\n"
        f"{result.stdout[-2000:]}\n{result.stderr[-3000:]}"
    )
    answers = [line for line in result.stdout.splitlines() if line.startswith(_RESULT_PREFIX)]
    assert answers, f"mrf_worker {backend} printed no result:\n{result.stdout[-2000:]}"
    return {
        name: np.asarray(values)
        for name, values in json.loads(answers[-1][len(_RESULT_PREFIX) :]).items()
    }


@pytest.fixture(scope="module")
def mrf(tmp_path_factory: pytest.TempPathFactory) -> dict[str, dict[str, np.ndarray]]:
    """``{backend: {quantity: array}}`` for one meshed copy of ``cases/mrfBox``."""
    case = tmp_path_factory.mktemp("mrfBox") / "case"
    shutil.copytree(_CASE, case)
    meshed = subprocess.run(
        ["blockMesh", "-case", str(case)], capture_output=True, text=True, timeout=120
    )
    assert meshed.returncode == 0, f"blockMesh failed: {meshed.stderr}"

    return {backend: _run(case, backend) for backend in ("openfoam", "neon")}


@pytest.mark.parametrize("quantity", QUANTITIES)
def test_neon_mrf_matches_openfoam(quantity: str, mrf: dict[str, Any]) -> None:
    np.testing.assert_allclose(
        mrf["neon"][quantity],
        mrf["openfoam"][quantity],
        rtol=RTOL,
        atol=ATOL,
        err_msg=f"mrfBox: MRF {quantity}",
    )


def test_frame_acceleration_is_the_rotor_zone_coriolis_term(mrf: dict[str, Any]) -> None:
    """Half the cells rotate and carry ``Omega x U``; the other half carry nothing.

    Guards the comparison above against the degenerate agreement in which both
    backends produce zero everywhere — what a zone lookup that silently found no
    cells would look like — and pins the term's sign and component mixing, which
    a shared convention error would hide.
    """
    acceleration = mrf["neon"]["acceleration"]
    rotating = np.any(acceleration != 0.0, axis=1)

    assert np.count_nonzero(rotating) == len(acceleration) // 2
    rotor = acceleration[rotating]
    np.testing.assert_allclose(
        rotor,
        np.broadcast_to(ROTOR_ACCELERATION, rotor.shape),
        rtol=RTOL,
        err_msg="mrfBox: rotor Coriolis",
    )
