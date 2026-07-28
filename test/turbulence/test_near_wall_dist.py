# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The NeoN kEpsilon closure builds on a case whose ``fvSchemes`` has no ``wallDist``.

OpenFOAM's kEpsilon needs no ``wallDist { method … }`` entry — its wall functions
read ``nearWallDist``, which is pure patch geometry — so the tutorials do not ship
one. The NeoN closure used to construct a ``Foam::wallDist`` instead, which reads
that entry and killed a stock case with an uncatchable ``FOAM FATAL IO ERROR``
during initialisation; it now goes through pybFoam's ``nearWallDist``.

The case (``walled_base``) is a unit cube of 4 x 4 x 4 uniform cells whose two z
patches are walls, so every wall face's owner-cell centre sits exactly half a cell
— 0.5/4 = 0.125 m — from its wall, and non-wall patches hold 0 (OpenFOAM's
``nearWallDist`` zeroes them). Those are the expected values below.

The model is built in a subprocess (see :mod:`_near_wall_dist_worker`): one
``Foam::Time`` per process. A missing ``wallDist`` entry aborts the process rather
than raising, so the regression shows up as a failed subprocess.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).parent
_WORKER = _HERE / "_near_wall_dist_worker.py"
_CASE = _HERE / "walled_base"  # kEpsilon box with walls and no fvSchemes/wallDist

#: Half the 0.25 m cell height — the owner-cell centre distance to its wall patch.
NEAR_WALL_DISTANCE = 0.125

#: Wall patches of ``walled_base`` (the two z faces) and the generic patches.
WALL_PATCHES = ("zMin", "zMax")
NON_WALL_PATCHES = ("xMin", "xMax", "yMin", "yMax")


def _run_worker(role: str, case: Path) -> None:
    """Run one worker role in a fresh process (one ``Foam::Time`` per process)."""
    subprocess.run(
        [sys.executable, str(_WORKER), role, str(case)],
        check=True,
        cwd=str(case.parent),
    )


def test_kepsilon_near_wall_dist_without_fvschemes_walldist_entry(tmp_path: Path) -> None:
    """kEpsilon initialises without an ``fvSchemes`` ``wallDist`` block, with sane ``y``."""
    case = tmp_path / "case"
    shutil.copytree(_CASE, case)
    # Guard the fixture: the point of the case is the absent scheme block.
    assert "wallDist\n{" not in (case / "system" / "fvSchemes").read_text()

    _run_worker("mesh", case)
    _run_worker("near_wall_dist", case)  # builds the closure: no wallDist entry needed

    near_wall_dist = np.load(case / "near_wall_dist.npz")
    for patch in WALL_PATCHES:
        np.testing.assert_allclose(
            near_wall_dist[patch],
            NEAR_WALL_DISTANCE,
            rtol=1e-12,
            err_msg=f"walled_base: wall patch {patch} near-wall distance",
        )
    for patch in NON_WALL_PATCHES:
        np.testing.assert_allclose(
            near_wall_dist[patch],
            0.0,
            atol=0.0,
            err_msg=f"walled_base: non-wall patch {patch} must hold 0",
        )
