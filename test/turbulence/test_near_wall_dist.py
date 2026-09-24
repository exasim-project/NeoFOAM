# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The NeoN kEpsilon closure builds on a case whose ``fvSchemes`` has no ``wallDist``.

OpenFOAM's kEpsilon needs no ``wallDist { method … }`` entry — its wall functions
read ``nearWallDist``, which is pure patch geometry — so the tutorials do not ship
one. The NeoN closure used to construct a ``Foam::wallDist`` instead, which reads
that entry and killed a stock case with an uncatchable ``FOAM FATAL IO ERROR``
during initialisation; it now goes through pybFoam's ``nearWallDist``.

The case (:func:`_parity_case.walled_case`, which drops the shared base's ``wallDist``
block) is a unit cube of 4 x 4 x 4 uniform cells whose two z patches are walls, so
every wall face's owner-cell centre sits exactly half a cell — 0.5/4 = 0.125 m — from
its wall, and non-wall patches hold 0 (OpenFOAM's ``nearWallDist`` zeroes them). Those
are the expected values below.

The model is built in a subprocess (the ``near_wall_dist`` role of
:mod:`_parity_worker`): one ``Foam::Time`` per process. A missing ``wallDist``
entry aborts the process rather than raising, so the regression shows up as a
failed subprocess.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from turbulence._parity_case import run_worker, walled_case

#: Half the 0.25 m cell height — the owner-cell centre distance to its wall patch.
NEAR_WALL_DISTANCE = 0.125

#: Wall patches of the walled case (the two z faces) and the generic patches.
WALL_PATCHES = ("zMin", "zMax")
NON_WALL_PATCHES = ("xMin", "xMax", "yMin", "yMax")


def test_kepsilon_near_wall_dist_without_fvschemes_walldist_entry(tmp_path: Path) -> None:
    """kEpsilon initialises without an ``fvSchemes`` ``wallDist`` block, with sane ``y``."""
    case = walled_case().build_at(tmp_path / "case").path
    # Guard the case: the point of it is the absent scheme block.
    assert "wallDist\n{" not in (case / "system" / "fvSchemes").read_text()

    run_worker("mesh", case)
    run_worker("near_wall_dist", case)  # builds the closure: no wallDist entry needed

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
