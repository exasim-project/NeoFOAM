# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The disc of radius 0.15 centred at (0.3, 0.35), sampled 8x8 per cell.

A cell's fraction is the share of its 64 subcell centres inside the disc, on the
32x32 cells of blockMeshDict. The exact cut of ``iso_volume_fractions`` starts
0.91% below the disc's volume and makes OpenFOAM's MULES undershoot the bounds
of ``test_openfoam_backend_keeps_alpha_bounded``.
"""

from typing import Any

import numpy as np

from neofoam.preprocess import IsoVolumeFractions

CELL_SIZE = 1 / 32
N_SUB = 8


def set_alpha(mesh: Any) -> dict[str, IsoVolumeFractions]:
    centres = np.asarray(mesh.C().internalField())[:, :2]
    offsets = CELL_SIZE * ((np.arange(N_SUB) + 0.5) / N_SUB - 0.5)
    subcells = centres[:, None, None, :] + np.stack(np.meshgrid(offsets, offsets), axis=-1)
    inside = np.linalg.norm(subcells - (0.3, 0.35), axis=-1) < 0.15
    return {"alpha.water": IsoVolumeFractions(cells=inside.mean(axis=(1, 2)), patches={})}
