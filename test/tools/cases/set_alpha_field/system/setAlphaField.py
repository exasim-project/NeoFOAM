# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""A cylinder along z that crosses the wall y = 0; see system/setAlphaFieldDict."""

from typing import Any

import numpy as np

from neofoam.preprocess import IsoVolumeFractions, iso_volume_fractions


def set_alpha(mesh: Any) -> dict[str, IsoVolumeFractions]:
    points = np.asarray(mesh.points())
    level_set = 0.3 - np.linalg.norm(points[:, :2] - (0.5, 0.1), axis=1)
    return {"alpha.water": iso_volume_fractions(mesh, level_set)}
