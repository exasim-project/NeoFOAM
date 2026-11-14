# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Advect with the openfoam backend in the velocity ``U0 cos(2 pi t/period)``.

The oracle for the ``isoAdvector { period }`` flux: the same flow, but as a
prescribed velocity function, which the backend evaluates at the middle of each
step and turns into ``phi = fvc::flux(U)``.

Usage: ``python _prescribed_velocity_worker.py <case-dir> <period> <Ux> <Uy> <Uz> <out.npz>``
"""

import math
import sys
from pathlib import Path

import numpy as np

from neofoam.algorithms.vof import advect


def _main(case: Path, period: float, u0: np.ndarray, out: Path) -> None:
    def velocity(cell_centres: np.ndarray, time: float) -> np.ndarray:
        return np.tile(u0 * math.cos(2.0 * math.pi * time / period), (len(cell_centres), 1))

    result = advect(case, "openfoam", velocity=velocity)
    np.savez(out, times=result.times, alpha=result.alpha)


if __name__ == "__main__":
    _main(
        Path(sys.argv[1]),
        float(sys.argv[2]),
        np.array([float(v) for v in sys.argv[3:6]]),
        Path(sys.argv[6]),
    )
