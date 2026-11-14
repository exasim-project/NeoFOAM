# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Entry point of ``advect_in_subprocess``: one ``advect`` call, its result saved as ``.npz``."""

import sys
from pathlib import Path

import numpy as np

from .advection import advect


def _main(case: str, backend: str, executor: str, write: str, out: str) -> None:
    if backend not in ("openfoam", "neon"):
        raise ValueError(f"unknown backend {backend!r}")
    result = advect(Path(case), backend, executor, write == "1")  # type: ignore[arg-type]
    np.savez(out, times=result.times, alpha=result.alpha, cell_volumes=result.cell_volumes)


if __name__ == "__main__":
    _main(*sys.argv[1:])
