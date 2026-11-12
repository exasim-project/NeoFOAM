# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Fresh-interpreter worker for ``test_kernels.py``: one kernel on host numpy.

Run as ``python _kernels_worker.py <kernel> <n_elements> <out.npy>``. Nothing in
this process initializes NeoN (Kokkos) before the kernel is looked up — the
state an OpenFOAM solver such as ``incompressibleFluid`` leaves it in — and the
input is all ones, so the parent knows the answer without running the kernel.
"""

from __future__ import annotations

import sys

import numpy as np

from neofoam.postprocess import _kernels as pp


def main(kernel: str, n_elements: int, out: str) -> None:
    if kernel == "mag":
        result = pp.mag(np.ones((n_elements, 3)))
    else:
        result = getattr(pp, kernel)(np.ones(n_elements), 1, None, None)
    np.save(out, np.asarray(result, dtype=float))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3])
