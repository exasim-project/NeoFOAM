# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The post-processing kernels run on host numpy without NeoN initialized first.

**Why a subprocess.** Kokkos is initialized once per process and never undone,
so a kernel call in the pytest process would see whatever another test left
behind. Each case runs ``_kernels_worker.py`` in a fresh interpreter instead —
the state an OpenFOAM solver (``incompressibleFluid``) leaves Kokkos in.

**Why these sizes.** The numpy front doors copy their input into a
``NeoN::Vector`` on the ``SerialExecutor``. Kokkos does a host copy below 80 KiB
with ``memcpy`` and from there with a ``parallel_for``, which aborts before
``Kokkos::initialize``. So 20000 scalars (160 KB) and 5000 vectors (120 KB) are
above that limit, while every other postprocess test stays below it.

**Expectations.** The input is all ones: min/max are 1, the sum is the element
count, and the magnitude of (1, 1, 1) is sqrt(3). The kernels add exact ones
here, so the tolerance only covers the last bit of the sqrt.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_WORKER = Path(__file__).parent / "_kernels_worker.py"


@pytest.mark.parametrize(
    "kernel,n_elements,expected",
    [
        pytest.param("min", 20000, np.array([1.0]), id="min-scalars"),
        pytest.param("max", 20000, np.array([1.0]), id="max-scalars"),
        pytest.param("sum", 20000, np.array([20000.0]), id="sum-scalars"),
        pytest.param("mag", 5000, np.full(5000, np.sqrt(3.0)), id="mag-vectors"),
    ],
)
def test_a_kernel_on_a_large_host_array_runs_without_neon_initialized_first(
    tmp_path: Path, kernel: str, n_elements: int, expected: "np.ndarray"
) -> None:
    out = tmp_path / "result.npy"

    run = subprocess.run(
        [sys.executable, str(_WORKER), kernel, str(n_elements), str(out)],
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert run.returncode == 0, f"{kernel} aborted:\n{run.stderr[-2000:]}"
    np.testing.assert_allclose(
        np.load(out), expected, rtol=1e-15, atol=0.0, err_msg=f"{kernel} on {n_elements} ones"
    )
