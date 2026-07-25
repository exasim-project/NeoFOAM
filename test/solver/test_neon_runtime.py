# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The NeoN/Kokkos finalize handler must not abort at interpreter teardown.

``ensure_neon_initialized`` registers a finalize handler via ``atexit``. Because
``atexit`` runs *before* the interpreter's final garbage collection, calling
``Kokkos::finalize()`` there naively races the teardown of any NeoN ``Vector`` /
``MeshAdapter`` still alive — most often one pinned by a failed solve's traceback
in ``sys.last_*``. Its destructor then deallocates after ``Kokkos::finalize`` and
Kokkos calls ``host_abort`` (SIGABRT), which also clobbers the real init error in
the log. This pins the ordering fix: a process that finishes with a NeoN object
pinned by ``sys.last_traceback`` must still exit cleanly, no ``host_abort``.

The abort only fires at interpreter shutdown, so it cannot be observed in-process
— the test drives a subprocess and asserts on its exit code and stderr.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

pytest.importorskip("neon._neon")  # skip when the NeoN bindings are not built

# A failed solve, reduced to essentials: initialize through the shared guard,
# build a Kokkos-backed NeoN Vector as a local, raise, and record the exception
# in sys.last_* exactly as a top-level traceback print does. The Vector is now
# reachable only through the stored traceback and survives to final GC — the
# precise condition that aborted the process before the finalize ordering fix.
_PINNED_BY_TRACEBACK = """
import sys
from neofoam.solver.neon_runtime import ensure_neon_initialized
import neon._neon as nn

ensure_neon_initialized(["regression"])

def failing_solve():
    field = nn.ScalarVector(nn.SerialExecutor(), 128, 1.0)
    raise RuntimeError("InitStep failed holding " + repr(field))

try:
    failing_solve()
except RuntimeError:
    sys.last_type, sys.last_value, sys.last_traceback = sys.exc_info()

print("solve failed and recorded")
"""


def test_finalize_does_not_abort_with_neon_object_pinned_by_traceback() -> None:
    """A NeoN Vector pinned by sys.last_traceback still finalizes cleanly."""
    result = subprocess.run(
        [sys.executable, "-c", _PINNED_BY_TRACEBACK],
        capture_output=True,
        text=True,
        timeout=120,
    )

    # -6 / 134 is SIGABRT from Kokkos host_abort; a clean run exits 0.
    assert result.returncode == 0, (
        f"neon teardown aborted (returncode={result.returncode})\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "deallocated after Kokkos::finalize" not in result.stderr
    assert "solve failed and recorded" in result.stdout
