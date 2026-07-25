# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Process-wide NeoN (Kokkos) runtime initialization.

Kokkos may only be initialized/finalized once per process. Every NeoN-backed
solver calls :func:`ensure_neon_initialized` before touching NeoN so repeated
solver runs in one process (e.g. the test suite) reuse a single initialization;
finalization is registered once at interpreter exit.
"""

import atexit
import gc
import sys

import neon._neon as nn  # NeoN Python bindings

_neon_initialized = False


def _finalize_neon() -> None:
    """Finalize Kokkos, but only after Python-owned NeoN objects are gone.

    ``atexit`` handlers run *before* the interpreter's final garbage collection,
    so calling ``nn.finalize()`` (``Kokkos::finalize()``) directly here races the
    teardown of any NeoN ``Vector`` / ``MeshAdapter`` still reachable at exit.
    The usual culprit is a failed solve: the exception it raised is kept in
    ``sys.last_*`` (and, on 3.11+, ``sys.last_exc``), whose traceback pins the
    frames — and thus the partially-built runtime and fields — alive. Those
    destructors then deallocate *after* ``Kokkos::finalize`` and Kokkos aborts the
    process (``host_abort``), which also clobbers the real init error in the log.

    Dropping the stored exception and forcing a collection destroys those objects
    first, so ``finalize`` (and its ginkgo executor-release hook) runs cleanly.
    """
    for _attr in ("last_traceback", "last_value", "last_type", "last_exc"):
        if hasattr(sys, _attr):
            setattr(sys, _attr, None)
    gc.collect()
    nn.finalize()


def ensure_neon_initialized(argv: list[str]) -> None:
    """Initialize NeoN/Kokkos once per process (idempotent)."""
    global _neon_initialized
    if not _neon_initialized:
        nn.initialize(argv)
        _neon_initialized = True
        atexit.register(_finalize_neon)
