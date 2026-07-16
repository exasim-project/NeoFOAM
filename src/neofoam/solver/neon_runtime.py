# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Process-wide NeoN (Kokkos) runtime initialization.

Kokkos may only be initialized/finalized once per process. Every NeoN-backed
solver calls :func:`ensure_neon_initialized` before touching NeoN so repeated
solver runs in one process (e.g. the test suite) reuse a single initialization;
finalization is registered once at interpreter exit.
"""

import atexit

import neon._neon as nn  # NeoN Python bindings

_neon_initialized = False


def ensure_neon_initialized(argv: list[str]) -> None:
    """Initialize NeoN/Kokkos once per process (idempotent)."""
    global _neon_initialized
    if not _neon_initialized:
        nn.initialize(argv)
        _neon_initialized = True
        atexit.register(nn.finalize)
