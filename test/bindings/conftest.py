# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Initialize NeoN for the tests that build NeoN vectors in-process."""

import pytest


@pytest.fixture(scope="session", autouse=True)
def _neon_runtime() -> None:
    """Bring up NeoN/Kokkos once; since Kokkos 5.1 a NeoN vector aborts without it."""
    try:
        from neofoam.solver.neon_runtime import ensure_neon_initialized  # noqa: PLC0415
    except ImportError:  # NeoN bindings not built: those tests skip themselves
        return
    ensure_neon_initialized(["pytest"])
