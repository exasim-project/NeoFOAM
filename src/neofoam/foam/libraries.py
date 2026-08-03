# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Load extra OpenFOAM shared libraries that ``pybFoam`` does not link.

A native solver binary (e.g. ``interFoam``) links libraries beyond pybFoam's
dependency set — ``libwaveModels.so`` and the like — and no tutorial declares them
under ``libs (...)`` because, for OpenFOAM, the binary *is* the declaration. Cases
relying on their runtime-selection entries need them loaded here instead.
"""

import ctypes


def load_libraries(names: list[str]) -> None:
    """Load OpenFOAM shared libraries by name, registering their runtime-selection tables.

    ``RTLD_GLOBAL``, like OpenFOAM's own ``libs (...)``, so the static registrations
    become visible process-wide.
    """
    for name in names:
        try:
            ctypes.CDLL(name, mode=ctypes.RTLD_GLOBAL)
        except OSError as exc:
            raise OSError(f"failed to load OpenFOAM library {name!r}: {exc}") from exc
