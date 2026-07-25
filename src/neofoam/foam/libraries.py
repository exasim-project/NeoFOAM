# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Load extra OpenFOAM shared libraries that ``pybFoam`` does not link.

A native OpenFOAM solver binary (e.g. ``interFoam``) links libraries beyond
``pybFoam``'s own dependency set — for example ``libwaveModels.so``, which
registers the ``waveVelocity``/``waveAlpha`` boundary-condition runtime-selection
table entries. No tutorial declares those in a ``system/controlDict`` ``libs
(...)`` entry because, for OpenFOAM, the *binary itself* is the declaration. The
``neofoam`` entry point links only what ``pybFoam`` links, so a case that relies
on one of those entries fails with ``Unknown patchField type ...`` unless the
library is loaded explicitly first.
"""

import ctypes


def load_libraries(names: list[str]) -> None:
    """Load OpenFOAM shared libraries by name, registering their runtime-selection tables.

    Mirrors OpenFOAM's own ``libs (...)`` mechanism: each library is opened with
    ``RTLD_GLOBAL`` so its static registrations become visible process-wide.
    Raises ``OSError`` naming the library that failed to load.
    """
    for name in names:
        try:
            ctypes.CDLL(name, mode=ctypes.RTLD_GLOBAL)
        except OSError as exc:
            raise OSError(f"failed to load OpenFOAM library {name!r}: {exc}") from exc
