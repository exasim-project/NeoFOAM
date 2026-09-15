# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Spec for ``neofoam.foam.libraries.load_libraries``.

``libwaveModels.so`` ships with OpenFOAM itself (not with ``pybFoam``), so its
presence on the dynamic-loader search path depends on the environment's
``LD_LIBRARY_PATH`` rather than on the ``pybFoam``/OpenFOAM hard dependency the
rest of the suite assumes — hence the explicit availability check here rather
than a bare import.
"""

import ctypes

import pytest

from neofoam.foam.libraries import load_libraries

_WAVE_MODELS_LIB = "libwaveModels.so"


def _wave_models_available() -> bool:
    try:
        ctypes.CDLL(_WAVE_MODELS_LIB)
    except OSError:
        return False
    return True


@pytest.mark.skipif(
    not _wave_models_available(),
    reason=f"{_WAVE_MODELS_LIB} not found on the dynamic-loader search path",
)
def test_load_libraries_loads_libwavemodels() -> None:
    load_libraries([_WAVE_MODELS_LIB])


def test_load_libraries_raises_for_missing_library() -> None:
    with pytest.raises(OSError, match="does-not-exist-neofoam.so"):
        load_libraries(["does-not-exist-neofoam.so"])
