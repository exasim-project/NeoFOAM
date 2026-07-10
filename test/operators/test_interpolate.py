# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Cross-backend parity of linear surface interpolation (pybFoam fvc vs neon)."""

from __future__ import annotations

import numpy as np
import pytest
from backends import Field, nb, pyb
from conftest import EXECUTORS, MESH_NAMES


@pytest.mark.parametrize("executor", EXECUTORS)
@pytest.mark.parametrize("mesh", MESH_NAMES)
def test_interpolate_T(mesh: str, executor: str, T: Field) -> None:
    pyb_res = pyb.fvc.interpolate(T)
    nb_res = nb.interpolate(T)

    # neon appends boundary-face values after the internal faces
    nb_res = nb_res[: pyb_res.shape[0]]
    rtol = 1e-12 if executor == "Serial" else 1e-8
    np.testing.assert_allclose(
        nb_res, pyb_res, rtol=rtol, atol=1e-14 * np.abs(pyb_res).max()
    )
