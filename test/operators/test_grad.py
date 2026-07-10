# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Cross-backend parity of the explicit gradient (pybFoam fvc vs neon)."""

from __future__ import annotations

import numpy as np
import pytest
from backends import Field, nb, pyb
from conftest import EXECUTORS, MESH_NAMES, TWO_D_MESHES


@pytest.mark.parametrize("executor", EXECUTORS)
@pytest.mark.parametrize("mesh", MESH_NAMES)
def test_grad_T(mesh: str, executor: str, T: Field) -> None:
    pyb_res = pyb.fvc.grad(T)
    nb_res = nb.exp.grad(T)

    rtol = 1e-12 if executor == "Serial" else 1e-8
    scale = np.abs(pyb_res).max()
    if mesh in TWO_D_MESHES:
        # z is only defined up to the empty-patch treatment on 2D meshes
        np.testing.assert_allclose(
            nb_res[:, :2], pyb_res[:, :2], rtol=rtol, atol=1e-12 * scale
        )
        np.testing.assert_allclose(nb_res[:, 2], pyb_res[:, 2], atol=1e-4 * scale)
    else:
        np.testing.assert_allclose(nb_res, pyb_res, rtol=rtol, atol=1e-12 * scale)
