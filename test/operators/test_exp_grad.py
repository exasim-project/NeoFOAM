# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Cross-backend parity of the explicit gradient (pybFoam fvc vs neon)."""

from __future__ import annotations

import numpy as np
import pytest
from backends import nb, pyb
from case_setup import simulation
from conftest import EXECUTORS, MESH_NAMES


@pytest.mark.parametrize("executor", EXECUTORS)
@pytest.mark.parametrize("mesh", MESH_NAMES)
def test_grad_T(mesh: str, executor: str) -> None:
    sim = simulation(mesh, executor)
    x, y, z = sim.mesh.cell_centres.T

    T = sim.field("T")
    T[:] = 2.0 + np.sin(np.pi * x) * np.cos(np.pi * y) + 0.3 * z

    pyb_res = pyb.fvc.grad(T)
    nb_res = nb.exp.grad(T)

    rtol = 1e-12 if executor == "Serial" else 1e-8
    np.testing.assert_allclose(
        nb_res, pyb_res, rtol=rtol, atol=1e-12 * np.abs(pyb_res).max()
    )
