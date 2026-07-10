# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Cross-backend parity of the explicit laplacian operators (pybFoam fvc vs neon)."""

from __future__ import annotations

import numpy as np
import pytest
from backends import nb, pyb
from case_setup import simulation
from conftest import EXECUTORS, MESH_NAMES


@pytest.mark.parametrize("executor", EXECUTORS)
@pytest.mark.parametrize("mesh", MESH_NAMES)
def test_laplacian_Gamma_T(mesh: str, executor: str) -> None:
    sim = simulation(mesh, executor)
    x, y, z = sim.mesh.cell_centres.T

    T = sim.field("T")
    T[:] = 2.0 + np.sin(np.pi * x) * np.cos(np.pi * y) + 0.3 * z
    Gamma = sim.field("Gamma")  # uniform 1 from the staged 0/Gamma

    pyb_res = pyb.fvc.laplacian(Gamma, T)
    nb_res = nb.exp.laplacian(Gamma, T)

    rtol = 1e-9 if executor == "Serial" else 1e-8
    np.testing.assert_allclose(
        nb_res, pyb_res, rtol=rtol, atol=1e-10 * np.abs(pyb_res).max()
    )


@pytest.mark.parametrize("executor", EXECUTORS)
@pytest.mark.parametrize("mesh", MESH_NAMES)
def test_laplacian_Gamma_U(mesh: str, executor: str) -> None:
    sim = simulation(mesh, executor)
    x, y, z = sim.mesh.cell_centres.T

    U = sim.field("U")
    u = np.asarray(U)
    u[:, 0] = 1.0 + np.sin(np.pi * y) + 0.5 * np.sin(np.pi * x)
    u[:, 1] = 0.5 + np.cos(np.pi * x) + 0.5 * np.cos(np.pi * y)
    u[:, 2] = 0.1 + 0.2 * np.sin(np.pi * z)
    U[:] = u
    Gamma = sim.field("Gamma")  # uniform 1 from the staged 0/Gamma

    pyb_res = pyb.fvc.laplacian(Gamma, U)
    nb_res = nb.exp.laplacian(Gamma, U)

    rtol = 1e-9 if executor == "Serial" else 1e-8
    np.testing.assert_allclose(
        nb_res, pyb_res, rtol=rtol, atol=1e-10 * np.abs(pyb_res).max()
    )
