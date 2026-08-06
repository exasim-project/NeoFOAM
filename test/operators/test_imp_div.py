# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Cross-backend parity of the implicit convection operators.

Both backends assemble the implicit operator and apply the matrix to the
current field: pybFoam builds the ``fvm.div`` matrix and returns ``M & psi``;
neon assembles the linear system of ``nn.imp.div`` and returns
``(A·psi - b) / V``. In exact arithmetic both equal the explicit divergence.

The ``scheme=`` tests hand the scheme to each backend as a string/TokenList.
The last test instead lets both backends select ``div(phi,U)`` from the case's
own ``system/fvSchemes`` — on the neon side through the dictionary
``map_fv_schemes`` returned — which is the only path that proves the
``bounded`` convection prefix survives that mapping.
"""

from __future__ import annotations

import numpy as np
import pytest
from backends import nb, pyb
from case_setup import flux, simulation
from conftest import EXECUTORS, MESH_NAMES


@pytest.mark.parametrize("executor", EXECUTORS)
@pytest.mark.parametrize("mesh", MESH_NAMES)
@pytest.mark.parametrize("scheme", ["linear", "upwind", "linearUpwind", "boundedUpwind"])
def test_imp_div_phi_T(mesh: str, scheme: str, executor: str) -> None:
    sim = simulation(mesh, executor)
    x, y, z = sim.mesh.cell_centres.T

    T = sim.field("T")
    T[:] = 2.0 + np.sin(np.pi * x) * np.cos(np.pi * y) + 0.3 * z

    U = sim.field("U")
    u = np.asarray(U)
    u[:, 0] = 1.0 + np.sin(np.pi * y) + 0.5 * np.sin(np.pi * x)
    u[:, 1] = 0.5 + np.cos(np.pi * x) + 0.5 * np.cos(np.pi * y)
    u[:, 2] = 0.1 + 0.2 * np.sin(np.pi * z)
    U[:] = u
    phi = flux(U)

    pyb_res = pyb.fvm.div(phi, T, scheme=scheme)
    nb_res = nb.imp.div(phi, T, scheme=scheme)

    rtol = 1e-9 if executor == "Serial" else 1e-8
    np.testing.assert_allclose(nb_res, pyb_res, rtol=rtol, atol=1e-12 * np.abs(pyb_res).max())


@pytest.mark.parametrize("executor", EXECUTORS)
@pytest.mark.parametrize("mesh", MESH_NAMES)
@pytest.mark.parametrize("scheme", ["linear", "upwind", "linearUpwind", "boundedUpwind"])
def test_imp_div_phi_U(mesh: str, scheme: str, executor: str) -> None:
    sim = simulation(mesh, executor)
    x, y, z = sim.mesh.cell_centres.T

    U = sim.field("U")
    u = np.asarray(U)
    u[:, 0] = 1.0 + np.sin(np.pi * y) + 0.5 * np.sin(np.pi * x)
    u[:, 1] = 0.5 + np.cos(np.pi * x) + 0.5 * np.cos(np.pi * y)
    u[:, 2] = 0.1 + 0.2 * np.sin(np.pi * z)
    U[:] = u
    phi = flux(U)

    pyb_res = pyb.fvm.div(phi, U, scheme=scheme)
    nb_res = nb.imp.div(phi, U, scheme=scheme)

    rtol = 1e-9 if executor == "Serial" else 1e-8
    np.testing.assert_allclose(nb_res, pyb_res, rtol=rtol, atol=1e-12 * np.abs(pyb_res).max())


@pytest.mark.parametrize("executor", EXECUTORS)
@pytest.mark.parametrize("mesh", MESH_NAMES)
def test_imp_div_bounded_prefix_selected_from_fv_schemes(mesh: str, executor: str) -> None:
    sim = simulation(mesh, executor)
    x, y, z = sim.mesh.cell_centres.T

    U = sim.field("U")
    u = np.asarray(U)
    u[:, 0] = 1.0 + np.sin(np.pi * y) + 0.5 * np.sin(np.pi * x)
    u[:, 1] = 0.5 + np.cos(np.pi * x) + 0.5 * np.cos(np.pi * y)
    u[:, 2] = 0.1 + 0.2 * np.sin(np.pi * z)
    U[:] = u
    phi = flux(U)

    # the oracle: OpenFOAM's own bounded wrapper, scheme passed as a string
    reference = pyb.fvm.div(phi, U, scheme="boundedUpwind")
    np.testing.assert_array_equal(
        pyb.fvm.div(phi, U, scheme=None),
        reference,
        err_msg=f"{mesh}: div(phi,U) in system/fvSchemes is not `bounded Gauss upwind`",
    )

    nb_res = nb.imp.div(phi, U, scheme=None)

    rtol = 1e-9 if executor == "Serial" else 1e-8
    np.testing.assert_allclose(
        nb_res,
        reference,
        rtol=rtol,
        atol=1e-12 * np.abs(reference).max(),
        err_msg=f"{mesh}/{executor}: map_fv_schemes lost the bounded div prefix",
    )
