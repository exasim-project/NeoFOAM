# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Cross-backend parity of the explicit laplacian operators (pybFoam fvc vs neon)."""

from __future__ import annotations

import numpy as np
import pytest
from backends import Field, nb, pyb
from conftest import EXECUTORS, MESH_NAMES


@pytest.mark.parametrize("executor", EXECUTORS)
@pytest.mark.parametrize("mesh", MESH_NAMES)
def test_laplacian_Gamma_T(mesh: str, executor: str, Gamma: Field, T: Field) -> None:
    pyb_res = pyb.fvc.laplacian(Gamma, T)
    nb_res = nb.exp.laplacian(Gamma, T)

    rtol = 1e-9 if executor == "Serial" else 1e-8
    np.testing.assert_allclose(
        nb_res, pyb_res, rtol=rtol, atol=1e-10 * np.abs(pyb_res).max()
    )


@pytest.mark.parametrize("executor", EXECUTORS)
@pytest.mark.parametrize("mesh", MESH_NAMES)
def test_laplacian_Gamma_U(mesh: str, executor: str, Gamma: Field, U: Field) -> None:
    pyb_res = pyb.fvc.laplacian(Gamma, U)
    nb_res = nb.exp.laplacian(Gamma, U)

    rtol = 1e-9 if executor == "Serial" else 1e-8
    np.testing.assert_allclose(
        nb_res, pyb_res, rtol=rtol, atol=1e-10 * np.abs(pyb_res).max()
    )
