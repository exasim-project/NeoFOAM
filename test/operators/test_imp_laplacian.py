# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Cross-backend parity of the implicit laplacian operators.

Both backends assemble the implicit operator and apply the matrix to the
current field: pybFoam builds the ``fvm.laplacian`` matrix and returns
``M & psi``; neon assembles the linear system of ``nn.imp.laplacian`` and
returns ``(A·psi - b) / V``. In exact arithmetic both equal the explicit
laplacian.
"""

from __future__ import annotations

import numpy as np
import pytest
from backends import Field, nb, pyb
from conftest import EXECUTORS, MESH_NAMES, TWO_D_MESHES


@pytest.mark.parametrize("executor", EXECUTORS)
@pytest.mark.parametrize("mesh", MESH_NAMES)
def test_imp_laplacian_Gamma_T(
    mesh: str, executor: str, Gamma: Field, T: Field
) -> None:
    pyb_res = pyb.fvm.laplacian(Gamma, T)
    nb_res = nb.imp.laplacian(Gamma, T)

    rtol = 1e-9 if executor == "Serial" else 1e-8
    np.testing.assert_allclose(
        nb_res, pyb_res, rtol=rtol, atol=1e-10 * np.abs(pyb_res).max()
    )


@pytest.mark.parametrize("executor", EXECUTORS)
@pytest.mark.parametrize("mesh", MESH_NAMES)
def test_imp_laplacian_Gamma_U(
    mesh: str, executor: str, Gamma: Field, U: Field
) -> None:
    pyb_res = pyb.fvm.laplacian(Gamma, U)
    nb_res = nb.imp.laplacian(Gamma, U)

    rtol = 1e-9 if executor == "Serial" else 1e-8
    scale = np.abs(pyb_res).max()
    if mesh in TWO_D_MESHES:
        # OpenFOAM's matrix apply zeroes the empty-direction z on 2D meshes
        np.testing.assert_allclose(
            nb_res[:, :2], pyb_res[:, :2], rtol=rtol, atol=1e-10 * scale
        )
    else:
        np.testing.assert_allclose(nb_res, pyb_res, rtol=rtol, atol=1e-10 * scale)
