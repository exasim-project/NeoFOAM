# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Cross-backend parity of the implicit convection operators.

Both backends assemble the implicit operator and apply the matrix to the
current field: pybFoam builds the ``fvm.div`` matrix and returns ``M & psi``;
neon assembles the linear system of ``nn.imp.div`` and returns
``(A·psi - b) / V``. In exact arithmetic both equal the explicit divergence.
"""

from __future__ import annotations

import numpy as np
import pytest
from backends import Field, nb, pyb
from conftest import EXECUTORS, MESH_NAMES


@pytest.mark.parametrize("executor", EXECUTORS)
@pytest.mark.parametrize("mesh", MESH_NAMES)
@pytest.mark.parametrize("scheme", ["linear", "upwind", "linearUpwind"])
def test_imp_div_phi_T(
    mesh: str, scheme: str, executor: str, T: Field, phi: Field
) -> None:
    pyb_res = pyb.fvm.div(phi, T, scheme=scheme)
    nb_res = nb.imp.div(phi, T, scheme=scheme)

    rtol = 1e-9 if executor == "Serial" else 1e-8
    np.testing.assert_allclose(
        nb_res, pyb_res, rtol=rtol, atol=1e-12 * np.abs(pyb_res).max()
    )


@pytest.mark.parametrize("executor", EXECUTORS)
@pytest.mark.parametrize("mesh", MESH_NAMES)
@pytest.mark.parametrize("scheme", ["linear", "upwind", "linearUpwind"])
def test_imp_div_phi_U(
    mesh: str, scheme: str, executor: str, U: Field, phi: Field
) -> None:
    pyb_res = pyb.fvm.div(phi, U, scheme=scheme)
    nb_res = nb.imp.div(phi, U, scheme=scheme)

    rtol = 1e-9 if executor == "Serial" else 1e-8
    np.testing.assert_allclose(
        nb_res, pyb_res, rtol=rtol, atol=1e-12 * np.abs(pyb_res).max()
    )
