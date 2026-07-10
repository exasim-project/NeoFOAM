# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Cross-backend parity of the explicit divergence operators (pybFoam fvc vs neon).

Convection runs under every div scheme; the scheme is passed per call.
"""

from __future__ import annotations

import numpy as np
import pytest
from backends import Field, nb, pyb
from conftest import EXECUTORS, MESH_NAMES


@pytest.mark.parametrize("executor", EXECUTORS)
@pytest.mark.parametrize("mesh", MESH_NAMES)
def test_div_phi(mesh: str, executor: str, phi: Field) -> None:
    pyb_res = pyb.fvc.div(phi)
    nb_res = nb.exp.div(phi)

    rtol = 1e-9 if executor == "Serial" else 1e-8
    np.testing.assert_allclose(
        nb_res, pyb_res, rtol=rtol, atol=1e-12 * np.abs(pyb_res).max()
    )


@pytest.mark.parametrize("executor", EXECUTORS)
@pytest.mark.parametrize("mesh", MESH_NAMES)
@pytest.mark.parametrize("scheme", ["linear", "upwind", "linearUpwind"])
def test_div_phi_T(mesh: str, scheme: str, executor: str, T: Field, phi: Field) -> None:
    pyb_res = pyb.fvc.div(phi, T, scheme=scheme)
    nb_res = nb.exp.div(phi, T, scheme=scheme)

    rtol = 1e-9 if executor == "Serial" else 1e-8
    np.testing.assert_allclose(
        nb_res, pyb_res, rtol=rtol, atol=1e-12 * np.abs(pyb_res).max()
    )


@pytest.mark.parametrize("executor", EXECUTORS)
@pytest.mark.parametrize("mesh", MESH_NAMES)
@pytest.mark.parametrize("scheme", ["linear", "upwind", "linearUpwind"])
def test_div_phi_U(mesh: str, scheme: str, executor: str, U: Field, phi: Field) -> None:
    pyb_res = pyb.fvc.div(phi, U, scheme=scheme)
    nb_res = nb.exp.div(phi, U, scheme=scheme)

    rtol = 1e-9 if executor == "Serial" else 1e-8
    np.testing.assert_allclose(
        nb_res, pyb_res, rtol=rtol, atol=1e-12 * np.abs(pyb_res).max()
    )
