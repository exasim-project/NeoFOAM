# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Parity of the explicit divergence operators (pybFoam fvc vs neon).

``div_phi_T`` uses the Gauss identity ``fvc.div(fvc.flux(phi, T,
key="div(phi,T)"))`` on the pybFoam side (no scalar convection overload of
``fvc.div``); the convection operators run under every div scheme.
"""

from __future__ import annotations

from typing import Callable

import pytest
from conftest import MeshResults, assert_operator_parity, operator_params


@pytest.mark.parametrize(("mesh", "scheme", "executor"), operator_params("div_phi"))
def test_div_phi(
    mesh: str,
    scheme: str,
    executor: str,
    mesh_results: Callable[[str], MeshResults],
    gpu_available: bool,
) -> None:
    assert_operator_parity(
        "div_phi", mesh, scheme, executor, mesh_results, gpu_available
    )


@pytest.mark.parametrize(("mesh", "scheme", "executor"), operator_params("div_phi_T"))
def test_div_phi_T(
    mesh: str,
    scheme: str,
    executor: str,
    mesh_results: Callable[[str], MeshResults],
    gpu_available: bool,
) -> None:
    assert_operator_parity(
        "div_phi_T", mesh, scheme, executor, mesh_results, gpu_available
    )


@pytest.mark.parametrize(("mesh", "scheme", "executor"), operator_params("div_phi_U"))
def test_div_phi_U(
    mesh: str,
    scheme: str,
    executor: str,
    mesh_results: Callable[[str], MeshResults],
    gpu_available: bool,
) -> None:
    assert_operator_parity(
        "div_phi_U", mesh, scheme, executor, mesh_results, gpu_available
    )
