# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Matrix-apply parity of the implicit laplacian operators.

neon assembles ``nn.imp.laplacian(Gamma, psi)`` via ``nfb.evaluate_implicit``
and applies the matrix to the current field, ``(A·psi - b) / V``; the
reference is the pybFoam explicit twin (``M & psi == fvc.laplacian(...)`` in
OpenFOAM).
"""

from __future__ import annotations

from typing import Callable

import pytest
from conftest import MeshResults, assert_operator_parity, operator_params


@pytest.mark.parametrize(
    ("mesh", "scheme", "executor"), operator_params("imp_laplacian_Gamma_T")
)
def test_imp_laplacian_Gamma_T(
    mesh: str,
    scheme: str,
    executor: str,
    mesh_results: Callable[[str], MeshResults],
    gpu_available: bool,
) -> None:
    assert_operator_parity(
        "imp_laplacian_Gamma_T", mesh, scheme, executor, mesh_results, gpu_available
    )


@pytest.mark.parametrize(
    ("mesh", "scheme", "executor"), operator_params("imp_laplacian_Gamma_U")
)
def test_imp_laplacian_Gamma_U(
    mesh: str,
    scheme: str,
    executor: str,
    mesh_results: Callable[[str], MeshResults],
    gpu_available: bool,
) -> None:
    assert_operator_parity(
        "imp_laplacian_Gamma_U", mesh, scheme, executor, mesh_results, gpu_available
    )
