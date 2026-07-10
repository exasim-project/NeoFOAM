# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Matrix-apply parity of implicit scalar convection div(phi,T).

neon assembles ``nn.imp.div(phi, T)`` via ``nfb.evaluate_implicit`` and applies
the matrix to the current field, ``(A·psi - b) / V``; the reference is the
pybFoam explicit twin (``M & psi == fvc.div(...)`` in OpenFOAM). Runs under
every div scheme.
"""

from __future__ import annotations

from typing import Callable

import pytest
from conftest import MeshResults, assert_operator_parity, operator_params

OP = "imp_div_phi_T"


@pytest.mark.parametrize(("mesh", "scheme", "executor"), operator_params(OP))
def test_parity(
    mesh: str,
    scheme: str,
    executor: str,
    mesh_results: Callable[[str], MeshResults],
    gpu_available: bool,
) -> None:
    assert_operator_parity(OP, mesh, scheme, executor, mesh_results, gpu_available)
