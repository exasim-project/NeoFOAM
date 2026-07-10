# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Parity of the face flux: fvc.flux(U) vs nfb.flux(U)."""

from __future__ import annotations

from typing import Callable

import pytest
from conftest import MeshResults, assert_operator_parity, operator_params


@pytest.mark.parametrize(("mesh", "scheme", "executor"), operator_params("flux_U"))
def test_flux_U(
    mesh: str,
    scheme: str,
    executor: str,
    mesh_results: Callable[[str], MeshResults],
    gpu_available: bool,
) -> None:
    assert_operator_parity(
        "flux_U", mesh, scheme, executor, mesh_results, gpu_available
    )
