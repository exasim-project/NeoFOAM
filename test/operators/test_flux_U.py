# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Parity of the face flux: fvc.flux(U) vs nfb.flux(U)."""

from __future__ import annotations

from typing import Callable

import pytest
from conftest import MeshResults, assert_operator_parity, operator_params

OP = "flux_U"


@pytest.mark.parametrize(("mesh", "scheme", "executor"), operator_params(OP))
def test_parity(
    mesh: str,
    scheme: str,
    executor: str,
    mesh_results: Callable[[str], MeshResults],
    gpu_available: bool,
) -> None:
    assert_operator_parity(OP, mesh, scheme, executor, mesh_results, gpu_available)
