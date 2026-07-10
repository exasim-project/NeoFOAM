# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Parity of the gradient: fvc.grad(T) vs nn.exp.grad(T)."""

from __future__ import annotations

from typing import Callable

import pytest
from conftest import MeshResults, assert_operator_parity, operator_params


@pytest.mark.parametrize(("mesh", "scheme", "executor"), operator_params("grad_T"))
def test_grad_T(
    mesh: str,
    scheme: str,
    executor: str,
    mesh_results: Callable[[str], MeshResults],
    gpu_available: bool,
) -> None:
    assert_operator_parity(
        "grad_T", mesh, scheme, executor, mesh_results, gpu_available
    )
