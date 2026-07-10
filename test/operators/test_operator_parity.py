# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Cross-backend parity of FV operators (pybFoam fvc vs neon).

Every operator in ``operator_defs.OPERATORS`` is evaluated by both backends on
the same staged case — same mesh, same seeded ``0/`` fields, same
``system/fvSchemes`` — and the internal-field results must agree to the
operator's tolerances. The matrix is mesh x operator x div-scheme x executor;
GPU runs assert at a relaxed rtol (summation order differs) and skip when no
device is available.

Implicit operators (``imp_*``) are evaluated on the neon side as the assembled
matrix applied to the current field, ``(A·psi - b) / V``; in OpenFOAM that
matrix-apply reproduces the explicit operator (``M & psi == fvc.op(...)``), so
the reference is the pybFoam explicit result of the ``IMPLICIT_TWINS`` twin.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pytest
from conftest import TWO_D_MESHES, MeshResults
from operator_defs import IMPLICIT_TWINS, OPERATORS, OpSpec

MESH_NAMES = ["cartesian_nx5", "cartesian_nx20", "tiltedCube"]
EXECUTORS = ["Serial", "GPU"]
GPU_RTOL = 1e-8

PARAMS = [
    pytest.param(mesh, op, scheme, id=f"{mesh}-{op}-{scheme}")
    for mesh in MESH_NAMES
    for op, spec in OPERATORS.items()
    for scheme in spec.div_schemes
]


def _check(
    ref: np.ndarray, cand: np.ndarray, rtol: float, atol: float, label: str
) -> None:
    if np.allclose(cand, ref, rtol=rtol, atol=atol):
        return
    diff = np.abs(cand - ref)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(np.abs(ref) > 0.0, diff / np.abs(ref), np.inf)
        rel = np.where(diff == 0.0, 0.0, rel)
    raise AssertionError(
        f"{label}: max_abs_diff={diff.max():.6e} at {int(diff.argmax())}, "
        f"max_rel_diff={rel.max():.6e}, rtol={rtol:.1e}, atol={atol:.1e}, "
        f"n={ref.size}, ref_scale={np.abs(ref).max():.6e}"
    )


def _assert_match(
    ref: np.ndarray, cand: np.ndarray, spec: OpSpec, executor: str, two_d: bool
) -> None:
    # neon surface fields append boundary-face values after the internal faces;
    # compare against the OpenFOAM internal length (mirrors EqualsInternal).
    assert cand.shape[0] >= ref.shape[0], (
        f"neon result shorter than reference: {cand.shape} vs {ref.shape}"
    )
    cand = cand[: ref.shape[0]]

    rtol = spec.rtol if executor == "Serial" else max(spec.rtol, GPU_RTOL)
    scale = float(np.abs(ref).max())
    atol = spec.atol_scale * scale if scale > 0.0 else spec.atol_scale

    if ref.ndim == 2:
        assert cand.shape == ref.shape, f"shape mismatch: {cand.shape} vs {ref.shape}"
        for comp in range(ref.shape[1]):
            comp_atol = atol
            if comp == 2 and two_d and spec.z_atol is not None:
                comp_atol = max(atol, spec.z_atol * max(scale, 1.0))
            _check(ref[:, comp], cand[:, comp], rtol, comp_atol, f"component {comp}")
    else:
        assert cand.shape == ref.shape, f"shape mismatch: {cand.shape} vs {ref.shape}"
        _check(ref, cand, rtol, atol, "internal field")


@pytest.mark.parametrize("executor", EXECUTORS)
@pytest.mark.parametrize(("mesh", "op", "scheme"), PARAMS)
def test_operator_parity(
    mesh: str,
    op: str,
    scheme: str,
    executor: str,
    mesh_results: Callable[[str], MeshResults],
    gpu_available: bool,
) -> None:
    if executor == "GPU" and not gpu_available:
        pytest.skip("no GPU executor available on this host")
    results = mesh_results(mesh)
    reference = results.of[scheme][IMPLICIT_TWINS.get(op, op)]
    candidate = results.neon[executor][scheme][op]
    _assert_match(reference, candidate, OPERATORS[op], executor, mesh in TWO_D_MESHES)
