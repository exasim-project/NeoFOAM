# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Cross-backend parity of the VoF alpha fluxes (pybFoam vs NeoFOAM bindings).

The phase fraction is the ``sharpSphere`` case's sphere (alpha 1 inside, 0
outside) in a prescribed velocity, ``phi = fvc::flux(U)``: a solid-body rotation
with a vertical drift (face fluxes of every sign and size) or a uniform
translation. Two fields cover the boundary treatment: ``alpha`` has zeroGradient
walls; ``alphaFixed`` has fixedValue 1 walls, which with the zeroGradient ``U``
carry alpha inflow and outflow.

Tolerances: Serial follows OpenFOAM's arithmetic order face by face and matches
it bit for bit; ``rtol=1e-12`` only guards against compiler contraction. On CPU
(Kokkos threads) and GPU the per-cell sums are atomic and add up in a different
order (the GPU also fuses multiply-adds), which moves the last bits. One place
amplifies that: the interface normal ``gradAlphaf/(|gradAlphaf| + deltaN)`` with
``deltaN = 1e-8/h`` turns a round-off gradient (~1e-16/h) into a normal of size
~1e-8, so the compression flux may differ by ~1e-8 of its peak (seen: 3e-9 on
the 5-cell mesh with the fixedValue walls).

The two compression pieces of ``alpha_phase_flux`` are tested on their own,
boundary faces included: the compression velocity ``phic`` and the linear
compression flux, the latter with ``phir = phi`` (it is linear in any face
flux, and ``phi`` is the same on every executor, which keeps the nHatf
round-off out). Both are per-face kernels without sums, so Serial and CPU
match OpenFOAM exactly (``atol=0``); the GPU may fuse the linear
interpolation's multiply-add, a few ulps of the flux peak.

Every per-step binding the NeoN MULES ops use also writes into an existing
field (``out=``); the ``out`` cases fill a buffer of stale values that way and
must meet the same tolerances. ``alpha_phase_flux`` composed from those
variants (``vanleer_flux``, ``interface_nhatf``, ``interface_compression_velocity``,
``add_interface_compression_flux``) does the same face arithmetic as the fused
binding, so it shares its tolerances.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pytest

from ._parity_backends import _Simulation, _use_velocity, nb, pyb
from .conftest import EXECUTORS, MESH_NAMES

FIELDS = ["alpha", "alphaFixed"]


def _flux_tolerances(executor: str) -> tuple[float, float]:
    """(rtol, atol relative to the flux peak)."""
    return (1e-12, 1e-14) if executor == "Serial" else (1e-8, 2.2e-8)


def _per_face_tolerances(executor: str) -> tuple[float, float]:
    """(rtol, atol relative to the peak) of the per-face kernels: exact off the GPU."""
    return (1e-12, 1e-14) if executor == "GPU" else (0.0, 0.0)


@pytest.mark.parametrize("executor", EXECUTORS)
@pytest.mark.parametrize("mesh", MESH_NAMES)
@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("c_alpha", [0.0, 1.0])
@pytest.mark.parametrize("rotate", [True, False], ids=["rotation", "translation"])
@pytest.mark.parametrize("composed", [False, True], ids=["fused", "out"])
def test_alpha_phase_flux_matches_openfoam(
    parity_simulation: Callable[[str, str], _Simulation],
    mesh: str,
    executor: str,
    field: str,
    c_alpha: float,
    rotate: bool,
    composed: bool,
) -> None:
    sim = parity_simulation(mesh, executor)
    _use_velocity(sim, rotate)
    alpha = sim.field(field)

    expected = pyb.vof.alpha_phase_flux(alpha, c_alpha)
    result = nb.vof.alpha_phase_flux(alpha, c_alpha, composed)[: expected.shape[0]]

    rtol, atol = _flux_tolerances(executor)
    np.testing.assert_allclose(
        result,
        expected,
        rtol=rtol,
        atol=atol * np.abs(expected).max(),
        err_msg=f"{mesh}/{executor}/{field} cAlpha={c_alpha}",
    )


@pytest.mark.parametrize("executor", EXECUTORS)
@pytest.mark.parametrize("mesh", MESH_NAMES)
@pytest.mark.parametrize("c_alpha", [0.5, 1.0])
@pytest.mark.parametrize("into", [False, True], ids=["new", "out"])
def test_interface_compression_velocity_matches_openfoam(
    parity_simulation: Callable[[str, str], _Simulation],
    mesh: str,
    executor: str,
    c_alpha: float,
    into: bool,
) -> None:
    sim = parity_simulation(mesh, executor)
    phi = _use_velocity(sim)

    expected = pyb.vof.compression_velocity(phi, c_alpha)
    result = nb.vof.compression_velocity(phi, c_alpha, into)

    rtol, atol = _per_face_tolerances(executor)
    np.testing.assert_allclose(
        result,
        expected,
        rtol=rtol,
        atol=atol * np.abs(expected).max(),
        err_msg=f"{mesh}/{executor} cAlpha={c_alpha}",
    )


@pytest.mark.parametrize("executor", EXECUTORS)
@pytest.mark.parametrize("mesh", MESH_NAMES)
@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("rotate", [True, False], ids=["rotation", "translation"])
@pytest.mark.parametrize("into", [False, True], ids=["new", "out"])
def test_linear_compression_flux_matches_openfoam(
    parity_simulation: Callable[[str, str], _Simulation],
    mesh: str,
    executor: str,
    field: str,
    rotate: bool,
    into: bool,
) -> None:
    sim = parity_simulation(mesh, executor)
    _use_velocity(sim, rotate)
    alpha = sim.field(field)

    expected = pyb.vof.linear_compression_flux(alpha)
    result = nb.vof.linear_compression_flux(alpha, into)

    rtol, atol = _per_face_tolerances(executor)
    np.testing.assert_allclose(
        result,
        expected,
        rtol=rtol,
        atol=atol * np.abs(expected).max(),
        err_msg=f"{mesh}/{executor}/{field}",
    )
