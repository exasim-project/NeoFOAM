# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The Boussinesq helpers of the NeoFOAM bindings against OpenFOAM's own steps.

incompressibleFluidNeoN's buoyancy builds buoyantBoussinesqPimpleFoam's pEqn.H from
``nfb.sn_grad`` (``fvc::snGrad`` with the case's scheme), ``nfb.mag_sf``,
``nfb.create_gh``/``create_ghf`` (gh.H), ``nfb.constrain_pressure`` (constrainPressure)
and ``nfb.update_velocity_buoyant`` (``U = HbyA + rAU*reconstruct(phi/rAUf)``). Each is
compared with the OpenFOAM operation on the orthogonal and the sheared mesh, the latter
the one where ``corrected`` and ``uncorrected`` snGrad differ.

``g = (0 -9.81 -2)`` and ``hRef = 0.5`` (``cases/common/constant``) are off-axis, so the
gh.H reference ``ghRef = g & (cmptMag(g)/|g|) * hRef`` is not just ``g_y * hRef``; the
expected ``gh`` is written out from them. The constrainPressure flux ``flux(U) +
flux(U)*interpolate(T)`` stands in for the buoyancy flux; with a seeded ``U`` that crosses
every wall it differs from ``Sf & U`` there, so the gradient it sets is nonzero on every
wall face. The tolerances are those of the other
operator parities: a few ULP per face for the purely geometric results, ``1e-9``
relative for results through a gradient reconstruction (``rtol 1e-8`` on the GPU,
whose reductions reorder the sums).
"""

from __future__ import annotations

import numpy as np
import pytest
from backends import nb, pyb
from case_setup import Field, Simulation, simulation
from conftest import EXECUTORS, MESH_NAMES

G = np.array([0.0, -9.81, -2.0])
H_REF = 0.5
GH_REF = G @ (np.abs(G) / np.linalg.norm(G)) * H_REF

SCHEMES = ["corrected", "uncorrected"]


def _rtol(executor: str) -> float:
    """The relative tolerance of a comparison on ``executor``."""
    return 1e-9 if executor == "Serial" else 1e-8


def _seed(sim: Simulation) -> tuple[Field, Field, Field, Field]:
    """Smooth analytic ``T``, ``U``, ``rAU``, ``p_rgh`` on the cell centres."""
    x, y, z = sim.mesh.cell_centres.T

    T = sim.field("T")
    T[:] = 2.0 + np.sin(np.pi * x) * np.cos(np.pi * y) + 0.3 * z

    U = sim.field("U")
    # positive components: U crosses every wall face (zeroGradient walls)
    U[:] = np.stack(
        [1.0 + 0.5 * np.cos(np.pi * y), 1.0 + 0.5 * np.sin(np.pi * x), 0.5 + 0.2 * x * z], axis=1
    )

    rAU = sim.field("rAU")
    rAU[:] = 0.4 + 0.1 * np.sin(np.pi * x) * np.sin(np.pi * y)

    p_rgh = sim.field("p_rgh")
    p_rgh[:] = 1.0 + x * y - 0.5 * z

    return T, U, rAU, p_rgh


@pytest.mark.parametrize("executor", EXECUTORS)
@pytest.mark.parametrize("mesh", MESH_NAMES)
@pytest.mark.parametrize("scheme", SCHEMES)
def test_sn_grad_matches_openfoam(mesh: str, scheme: str, executor: str) -> None:
    T, _, _, _ = _seed(simulation(mesh, executor))

    expected = pyb.fvc.sn_grad(T, scheme)
    result = nb.buoyancy.sn_grad(T, scheme)

    np.testing.assert_allclose(
        result,
        expected,
        rtol=_rtol(executor),
        atol=1e-12 * np.abs(expected).max(),
        err_msg=f"{mesh}: snGrad(T) {scheme}",
    )


@pytest.mark.parametrize("executor", EXECUTORS)
def test_sn_grad_scheme_selection_is_not_vacuous(executor: str) -> None:
    """On ``sheared`` the corrected snGrad differs from the uncorrected one by ~30% of peak."""
    T, _, _, _ = _seed(simulation("sheared", executor))

    corrected = nb.buoyancy.sn_grad(T, "corrected")
    uncorrected = nb.buoyancy.sn_grad(T, "uncorrected")

    assert np.abs(corrected - uncorrected).max() > 0.05 * np.abs(uncorrected).max()


@pytest.mark.parametrize("executor", EXECUTORS)
@pytest.mark.parametrize("mesh", MESH_NAMES)
def test_mag_sf_matches_openfoam(mesh: str, executor: str) -> None:
    simulation(mesh, executor)

    result = nb.buoyancy.mag_sf(mesh, executor)

    np.testing.assert_allclose(result, pyb.buoyancy.mag_sf(mesh), rtol=1e-14, err_msg=mesh)


@pytest.mark.parametrize("executor", EXECUTORS)
@pytest.mark.parametrize("mesh", MESH_NAMES)
def test_gh_is_the_geopotential_of_the_cell_centres(mesh: str, executor: str) -> None:
    sim = simulation(mesh, executor)

    result = nb.buoyancy.gh(mesh, executor)

    # the boundary values sit on the wall faces, the tail of the face centres
    n_walls = len(result) - len(sim.mesh.cell_centres)
    wall_centres = pyb.buoyancy.face_centres(mesh)[-n_walls:]
    expected = np.concatenate([sim.mesh.cell_centres @ G, wall_centres @ G]) - GH_REF
    np.testing.assert_allclose(result, expected, rtol=1e-13, atol=1e-13, err_msg=mesh)


@pytest.mark.parametrize("executor", EXECUTORS)
@pytest.mark.parametrize("mesh", MESH_NAMES)
def test_ghf_is_the_geopotential_of_the_face_centres(mesh: str, executor: str) -> None:
    simulation(mesh, executor)

    result = nb.buoyancy.ghf(mesh, executor)

    expected = pyb.buoyancy.face_centres(mesh) @ G - GH_REF
    np.testing.assert_allclose(result, expected, rtol=1e-13, atol=1e-13, err_msg=mesh)


@pytest.mark.parametrize("executor", EXECUTORS)
@pytest.mark.parametrize("mesh", MESH_NAMES)
def test_constrain_pressure_matches_openfoam(mesh: str, executor: str) -> None:
    T, U, rAU, p_rgh = _seed(simulation(mesh, executor))

    expected = pyb.buoyancy.constrain_pressure(p_rgh, U, rAU, T)
    result = nb.buoyancy.constrain_pressure(p_rgh, U, rAU, T)

    assert np.abs(expected).min() > 0.0, "the flux must leave every wall face open"
    np.testing.assert_allclose(
        result,
        expected,
        rtol=_rtol(executor),
        atol=1e-12 * np.abs(expected).max(),
        err_msg=f"{mesh}: fixedFluxPressure gradient",
    )


def test_constrain_pressure_rejects_a_flux_of_another_mesh() -> None:
    _, U, rAU, p_rgh = _seed(simulation("cartesian_nx5", "Serial"))

    with pytest.raises(RuntimeError, match="phiHbyA lives on another mesh"):
        nb.buoyancy.constrain_pressure_with_foreign_flux(p_rgh, U, rAU)


@pytest.mark.parametrize("executor", EXECUTORS)
@pytest.mark.parametrize("mesh", MESH_NAMES)
def test_update_velocity_buoyant_matches_openfoam(mesh: str, executor: str) -> None:
    T, U, rAU, _ = _seed(simulation(mesh, executor))

    expected = pyb.buoyancy.update_velocity_buoyant(U, rAU, T)
    result = nb.buoyancy.update_velocity_buoyant(U, rAU, T)

    np.testing.assert_allclose(
        result,
        expected,
        rtol=_rtol(executor),
        atol=1e-12 * np.abs(expected).max(),
        err_msg=f"{mesh}: U + rAU*reconstruct(phi/rAUf)",
    )
