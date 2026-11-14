# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Cross-backend parity of interFoam's two-phase mixture and buoyancy terms.

pybFoam spells each term as interFoam's createFields.H, gh.H, UEqn.H and pEqn.H do (the
mixture properties from a real ``immiscibleIncompressibleTwoPhaseMixture``); neon calls the
NeoFOAM bindings. Surface tension is left out, as in the NeoN VoF solver for now. The
``twoPhase`` case keeps these inputs apart from ``sharpSphere``, whose parity tests stay as
they are; its three operator meshes are the ``sharpSphere`` ones.

Setup: a smooth, tilted water/air interface whose phase fraction over- and undershoots
[0, 1] by 0.05 (and whose fixed-value patches lie outside it), so the clamp of the mixture
viscosity acts on cells and patches; a solid-body rotation with drift for ``U`` and its
``phi``; a smooth positive ``rAU``; a hydrostatic ``p_rgh`` with a horizontal ramp. ``g`` is
oblique and ``hRef`` set, so every term of gh.H counts. Besides the three operator meshes
(walls only, the sheared one non-orthogonal) the ``damBreak`` mesh is a small graded 2-D
damBreak: fixedFluxPressure walls next to a fixed-value atmosphere (``constrain_pressure``
must leave it alone) and an empty direction, which takes the 2-D branch of reconstruct's
tensor inverse.

Tolerances: the per-face and per-cell kernels (density, viscosity, products, interpolation,
gh, the pressure constraint, the static pressure) follow OpenFOAM's operation order with
OpenFOAM's weights and delta coefficients, so Serial and CPU are exact (``atol=0``); the GPU
may fuse multiply-adds, a few ulps of the peak. The snGrad terms (gravity face force,
buoyancy flux, momentum flux) add the non-orthogonal correction from a Gauss gradient whose
face sums are atomic on CPU and GPU, so only Serial is guaranteed exact (the GPU within 1e-14
of the peak). The buoyant velocity update inherits this through its buoyancy flux (seen on
the sheared mesh with threads: 6e-14 absolute). The other reconstructions gather each cell's
faces in OpenFOAM's order, exact off the GPU, which fuses the tensor algebra (seen: 1e-14 of
the peak at most).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pytest

from ._parity_backends import _Simulation
from ._two_phase_backends import _flux, _phases, _transport_properties, nb, pyb
from .conftest import EXECUTORS, MESH_NAMES

MESHES = [*MESH_NAMES, "damBreak"]

_SimulationFactory = Callable[[str, str], _Simulation]


def _seed(sim: _Simulation) -> None:
    """Interface, temperature, rAU, rotation with drift and its phi, then p_rgh."""
    centres = sim.mesh.cell_centres
    extent = np.ptp(centres, axis=0).max()
    x, y, _ = ((centres - centres.mean(axis=0)) / extent).T
    sim.field("alpha.water")[:] = 1.1 * 0.5 * (1.0 - np.tanh((y + 0.3 * x) / 0.1)) - 0.05
    sim.field("T")[:] = 300.0 + 10.0 * np.sin(3.0 * x) * np.cos(2.0 * y)
    sim.field("rAU")[:] = 1e-3 * (1.0 + 0.5 * np.sin(3.0 * x + 1.0) * np.cos(2.0 * y))
    U = sim.field("U")
    U[:] = np.column_stack([-y, x, np.full_like(x, 0.2)])
    _flux(U)
    # after U and phi: setting p_rgh constrains its fixedFluxPressure walls with them
    sim.field("p_rgh")[:] = 1000.0 * 9.81 * (0.5 - y) + 20.0 * x


def _exact_off_gpu(executor: str) -> tuple[float, float]:
    """(rtol, atol relative to the peak) of the kernels without sums."""
    return (1e-12, 1e-14) if executor == "GPU" else (0.0, 0.0)


def _atomic_sums(executor: str) -> tuple[float, float]:
    """(rtol, atol relative to the peak) of the snGrad terms: exact on Serial only."""
    return (0.0, 0.0) if executor == "Serial" else (1e-12, 1e-14)


def _assert_matches(
    result: np.ndarray, expected: np.ndarray, tol: tuple[float, float], msg: str
) -> None:
    rtol, atol = tol
    np.testing.assert_allclose(
        result, expected, rtol=rtol, atol=atol * np.abs(expected).max(), err_msg=msg
    )


def test_read_two_phase_transport_properties_reads_the_case(
    two_phase_simulation: _SimulationFactory,
) -> None:
    """The values of cases/twoPhase/constant/transportProperties."""
    alpha = two_phase_simulation("cartesian_nx5", "Serial").field("alpha.water")

    properties = _transport_properties(alpha)

    np.testing.assert_array_equal(properties, [1000.0, 1e-06, 1.0, 1.48e-05, 0.07])
    np.testing.assert_array_equal(_phases(alpha), ["water", "air"])


@pytest.mark.parametrize(
    "term", ["density", "viscosity", "rho_phi", "gh", "ghf", "static_pressure"]
)
@pytest.mark.parametrize("executor", EXECUTORS)
@pytest.mark.parametrize("mesh", MESHES)
def test_mixture_term_matches_openfoam(
    two_phase_simulation: _SimulationFactory, mesh: str, executor: str, term: str
) -> None:
    sim = two_phase_simulation(mesh, executor)
    _seed(sim)
    alpha = sim.field("alpha.water")

    expected = getattr(pyb, term)(alpha)
    result = getattr(nb, term)(alpha)

    _assert_matches(result, expected, _exact_off_gpu(executor), f"{mesh}/{executor}")


@pytest.mark.parametrize("into", [False, True], ids=["new", "out"])
@pytest.mark.parametrize("executor", EXECUTORS)
@pytest.mark.parametrize("mesh", MESHES)
def test_linear_interpolate_matches_openfoam(
    two_phase_simulation: _SimulationFactory, mesh: str, executor: str, into: bool
) -> None:
    sim = two_phase_simulation(mesh, executor)
    _seed(sim)
    rau = sim.field("rAU")

    expected = pyb.interpolate(rau)
    result = nb.interpolate(rau, into)

    _assert_matches(result, expected, _exact_off_gpu(executor), f"{mesh}/{executor}")


@pytest.mark.parametrize("executor", EXECUTORS)
@pytest.mark.parametrize("mesh", MESHES)
def test_multiply_matches_openfoam(
    two_phase_simulation: _SimulationFactory, mesh: str, executor: str
) -> None:
    sim = two_phase_simulation(mesh, executor)
    _seed(sim)
    t, rau = sim.field("T"), sim.field("rAU")

    expected = pyb.multiply(t, rau)
    result = nb.multiply(t, rau)

    _assert_matches(result, expected, _exact_off_gpu(executor), f"{mesh}/{executor}")


@pytest.mark.parametrize("term", ["gravity_face_force", "buoyancy_flux", "momentum_flux"])
@pytest.mark.parametrize("into", [False, True], ids=["new", "out"])
@pytest.mark.parametrize("executor", EXECUTORS)
@pytest.mark.parametrize("mesh", MESHES)
def test_buoyancy_face_term_matches_openfoam(
    two_phase_simulation: _SimulationFactory, mesh: str, executor: str, term: str, into: bool
) -> None:
    sim = two_phase_simulation(mesh, executor)
    _seed(sim)
    alpha = sim.field("alpha.water")

    expected = getattr(pyb, term)(alpha)
    result = getattr(nb, term)(alpha, into)

    _assert_matches(result, expected, _atomic_sums(executor), f"{mesh}/{executor}")


@pytest.mark.parametrize("executor", EXECUTORS)
@pytest.mark.parametrize("mesh", MESHES)
def test_constrain_pressure_matches_openfoam(
    two_phase_simulation: _SimulationFactory, mesh: str, executor: str
) -> None:
    """The wall values P + gradient/deltaCoeffs; the atmosphere keeps its fixed value."""
    sim = two_phase_simulation(mesh, executor)
    _seed(sim)
    alpha = sim.field("alpha.water")

    expected = pyb.constrain_pressure(alpha)
    result = nb.constrain_pressure(alpha)

    _assert_matches(result, expected, _exact_off_gpu(executor), f"{mesh}/{executor}")


@pytest.mark.parametrize("executor", EXECUTORS)
@pytest.mark.parametrize("mesh", MESHES)
def test_reconstruct_matches_openfoam(
    two_phase_simulation: _SimulationFactory, mesh: str, executor: str
) -> None:
    sim = two_phase_simulation(mesh, executor)
    _seed(sim)
    phi = _flux(sim.field("U"))

    expected = pyb.reconstruct(phi)
    result = nb.reconstruct(phi)

    _assert_matches(result, expected, _exact_off_gpu(executor), f"{mesh}/{executor}")


@pytest.mark.parametrize("executor", EXECUTORS)
@pytest.mark.parametrize("mesh", MESHES)
def test_update_velocity_buoyant_matches_openfoam(
    two_phase_simulation: _SimulationFactory, mesh: str, executor: str
) -> None:
    sim = two_phase_simulation(mesh, executor)
    _seed(sim)
    alpha = sim.field("alpha.water")

    expected = pyb.buoyant_velocity(alpha)
    result = nb.buoyant_velocity(alpha)

    _assert_matches(result, expected, _atomic_sums(executor), f"{mesh}/{executor}")
