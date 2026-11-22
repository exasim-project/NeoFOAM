# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Cross-backend parity of the SIMPLEC flux correction (pybFoam fvc vs neon).

The term is ``interpolate(rAtU - rAU) * snGrad(p) * magSf``. The oracle is the
line the trusted ``incompressibleFluid`` solver runs (``simpleAlgorithm.py``);
neon's ``nfb.add_consistent_flux_correction`` is applied to a *zeroed*
``phiHbyA``, so what comes back is the correction alone. ``T`` stands in for
the pressure — the case stages ``grad(T) Gauss linear``, which OpenFOAM's
``correctedSnGrad`` needs for its non-orthogonal correction.

``scheme`` is what the test is really about: OpenFOAM takes it inline through
``fvc.snGrad``, neon resolves it from the ``snGradSchemes`` subdict the worker
hands over. Only the ``sheared`` mesh can tell the two schemes apart, and it
does so by a wide margin — measured on the OpenFOAM oracle:

===============  ============  ==========================  ========
mesh             peak |term|   max |corrected−uncorrected|  of peak
===============  ============  ==========================  ========
cartesian_nx5    3.26e-2       2.9e-17                     9.0e-16
cartesian_nx20   3.05e-3       1.6e-17                     5.1e-15
sheared          1.13e-2       3.50e-3                     3.1e-1
===============  ============  ==========================  ========

So on the orthogonal meshes the two schemes coincide to machine precision and
the parity comparison says nothing about scheme selection; the guard below
pins the discrimination on ``sheared``, on *both* backends, so a neon side
that ignored the dictionary and kept its old hardcoded ``corrected`` fails by
31% of peak rather than passing quietly.

Only internal faces are compared. ``T``'s walls are ``zeroGradient``, whose
``snGrad`` is identically zero (``zeroGradientFvPatchField::snGrad``), so the
boundary faces carry no correction on either backend and could not
discriminate anything — they would also be the only place NeoFOAM's
extrapolated ``drAU`` boundary values could differ from OpenFOAM's
``zeroGradient`` ones.
"""

from __future__ import annotations

import numpy as np
import pytest
from backends import nb, pyb
from case_setup import Field, Simulation, simulation
from conftest import EXECUTORS, MESH_NAMES

SCHEMES = ["corrected", "uncorrected"]


def _seed(sim: Simulation) -> tuple[Field, Field, Field]:
    """Smooth analytic ``rAU``/``rAtU``/``T`` on the cell centres."""
    x, y, z = sim.mesh.cell_centres.T

    rAU = sim.field("rAU")
    rAU[:] = 0.4 + 0.1 * np.sin(np.pi * x) * np.sin(np.pi * y)

    rAtU = sim.field("rAtU")
    rAtU[:] = 0.6 + 0.15 * np.cos(np.pi * x) + 0.05 * z

    T = sim.field("T")
    T[:] = 2.0 + np.sin(np.pi * x) * np.cos(np.pi * y) + 0.3 * z

    # the whole term scales with rAtU - rAU; equal fields would zero it out
    assert np.abs(np.asarray(rAtU) - np.asarray(rAU)).min() > 1e-2
    return rAU, rAtU, T


@pytest.mark.parametrize("executor", EXECUTORS)
@pytest.mark.parametrize("mesh", MESH_NAMES)
@pytest.mark.parametrize("scheme", SCHEMES)
def test_simplec_flux_correction(mesh: str, scheme: str, executor: str) -> None:
    sim = simulation(mesh, executor)
    rAU, rAtU, T = _seed(sim)

    pyb_res = pyb.fvc.simplec_flux_correction(rAU, rAtU, T, scheme)
    nb_res = nb.simplec_flux_correction(rAU, rAtU, T, scheme)

    # neon appends boundary-face values after the internal faces
    nb_res = nb_res[: pyb_res.shape[0]]
    rtol = 1e-9 if executor == "Serial" else 1e-8
    np.testing.assert_allclose(nb_res, pyb_res, rtol=rtol, atol=1e-12 * np.abs(pyb_res).max())


@pytest.mark.parametrize("executor", EXECUTORS)
def test_snGrad_scheme_selection_is_not_vacuous(executor: str) -> None:
    """On ``sheared`` the two snGrad schemes must differ — on both backends.

    Without this the parity test could pass with either backend ignoring the
    requested scheme: on the orthogonal meshes ``corrected`` and
    ``uncorrected`` agree to 1e-15 of peak (table in the module docstring).
    """
    sim = simulation("sheared", executor)
    rAU, rAtU, T = _seed(sim)

    n_internal = pyb.fvc.simplec_flux_correction(rAU, rAtU, T, "corrected").shape[0]

    for term in (pyb.fvc.simplec_flux_correction, nb.simplec_flux_correction):
        corrected = term(rAU, rAtU, T, "corrected")[:n_internal]
        uncorrected = term(rAU, rAtU, T, "uncorrected")[:n_internal]
        # measured on the OpenFOAM oracle: 3.50e-3 against a peak of 1.13e-2
        assert np.abs(corrected - uncorrected).max() > 0.05 * np.abs(uncorrected).max()
