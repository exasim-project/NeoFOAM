# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Cross-backend parity of the explicit gradient (pybFoam fvc vs neon).

``test_grad_U_cell_limited`` covers the runtime-selected gradient operator
(``nfb.GradScheme``, the one the momentum equation and the turbulence closures
build from ``gradSchemes``) under the two ``cellLimited`` variants, comparing
all nine tensor components. The variants are staged as their own
``grad(U_<scheme>)`` keys in the case's ``fvSchemes`` rather than passed as a
raw ``TokenList``: NeoN's ``CellLimitedGrad::readCoeff`` only reads a *numeric*
coefficient token and silently falls back to k=1 otherwise, so a TokenList built
from Python strings would turn ``cellLimitedOff`` into full limiting unnoticed.

The two variants make distinct claims:

* **k=1** — the limiter is actually applied. The unlimited gradient differs
  from it by 4.3e-1 … 5.0e-1 of peak across these meshes, against a parity of
  ~1e-14, so a scheme selection that silently fell back to the plain
  Gauss-Green gradient fails by thirteen orders of magnitude.
* **k=0** — the *coefficient* is read. NeoN defaults an absent k to 1
  (strongest limiting), so dropping the coefficient would limit here; k=0 must
  reproduce the unlimited gradient, which it does to machine precision.
"""

from __future__ import annotations

import numpy as np
import pytest
from backends import nb, pyb
from case_setup import simulation
from conftest import EXECUTORS, MESH_NAMES


@pytest.mark.parametrize("executor", EXECUTORS)
@pytest.mark.parametrize("mesh", MESH_NAMES)
def test_grad_T(mesh: str, executor: str) -> None:
    sim = simulation(mesh, executor)
    x, y, z = sim.mesh.cell_centres.T

    T = sim.field("T")
    T[:] = 2.0 + np.sin(np.pi * x) * np.cos(np.pi * y) + 0.3 * z

    pyb_res = pyb.fvc.grad(T)
    nb_res = nb.exp.grad(T)

    rtol = 1e-12 if executor == "Serial" else 1e-8
    np.testing.assert_allclose(nb_res, pyb_res, rtol=rtol, atol=1e-12 * np.abs(pyb_res).max())


@pytest.mark.parametrize("executor", EXECUTORS)
@pytest.mark.parametrize("mesh", MESH_NAMES)
@pytest.mark.parametrize("scheme", ["cellLimited", "cellLimitedOff"])
def test_grad_U_cell_limited(mesh: str, scheme: str, executor: str) -> None:
    sim = simulation(mesh, executor)
    x, y, z = sim.mesh.cell_centres.T

    U = sim.field("U")
    u = np.asarray(U)
    u[:, 0] = 1.0 + np.sin(np.pi * y) + 0.5 * np.sin(np.pi * x)
    u[:, 1] = 0.5 + np.cos(np.pi * x) + 0.5 * np.cos(np.pi * y)
    u[:, 2] = 0.1 + 0.2 * np.sin(np.pi * z)
    U[:] = u

    # NeoN stores grad(U)_ij = dU_i/dx_j, OpenFOAM the transpose; measured, not
    # assumed — comparing the two untransposed is off by 1.975 against a peak of
    # 3.119, transposed they agree to 8.9e-15.
    pyb_res = pyb.fvc.grad_tensor(U, scheme).reshape(-1, 3, 3).transpose(0, 2, 1)
    nb_res = nb.exp.grad_tensor(U, scheme).reshape(-1, 3, 3)

    rtol = 1e-12 if executor == "Serial" else 1e-8
    np.testing.assert_allclose(nb_res, pyb_res, rtol=rtol, atol=1e-12 * np.abs(pyb_res).max())
