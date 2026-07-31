# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The OpenFOAM viscosity fallback publishes a real transport model's ``nu``.

``select_viscosity_model`` returns an ``OpenFOAMViscosityModel`` for every
``transportModel`` with no registered native model, and the solver then asks it
for ``fields.nu``. Every other test of that adapter injects a fake factory, so
they all passed while pybFoam's ``singlePhaseTransportModel`` exposed no ``nu()``
binding at all and the fallback died on its first real use (a non-Newtonian
tutorial, ``AttributeError: … has no attribute 'nu'``). This test therefore uses
the *real* pybFoam transport, in a subprocess (one ``Foam::Time`` per process,
see :mod:`_non_newtonian_nu_worker`).

The case (``non_newtonian_base``) is a unit cube of 1 x 1 x 4 uniform cells whose
two z faces are noSlip walls, carrying a uniform ``U = (1 0 0)``. With
``Gauss linear`` gradients the two interior cells see a uniform neighbourhood, so
their strain rate is 0; the two near-wall cells see the wall's 0 against their own
1 across half a cell, giving ``dUx/dz = 1/0.25 = 4`` and hence
``strainRate = sqrt(2)*mag(symm(grad U)) = 4 1/s``. Feeding those into the case's
CrossPowerLaw, ``nu = (nu0 - nuInf)/(1 + (m*strainRate)^n) + nuInf`` with
``nu0 = 1e-3``, ``nuInf = 1e-5``, ``m = 1``, ``n = 2``, gives the two expected
values below. They are exact rational arithmetic on an orthogonal mesh, so the
tolerance is at round-off.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).parent
_WORKER = _HERE / "_non_newtonian_nu_worker.py"
_CASE = _HERE / "non_newtonian_base"  # CrossPowerLaw box: no native model exists

#: Zero strain rate ⇒ nu = (nu0 - nuInf)/1 + nuInf = nu0.
NU_UNSHEARED = 1e-3

#: strainRate = 4 1/s ⇒ nu = (1e-3 - 1e-5)/(1 + 4**2) + 1e-5.
NU_SHEARED = 9.9e-4 / 17.0 + 1e-5

#: Cell order along z for a 1 x 1 x 4 block: the two outer cells touch the walls.
NU_PER_CELL = [NU_SHEARED, NU_UNSHEARED, NU_UNSHEARED, NU_SHEARED]


def test_non_newtonian_fallback_publishes_transport_nu(tmp_path: Path) -> None:
    """The fallback's ``nu_field`` carries the pybFoam transport's rate-dependent nu."""
    case = tmp_path / "case"
    shutil.copytree(_CASE, case)

    subprocess.run([sys.executable, str(_WORKER), str(case)], check=True, cwd=str(case.parent))

    result = json.loads((case / "nu.json").read_text())
    assert result["selected"] == "OpenFOAMViscosityModel"
    assert result["name"] == "nu"
    np.testing.assert_allclose(
        result["internal"],
        NU_PER_CELL,
        rtol=1e-12,
        err_msg="non_newtonian_base: CrossPowerLaw nu published by the OpenFOAM fallback",
    )
