# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Advance alpha with incompressibleVoF's own alpha solve, the flow frozen.

The reference for the openfoam backend: the real two-phase mixture
(``immiscibleIncompressibleTwoPhaseMixture``, so the real interfaceProperties
``cAlpha``/``nHatf``) and the solver's ``_solve_alpha_python`` with its
rho/rhoPhi bookkeeping, called once per time step with ``phi = createPhi(U)``
held fixed — what incompressibleVoF does to alpha when the momentum and
pressure steps change nothing. Fixed ``deltaT`` only.

Usage: ``python _incompressible_vof_alpha_worker.py <case-dir> <out.npz>``
"""

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pybFoam as pyf
import pybFoam.multiphase as multiphase
from pybFoam import Word, fvc, surfaceScalarField, volScalarField

from neofoam.solver.incompressibleVoF.models.alpha_advection.models.mules import (
    _solve_alpha_python,
)


def _main(case: Path, out: Path) -> None:
    os.chdir(case)
    arg_list = pyf.argList(["incompressibleVoFAlpha", "-noFunctionObjects"])
    runtime = pyf.Time(arg_list)
    mesh = pyf.fvMesh(runtime)
    U = pyf.volVectorField.read_field(mesh, "U")
    phi = pyf.createPhi(U)
    mixture = multiphase.immiscibleIncompressibleTwoPhaseMixture(U, phi)
    alpha1 = mixture.alpha1()
    alpha2 = mixture.alpha2()
    mesh.setFluxRequired(alpha1.name())
    # The fields of shared_field_build_steps, built the same way.
    rho = volScalarField(Word("rho"), alpha1 * mixture.rho1() + alpha2 * mixture.rho2())
    rho.oldTime()
    rho_phi = surfaceScalarField(Word("rhoPhi"), fvc.interpolate(rho) * phi)
    alpha_phi_un = surfaceScalarField(Word("alphaPhiUn"), 0.0 * phi)
    alpha_phi10 = surfaceScalarField(Word("alphaPhi10"), phi * fvc.interpolate(alpha1))
    alpha_phi1_corr0: list[Optional[surfaceScalarField]] = [None]

    times = [runtime.value()]
    alphas = [np.asarray(alpha1.internalField()).copy()]
    while runtime.run():
        runtime.increment()
        _solve_alpha_python(
            alpha1,
            alpha2,
            phi,
            rho_phi,
            rho,
            mixture,
            alpha_phi_un,
            alpha_phi10,
            alpha_phi1_corr0,
        )
        if runtime.outputTime():
            times.append(runtime.value())
            alphas.append(np.asarray(alpha1.internalField()).copy())
    np.savez(out, times=np.array(times), alpha=np.array(alphas))


if __name__ == "__main__":
    _main(Path(sys.argv[1]), Path(sys.argv[2]))
