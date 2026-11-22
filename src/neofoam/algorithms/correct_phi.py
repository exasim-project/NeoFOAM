# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The CorrectPhi divergence-free flux projection, composed from primitives.

A faithful port of the incompressible ``CorrectPhi<RAUfType, DivUType>``
(OpenFOAM ``src/finiteVolume/cfdTools/general/CorrectPhi/CorrectPhi.C``) with
its two non-transferable parameters replaced by their only inputs:

* ``pimple`` -> the non-orthogonal corrector count. The template only counts
  0..nNonOrthCorr (``solutionControlI.H``); a negative count runs no solve at
  all, which the ``frozenFlow`` tutorials (``nNonOrthogonalCorrectors -1``)
  rely on.
* ``divU`` -> dropped. Every interFoam/interIsoFoam call site passes
  ``geometricZeroField()``, for which ``fvc::div(phi) - divU`` is the operand
  itself.
"""

from __future__ import annotations

import pybFoam as pyf
from pybFoam import (
    dimensionedScalar,
    fvc,
    fvm,
    fvScalarMatrix,
    surfaceScalarField,
    volScalarField,
    volVectorField,
)

__all__ = ["correct_phi"]


def correct_phi(
    U: volVectorField,
    phi: surfaceScalarField,
    p: volScalarField,
    rAUf: dimensionedScalar | surfaceScalarField,
    n_non_orthogonal_correctors: int,
) -> None:
    """Project ``phi`` divergence-free by solving for a pressure correction."""
    mesh = U.mesh()

    pyf.correctUphiBCs(U, phi)

    # pcorr fixes its value exactly where p does; zero-gradient everywhere else.
    pcorr_types = [
        "fixedValue" if p.fixesValue(patchi) else "zeroGradient"
        for patchi in range(len(mesh.boundary()))
    ]
    pcorr = volScalarField.uniform(
        "pcorr",
        mesh,
        pyf.dimensionedScalar(pyf.Word("pcorr"), p.dimensions(), 0.0),
        pcorr_types,
    )

    if pcorr.needReference():
        # adjustPhi balances the fluxes the boundary prescribes, which on a
        # moving mesh are the relative ones.
        fvc.makeRelative(phi, U)
        pyf.adjustPhi(phi, U, pcorr)
        fvc.makeAbsolute(phi, U)

    mesh.setFluxRequired(pyf.Word("pcorr"))

    for non_orth in range(n_non_orthogonal_correctors + 1):
        final_non_orth = non_orth == n_non_orthogonal_correctors
        pcorr_eqn = fvScalarMatrix(fvm.laplacian(rAUf, pcorr) - fvc.div(phi))
        pcorr_eqn.setReference(0, 0.0, False)
        pcorr_eqn.solve(pcorr.select(final_non_orth))
        if final_non_orth:
            phi.assign(phi - pcorr_eqn.flux())
