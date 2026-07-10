# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Evaluate explicit fvc operators with pybFoam and dump the results as ``.npy``.

Run as a standalone subprocess (``python run_openfoam_ops.py <case_dir> <out_dir>
<op> [<op> ...]``) — constructing a ``Foam::Time`` must happen exactly once per
process. Operator keys are defined in ``operator_defs.OPERATORS``.

All schemes are resolved from the case's ``system/fvSchemes`` — the same file
the neon runner consumes — so both backends discretise identically.
``div_phi_T`` uses the Gauss identity ``div(flux(phi, T, "div(phi,T)"))``
because pybFoam has no scalar convection overload of ``fvc.div``; the named
``fvc.flux`` overload looks the scheme up under ``div(phi,T)``, so the identity
holds for every scheme variant.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pybFoam as pyf
from pybFoam import fvc, volScalarField, volVectorField


def compute_ops(case_dir: Path, out_dir: Path, ops: list[str]) -> None:
    os.chdir(case_dir)
    runtime = pyf.Time(pyf.argList(["ofOpsRunner"]))
    mesh = pyf.fvMesh(runtime)

    t_field = volScalarField.read_field(mesh, "T")
    u_field = volVectorField.read_field(mesh, "U")
    gamma = volScalarField.read_field(mesh, "Gamma")
    phi = pyf.createPhi(u_field)

    results: dict[str, Any] = {
        "interpolate_T": lambda: fvc.interpolate(t_field),
        "flux_U": lambda: fvc.flux(u_field),
        "grad_T": lambda: fvc.grad(t_field),
        "div_phi": lambda: fvc.div(phi),
        "div_phi_T": lambda: fvc.div(fvc.flux(phi, t_field, key="div(phi,T)")),
        "div_phi_U": lambda: fvc.div(phi, u_field),
        "laplacian_Gamma_T": lambda: fvc.laplacian(gamma, t_field),
        "laplacian_Gamma_U": lambda: fvc.laplacian(gamma, u_field),
    }

    for op in ops:
        result = results[op]()
        np.save(out_dir / f"{op}.npy", np.asarray(result.ref().internalField()))


def main() -> None:
    case_dir, out_dir = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    compute_ops(case_dir, out_dir, sys.argv[3:])


if __name__ == "__main__":
    main()
