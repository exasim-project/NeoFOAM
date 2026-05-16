# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""SA solve script — run as subprocess with mode argument.

Usage:
    python sa_solve.py <mode> <test_dir> <output_file>

Modes:
    correct  — turbulence.correct()
    manual   — manual pybFoam fvm assembly
"""

import ctypes
import ctypes.util
import os
import signal
import sys
from pathlib import Path

os.environ["FOAM_SIGFPE"] = ""
signal.signal(signal.SIGFPE, signal.SIG_IGN)

import numpy as np
import pybFoam as pyf
from pybFoam import (
    fvc,
    fvm,
    fvScalarMatrix,
    volScalarField,
    volTensorField,
    volVectorField,
    wallDist,
)
from pybFoam.meshing import generate_blockmesh
from pybFoam.turbulence import incompressibleTurbulenceModel, singlePhaseTransportModel

libm = ctypes.CDLL(ctypes.util.find_library("m"))
libm.fedisableexcept(0x3F)

# SA constants
CFG_SIGMA = 2.0 / 3.0
CFG_CB1 = 0.1355
CFG_CB2 = 0.622
CFG_CW2 = 0.3
CFG_CW3 = 2.0
CFG_CV1 = 7.1
CFG_KAPPA = 0.41
CFG_CS = 0.3
CFG_CW1 = CFG_CB1 / CFG_KAPPA**2 + (1 + CFG_CB2) / CFG_SIGMA


def setup_fields(test_dir: str) -> None:
    """Write non-uniform analytical fields to 0/."""
    sys.path.insert(0, test_dir)
    from generate_fields import (
        compute_nuTilda,
        compute_velocity,
        compute_nut_from_nuTilda,
        write_scalar_field,
        write_vector_field,
        _read_boundary_block,
    )

    mesh = globals()["mesh"]
    cc = np.array(mesh.C().internalField())
    nu = 1e-5
    zero = Path("0")
    bc_U = _read_boundary_block(zero / "U")
    bc_nT = _read_boundary_block(zero / "nuTilda")
    bc_nut = _read_boundary_block(zero / "nut")
    write_vector_field(zero / "U", "U", "[0 1 -1 0 0 0 0]", compute_velocity(cc), bc_U)
    write_scalar_field(
        zero / "nuTilda", "nuTilda", "[0 2 -1 0 0 0 0]", compute_nuTilda(cc, nu=nu), bc_nT
    )
    write_scalar_field(
        zero / "nut",
        "nut",
        "[0 2 -1 0 0 0 0]",
        compute_nut_from_nuTilda(compute_nuTilda(cc, nu=nu), nu=nu),
        bc_nut,
    )


def solve_correct(turb: object, lam: object) -> None:
    """Run turbulence.correct() — the compiled reference."""
    lam.correct()
    turb.correct()


def solve_manual(nT: object, turb: object) -> None:
    """Manual SA using pybFoam fvm operators.

    Mirrors SpalartAllmarasBase::correct() from OpenFOAM.
    """
    nu_f = volScalarField(turb.nu())
    U = turb.U()
    mesh = nT.mesh()
    phi = turb.phi()
    d = wallDist.New(mesh).y()

    # chi, fv1, fv2
    chi = volScalarField(nT / nu_f)
    chi3 = volScalarField(pyf.pow3(chi))
    fv1 = volScalarField(chi3 / (chi3 + CFG_CV1**3))
    fv2 = volScalarField(
        -volScalarField(chi / (volScalarField(chi * fv1) + 1.0)) + 1.0
    )

    # Omega, Stilda
    gradU = volTensorField(fvc.grad(U))
    Omega = volScalarField(pyf.sqrt(2.0 * pyf.magSqr(pyf.skew(gradU))))
    kd2 = volScalarField(pyf.sqr(CFG_KAPPA * d))
    Stilda = volScalarField(
        pyf.max(
            volScalarField(Omega + volScalarField(fv2 * nT / kd2)),
            volScalarField(CFG_CS * Omega),
        )
    )

    # r, g, fw
    Ss = volScalarField(pyf.max(Stilda, 1e-10))
    r = volScalarField(pyf.min(volScalarField(nT / volScalarField(Ss * kd2)), 10.0))
    g = volScalarField(
        r + CFG_CW2 * volScalarField(volScalarField(pyf.pow6(r)) - r)
    )
    g6 = volScalarField(pyf.pow6(g))
    fw = volScalarField(
        g
        * volScalarField(
            pyf.pow(
                volScalarField(
                    (1.0 + CFG_CW3**6) / volScalarField(g6 + CFG_CW3**6)
                ),
                1.0 / 6.0,
            )
        )
    )

    # Diffusion coefficient and destruction
    DnuTildaEff = (nT + nu_f) / CFG_SIGMA
    d2 = volScalarField(pyf.sqr(d))

    # Equation assembly — mirrors OF's correct()
    #   LHS: ddt + div - laplacian - nonConsDiff
    #   RHS (==): production - Sp(destruction)
    # Converted to all-LHS: + Sp - Su
    eqn = fvScalarMatrix(
        fvm.ddt(nT)
        + fvm.div(phi, nT)
        - fvm.laplacian(DnuTildaEff, nT)
        - (CFG_CB2 / CFG_SIGMA) * pyf.magSqr(fvc.grad(nT))
        + fvm.Sp(CFG_CW1 * fw * nT / d2, nT)
        - fvm.Su(CFG_CB1 * Stilda * nT, nT)
    )
    eqn.relax()
    eqn.solve()
    pyf.bound(nT, pyf.dimensionedScalar("z", pyf.dimViscosity, 0.0))
    nT.correctBoundaryConditions()


def main() -> None:
    mode = sys.argv[1]
    test_dir = sys.argv[2]
    output_file = sys.argv[3]

    args = pyf.argList(["test"])
    rt = pyf.Time(args)
    libm.fedisableexcept(0x3F)

    bmd = pyf.dictionary.read("system/blockMeshDict")
    global mesh
    mesh = generate_blockmesh(rt, bmd)
    libm.fedisableexcept(0x3F)

    setup_fields(test_dir)

    U = volVectorField.read_field(mesh, "U")
    phi = pyf.createPhi(U)
    nT = volScalarField.read_field(mesh, "nuTilda")
    nut = volScalarField.read_field(mesh, "nut")
    lam = singlePhaseTransportModel(U, phi)
    turb = incompressibleTurbulenceModel.New(U, phi, lam)

    if mode == "correct":
        solve_correct(turb, lam)
    elif mode == "manual":
        solve_manual(nT, turb)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    result = np.array(nT.internalField())
    np.save(output_file, result)


if __name__ == "__main__":
    main()
