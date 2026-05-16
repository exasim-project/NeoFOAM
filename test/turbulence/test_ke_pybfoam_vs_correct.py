# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Validate pybFoam kEpsilon against turbulence.correct(), then compare NeoN.

Same pattern as test_sa_pybfoam_vs_correct.py.
Standalone — does NOT use conftest fixtures.
"""

import ctypes
import ctypes.util
import os
import shutil
import signal
import sys
from pathlib import Path
from typing import Any

os.environ["FOAM_SIGFPE"] = ""
signal.signal(signal.SIGFPE, signal.SIG_IGN)

import numpy as np
import pytest

import neon._neon as nn
import pybFoam as pyf
from pybFoam import volScalarField, volVectorField
from pybFoam.meshing import generate_blockmesh
from pybFoam.turbulence import incompressibleTurbulenceModel, singlePhaseTransportModel

from neofoam import neofoam_bindings as nfb
from neofoam.solver.incompressibleFluidNeon.models.turbulence.kEpsilon import (
    KEpsilonConfig,
    correct as ke_correct_neon,
)

sys.path.insert(0, str(Path(__file__).parent))
from generate_fields import (
    compute_k,
    compute_epsilon,
    compute_velocity,
    compute_nut_from_k_epsilon,
    write_scalar_field,
    write_vector_field,
    _read_boundary_block,
)
from ke_pybfoam import correct as ke_correct_pybfoam

CASE_SOURCE = Path(__file__).parent / "turbTest"
CASE_DIR = Path(__file__).parent.parent.parent / "test_cases" / "ke_pybfoam_vs_correct"

KE_TURB_PROPS = """\
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      turbulenceProperties;
}
simulationType      RAS;
RAS
{
    RASModel        kEpsilon;
    turbulence      on;
    printCoeffs     on;
}
"""


def _disable_fpe() -> None:
    libm = ctypes.CDLL(ctypes.util.find_library("m"))
    libm.fedisableexcept(0x3F)


def _of(f: Any) -> np.ndarray:
    return np.array(f.internalField())


def _nn(f: Any) -> np.ndarray:
    return np.asarray(f.internal_vector().__array__())


def _report(name: str, a: np.ndarray, b: np.ndarray, la: str, lb: str) -> None:
    abs_diff = np.max(np.abs(a - b))
    denom = np.max(np.abs(a)) + 1e-30
    w1 = np.sum(np.abs(a - b) / (np.abs(a) + 1e-30) < 0.01)
    print(f"\n  {name}:")
    print(f"    {la}: min={a.min():.6e}  max={a.max():.6e}  mean={a.mean():.6e}")
    print(f"    {lb}: min={b.min():.6e}  max={b.max():.6e}  mean={b.mean():.6e}")
    print(f"    diff: abs={abs_diff:.6e}  rel={abs_diff / denom:.6e}")
    print(f"    within 1%: {w1}/{len(a)} ({100 * w1 / len(a):.1f}%)")


@pytest.fixture(scope="session")
def ke_case() -> Any:
    """Fresh kEpsilon case with non-uniform fields and dt=0.01."""
    original_dir = Path.cwd()

    if CASE_DIR.exists():
        shutil.rmtree(CASE_DIR)
    CASE_DIR.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(CASE_SOURCE, CASE_DIR)

    # Write kEpsilon turbulenceProperties
    (CASE_DIR / "constant" / "turbulenceProperties").write_text(KE_TURB_PROPS)

    os.chdir(CASE_DIR)
    _disable_fpe()
    args = pyf.argList(["test"])
    rt = pyf.Time(args)
    _disable_fpe()
    mesh = generate_blockmesh(rt, pyf.dictionary.read("system/blockMeshDict"))
    _disable_fpe()

    # Write non-uniform analytical fields
    cc = np.array(mesh.C().internalField())
    nu = 1e-5
    zero = CASE_DIR / "0"
    write_vector_field(
        zero / "U", "U", "[0 1 -1 0 0 0 0]",
        compute_velocity(cc), _read_boundary_block(zero / "U"),
    )
    k_vals = compute_k(cc, nu=nu)
    eps_vals = compute_epsilon(cc, nu=nu)
    nut_vals = compute_nut_from_k_epsilon(k_vals, eps_vals)
    write_scalar_field(zero / "k", "k", "[0 2 -2 0 0 0 0]",
                       k_vals, _read_boundary_block(zero / "k"))
    write_scalar_field(zero / "epsilon", "epsilon", "[0 2 -3 0 0 0 0]",
                       eps_vals, _read_boundary_block(zero / "epsilon"))
    write_scalar_field(zero / "nut", "nut", "[0 2 -1 0 0 0 0]",
                       nut_vals, _read_boundary_block(zero / "nut"))

    # OF fields
    U_of = volVectorField.read_field(mesh, "U")
    phi_of = pyf.createPhi(U_of)
    lam = singlePhaseTransportModel(U_of, phi_of)
    turb = incompressibleTurbulenceModel.New(U_of, phi_of, lam)
    k_of = volScalarField.from_registry(mesh, "k")
    eps_of = volScalarField.from_registry(mesh, "epsilon")
    nut_of = volScalarField.from_registry(mesh, "nut")

    # NeoN fields
    _disable_fpe()
    rt_nn = nfb.create_adapter_run_time(rt)
    solvers = rt_nn.fv_solution_dict.subDict("solvers")
    for name in ["p", "U", "k", "epsilon"]:
        if solvers.contains(name):
            solvers.insert_dict(name, nfb.map_fv_solution(solvers.subDict(name)))
    rt_nn.fv_schemes_dict = nfb.map_fv_schemes(rt_nn.fv_schemes_dict)

    yield {
        "k_of": k_of, "eps_of": eps_of, "nut_of": nut_of,
        "turb": turb, "lam": lam, "mesh": mesh,
        "rt_nn": rt_nn,
        "k_nn": nfb.read_scalar_volume_field(rt_nn, "k"),
        "epsilon_nn": nfb.read_scalar_volume_field(rt_nn, "epsilon"),
        "nut_nn": nfb.read_scalar_volume_field(rt_nn, "nut"),
        "U_nn": nfb.read_vector_volume_field(rt_nn, "U"),
        "phi_nn": nfb.create_phi(rt_nn, "U"),
        "nu_value": float(nfb.read_transport_viscosity(rt_nn)),
    }

    os.chdir(original_dir)
    if CASE_DIR.exists():
        shutil.rmtree(CASE_DIR)


def test_pybfoam_ke_matches_compiled(ke_case: dict[str, Any]) -> None:
    """Step 1: pybFoam kEpsilon must match turbulence.correct()."""
    k_of = ke_case["k_of"]
    eps_of = ke_case["eps_of"]
    nut_of = ke_case["nut_of"]
    turb = ke_case["turb"]
    lam = ke_case["lam"]

    # Save initial
    k_saved = volScalarField(k_of)
    eps_saved = volScalarField(eps_of)
    nut_saved = volScalarField(nut_of)
    k_init = _of(k_of).copy()
    eps_init = _of(eps_of).copy()

    # Run compiled correct()
    lam.correct()
    turb.correct()
    k_correct = _of(k_of).copy()
    eps_correct = _of(eps_of).copy()
    print(f"\n  correct() k change: {np.max(np.abs(k_correct - k_init)):.6e}")
    print(f"  correct() eps change: {np.max(np.abs(eps_correct - eps_init)):.6e}")

    # Restore
    k_of.assign(k_saved)
    eps_of.assign(eps_saved)
    nut_of.assign(nut_saved)

    # Run pybFoam kEpsilon
    ke_correct_pybfoam(k_of, eps_of, nut_of, turb)
    k_pybfoam = _of(k_of).copy()
    eps_pybfoam = _of(eps_of).copy()

    _report("k: correct() vs pybFoam", k_correct, k_pybfoam, "correct()", "pybfoam")
    _report("eps: correct() vs pybFoam", eps_correct, eps_pybfoam, "correct()", "pybfoam")

    np.testing.assert_allclose(k_pybfoam, k_correct, rtol=1e-4, atol=1e-10,
                               err_msg="pybFoam k does not match correct()")
    np.testing.assert_allclose(eps_pybfoam, eps_correct, rtol=1e-4, atol=1e-10,
                               err_msg="pybFoam epsilon does not match correct()")


def test_neon_ke_matches_compiled(ke_case: dict[str, Any]) -> None:
    """Step 2: NeoN kEpsilon vs turbulence.correct()."""
    k_of = ke_case["k_of"]
    eps_of = ke_case["eps_of"]
    nut_of = ke_case["nut_of"]
    turb = ke_case["turb"]
    lam = ke_case["lam"]
    cfg = KEpsilonConfig()

    # Save initial
    k_saved = volScalarField(k_of)
    eps_saved = volScalarField(eps_of)
    nut_saved = volScalarField(nut_of)

    # Run compiled correct()
    lam.correct()
    turb.correct()
    k_correct = _of(k_of).copy()
    eps_correct = _of(eps_of).copy()

    # Restore OF fields
    k_of.assign(k_saved)
    eps_of.assign(eps_saved)
    nut_of.assign(nut_saved)

    # Run NeoN correct()
    ke_correct_neon(
        cfg, ke_case["rt_nn"],
        ke_case["k_nn"], ke_case["epsilon_nn"], ke_case["nut_nn"],
        ke_case["U_nn"], ke_case["phi_nn"], ke_case["nu_value"],
    )
    k_neon = _nn(ke_case["k_nn"]).copy()
    eps_neon = _nn(ke_case["epsilon_nn"]).copy()

    _report("k: correct() vs NeoN", k_correct, k_neon, "correct()", "NeoN")
    _report("eps: correct() vs NeoN", eps_correct, eps_neon, "correct()", "NeoN")

    np.testing.assert_allclose(k_neon, k_correct, rtol=1e-2, atol=1e-10,
                               err_msg="NeoN k does not match correct()")
    np.testing.assert_allclose(eps_neon, eps_correct, rtol=1e-2, atol=1e-10,
                               err_msg="NeoN epsilon does not match correct()")
