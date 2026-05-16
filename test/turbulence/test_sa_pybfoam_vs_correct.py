# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Compare SA implementations: compiled OF, pybFoam manual, and NeoN.

Standalone — does NOT use conftest fixtures.
Sets up its own case with deltaT=0.01 and non-uniform fields.
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
from neofoam.solver.incompressibleFluid.models.spalartAllmaras import (
    SpalartAllmarasConfig,
)
from neofoam.solver.incompressibleFluidNeon.models.turbulence.spalartAllmaras import (
    correct as sa_correct_neon,
)

sys.path.insert(0, str(Path(__file__).parent))
from generate_fields import (
    compute_nuTilda,
    compute_velocity,
    compute_nut_from_nuTilda,
    write_scalar_field,
    write_vector_field,
    _read_boundary_block,
)
from sa_pybfoam import correct as sa_correct_pybfoam

CASE_SOURCE = Path(__file__).parent / "pitzDaily_SA"
CASE_DIR = Path(__file__).parent.parent.parent / "test_cases" / "sa_pybfoam_vs_correct"


def _disable_fpe() -> None:
    libm = ctypes.CDLL(ctypes.util.find_library("m"))
    libm.fedisableexcept(0x3F)


def _of(field: Any) -> np.ndarray:
    return np.array(field.internalField())


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
def sa_case() -> Any:
    """Fresh SA case with non-uniform fields and dt=0.01."""
    original_dir = Path.cwd()

    if CASE_DIR.exists():
        shutil.rmtree(CASE_DIR)
    CASE_DIR.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(CASE_SOURCE, CASE_DIR)

    # deltaT=0.01 so ddt doesn't dominate
    cd = CASE_DIR / "system" / "controlDict"
    text = cd.read_text()
    text = text.replace("deltaT          0.0001;", "deltaT          0.01;")
    cd.write_text(text)

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
    write_scalar_field(
        zero / "nuTilda", "nuTilda", "[0 2 -1 0 0 0 0]",
        compute_nuTilda(cc, nu=nu), _read_boundary_block(zero / "nuTilda"),
    )
    write_scalar_field(
        zero / "nut", "nut", "[0 2 -1 0 0 0 0]",
        compute_nut_from_nuTilda(compute_nuTilda(cc, nu=nu), nu=nu),
        _read_boundary_block(zero / "nut"),
    )

    U = volVectorField.read_field(mesh, "U")
    phi = pyf.createPhi(U)
    lam = singlePhaseTransportModel(U, phi)
    turb = incompressibleTurbulenceModel.New(U, phi, lam)
    nT = volScalarField.from_registry(mesh, "nuTilda")

    # NeoN runtime and fields (nn.initialize already called by conftest neon_session)
    _disable_fpe()
    rt_nn = nfb.create_adapter_run_time(rt)
    solvers = rt_nn.fv_solution_dict.subDict("solvers")
    solvers.insert_dict("p", nfb.map_fv_solution(solvers.subDict("p")))
    solvers.insert_dict("U", nfb.map_fv_solution(solvers.subDict("U")))
    solvers.insert_dict("nuTilda", nfb.map_fv_solution(solvers.subDict("nuTilda")))
    rt_nn.fv_schemes_dict = nfb.map_fv_schemes(rt_nn.fv_schemes_dict)

    nuTilda_nn = nfb.read_scalar_volume_field(rt_nn, "nuTilda")
    nut_nn = nfb.read_scalar_volume_field(rt_nn, "nut")
    U_nn = nfb.read_vector_volume_field(rt_nn, "U")
    phi_nn = nfb.create_phi(rt_nn, "U")
    d_nn = nfb.compute_wall_distance(rt_nn)
    nu_value = float(nfb.read_transport_viscosity(rt_nn))

    yield {
        "nT": nT, "turb": turb, "lam": lam, "mesh": mesh,
        "rt_nn": rt_nn, "nuTilda_nn": nuTilda_nn, "nut_nn": nut_nn,
        "U_nn": U_nn, "phi_nn": phi_nn, "d_nn": d_nn, "nu_value": nu_value,
    }

    os.chdir(original_dir)
    if CASE_DIR.exists():
        shutil.rmtree(CASE_DIR)


def _nn(field: Any) -> np.ndarray:
    return np.asarray(field.internal_vector().__array__())


def test_all_sa_implementations(sa_case: dict[str, Any]) -> None:
    """Compare all three SA implementations from the same initial state.

    1. turbulence.correct() (compiled OF) — reference
    2. sa_pybfoam.correct() (pybFoam fvm) — validated manual assembly
    3. sa_correct_neon (NeoN imp) — under test
    """
    nT = sa_case["nT"]
    turb = sa_case["turb"]
    lam = sa_case["lam"]
    cfg = SpalartAllmarasConfig()

    initial = _of(nT).copy()
    saved = volScalarField(nT)

    # --- 1. Compiled correct() ---
    lam.correct()
    turb.correct()
    result_correct = _of(nT).copy()
    print(f"\n  correct() change: {np.max(np.abs(result_correct - initial)):.6e}")

    # --- 2. pybFoam manual ---
    nT.assign(saved)
    sa_correct_pybfoam(nT, turb)
    result_pybfoam = _of(nT).copy()

    # --- 3. NeoN ---
    sa_correct_neon(
        cfg, sa_case["rt_nn"],
        sa_case["nuTilda_nn"], sa_case["nut_nn"],
        sa_case["U_nn"], sa_case["phi_nn"],
        sa_case["d_nn"], sa_case["nu_value"],
    )
    result_neon = _nn(sa_case["nuTilda_nn"]).copy()

    # --- Compare ---
    _report("correct() vs pybFoam", result_correct, result_pybfoam, "correct()", "pybfoam")
    _report("correct() vs NeoN", result_correct, result_neon, "correct()", "NeoN")
    _report("pybFoam vs NeoN", result_pybfoam, result_neon, "pybfoam", "NeoN")

    np.testing.assert_allclose(
        result_pybfoam, result_correct, rtol=1e-4, atol=1e-10,
        err_msg="pybFoam SA does not match compiled correct()",
    )
    np.testing.assert_allclose(
        result_neon, result_correct, rtol=1e-2, atol=1e-10,
        err_msg="NeoN SA does not match compiled correct()",
    )
