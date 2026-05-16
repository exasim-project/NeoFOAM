# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Minimal laplacian debug: compare ddt+lap for nuTilda vs epsilon."""

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
from pybFoam import fvm, fvScalarMatrix, volScalarField, volVectorField
from pybFoam.meshing import generate_blockmesh
from pybFoam.turbulence import incompressibleTurbulenceModel, singlePhaseTransportModel
from neofoam import neofoam_bindings as nfb

sys.path.insert(0, str(Path(__file__).parent))
from generate_fields import (
    compute_k, compute_epsilon, compute_velocity, compute_nuTilda,
    compute_nut_from_nuTilda, compute_nut_from_k_epsilon,
    write_scalar_field, write_vector_field, _read_boundary_block,
)

CASE_SOURCE = Path(__file__).parent / "turbTest"
CASE_DIR = Path(__file__).parent.parent.parent / "test_cases" / "ke_lap_debug"


def _disable_fpe() -> None:
    libm = ctypes.CDLL(ctypes.util.find_library("m"))
    libm.fedisableexcept(0x3F)


@pytest.fixture(scope="session")
def env() -> Any:
    """Setup kEpsilon case."""
    original_dir = Path.cwd()
    if CASE_DIR.exists():
        shutil.rmtree(CASE_DIR)
    CASE_DIR.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(CASE_SOURCE, CASE_DIR)
    (CASE_DIR / "constant" / "turbulenceProperties").write_text(
        "FoamFile\n{\n    version 2.0;\n    format ascii;\n    class dictionary;\n"
        "    object turbulenceProperties;\n}\n"
        "simulationType RAS;\nRAS\n{\n    RASModel kEpsilon;\n    turbulence on;\n"
        "    printCoeffs on;\n}\n"
    )
    os.chdir(CASE_DIR); _disable_fpe()
    args = pyf.argList(["test"]); rt = pyf.Time(args); _disable_fpe()
    mesh = generate_blockmesh(rt, pyf.dictionary.read("system/blockMeshDict")); _disable_fpe()
    cc = np.array(mesh.C().internalField()); nu = 1e-5; zero = CASE_DIR / "0"
    write_vector_field(zero / "U", "U", "[0 1 -1 0 0 0 0]",
                       compute_velocity(cc), _read_boundary_block(zero / "U"))
    write_scalar_field(zero / "k", "k", "[0 2 -2 0 0 0 0]",
                       compute_k(cc, nu), _read_boundary_block(zero / "k"))
    write_scalar_field(zero / "epsilon", "epsilon", "[0 2 -3 0 0 0 0]",
                       compute_epsilon(cc, nu), _read_boundary_block(zero / "epsilon"))
    write_scalar_field(zero / "nut", "nut", "[0 2 -1 0 0 0 0]",
                       compute_nut_from_k_epsilon(compute_k(cc, nu), compute_epsilon(cc, nu)),
                       _read_boundary_block(zero / "nut"))
    # Also write nuTilda so we can test with SA-style field
    write_scalar_field(zero / "nuTilda", "nuTilda", "[0 2 -1 0 0 0 0]",
                       compute_nuTilda(cc, nu), _read_boundary_block(zero / "nuTilda"))

    U = volVectorField.read_field(mesh, "U"); phi = pyf.createPhi(U)
    lam = singlePhaseTransportModel(U, phi)
    turb = incompressibleTurbulenceModel.New(U, phi, lam)
    _disable_fpe()
    rt_nn = nfb.create_adapter_run_time(rt)
    solvers = rt_nn.fv_solution_dict.subDict("solvers")
    for n in ["p", "U", "k", "epsilon", "nuTilda"]:
        if solvers.contains(n):
            solvers.insert_dict(n, nfb.map_fv_solution(solvers.subDict(n)))
    rt_nn.fv_schemes_dict = nfb.map_fv_schemes(rt_nn.fv_schemes_dict)

    yield {
        "mesh": mesh, "turb": turb, "phi_of": pyf.surfaceScalarField(turb.phi()),
        "rt_nn": rt_nn, "phi_nn": nfb.create_phi(rt_nn, "U"), "nu": nu,
    }
    os.chdir(original_dir)
    if CASE_DIR.exists():
        shutil.rmtree(CASE_DIR)


def _solve_transport(env: dict[str, Any], field_name: str, coeff_val: float) -> tuple[np.ndarray, np.ndarray]:
    """Solve ddt + div - laplacian(const, field) with both OF and NeoN."""
    mesh = env["mesh"]; rt = env["rt_nn"]

    f_of = volScalarField.from_registry(mesh, field_name)
    f_nn = nfb.read_scalar_volume_field(rt, field_name)

    nu_ds = pyf.dimensionedScalar("nu", pyf.dimViscosity, coeff_val)
    nu_surf = nfb.create_uniform_surface_field(rt, f"nu_{field_name}", coeff_val)

    # OF
    saved = volScalarField(f_of)
    fvScalarMatrix(fvm.ddt(f_of) + fvm.div(env["phi_of"], f_of)
                   - fvm.laplacian(nu_ds, f_of)).solve()
    of_arr = np.array(f_of.internalField()).copy()
    f_of.assign(saved)

    # NeoN
    nn.rotate_old_times(f_nn)
    f_nn.correct_boundary_conditions()
    nfb.PDESolverScalar(
        nn.imp.ddt(f_nn) + nn.imp.div(env["phi_nn"], f_nn)
        - nn.imp.laplacian(nu_surf, f_nn), f_nn, rt,
    ).solve()
    nn_arr = np.asarray(f_nn.internal_vector().__array__()).copy()

    return of_arr, nn_arr


def test_transport_comparison(env: dict[str, Any]) -> None:
    """Compare ddt+div-lap(const) for k, epsilon, and nuTilda."""
    nu = env["nu"]
    print()
    # Also test with field-valued coefficient
    print("\n  With field-valued coefficient (field/1.3):")
    for field_name in ["k", "epsilon"]:
        try:
            f_of = volScalarField.from_registry(env["mesh"], field_name)
            f_nn = nfb.read_scalar_volume_field(env["rt_nn"], field_name)
            coeff_of = f_of / 1.3
            coeff_nn = f_nn / 1.3
            interp = nn.SurfaceInterpolationScalar(env["rt_nn"].executor, env["rt_nn"].nf_mesh, nn.TokenList(["linear"]))
            coeff_f = interp.interpolate(coeff_nn)

            saved = volScalarField(f_of)
            fvScalarMatrix(fvm.ddt(f_of) + fvm.div(env["phi_of"], f_of)
                           - fvm.laplacian(coeff_of, f_of)).solve()
            of_arr = np.array(f_of.internalField()).copy()
            f_of.assign(saved)

            nn.rotate_old_times(f_nn)
            f_nn.correct_boundary_conditions()
            nfb.PDESolverScalar(
                nn.imp.ddt(f_nn) + nn.imp.div(env["phi_nn"], f_nn)
                - nn.imp.laplacian(coeff_f, f_nn), f_nn, env["rt_nn"],
            ).solve()
            nn_arr = np.asarray(f_nn.internal_vector().__array__()).copy()

            d = np.max(np.abs(nn_arr - of_arr))
            r = d / (np.max(np.abs(of_arr)) + 1e-30)
            w = np.sum(np.abs(nn_arr - of_arr) / (np.abs(of_arr) + 1e-30) < 0.01)
            print(f"  {field_name:10s}: abs={d:.2e}  rel={r:.2e}  "
                  f"within1%={w}/{len(of_arr)} ({100*w/len(of_arr):.1f}%)")
        except Exception as e:
            print(f"  {field_name:10s}: ERROR — {e}")

    # Also test with nut/sigmaEps (different field as coefficient)
    print("\n  With nut/sigmaEps coefficient:")
    nut_of = volScalarField.from_registry(env["mesh"], "nut")
    nut_nn = nfb.read_scalar_volume_field(env["rt_nn"], "nut")
    for field_name in ["k", "epsilon"]:
        try:
            f_of = volScalarField.from_registry(env["mesh"], field_name)
            f_nn = nfb.read_scalar_volume_field(env["rt_nn"], field_name)
            coeff_of = nut_of / 1.3
            coeff_nn = nut_nn / 1.3
            interp = nn.SurfaceInterpolationScalar(env["rt_nn"].executor, env["rt_nn"].nf_mesh, nn.TokenList(["linear"]))
            coeff_f = interp.interpolate(coeff_nn)

            saved = volScalarField(f_of)
            fvScalarMatrix(fvm.ddt(f_of) + fvm.div(env["phi_of"], f_of)
                           - fvm.laplacian(coeff_of, f_of)).solve()
            of_arr = np.array(f_of.internalField()).copy()
            f_of.assign(saved)

            nn.rotate_old_times(f_nn)
            f_nn.correct_boundary_conditions()
            nfb.PDESolverScalar(
                nn.imp.ddt(f_nn) + nn.imp.div(env["phi_nn"], f_nn)
                - nn.imp.laplacian(coeff_f, f_nn), f_nn, env["rt_nn"],
            ).solve()
            nn_arr = np.asarray(f_nn.internal_vector().__array__()).copy()

            d = np.max(np.abs(nn_arr - of_arr))
            r = d / (np.max(np.abs(of_arr)) + 1e-30)
            w = np.sum(np.abs(nn_arr - of_arr) / (np.abs(of_arr) + 1e-30) < 0.01)
            print(f"  {field_name:10s}: abs={d:.2e}  rel={r:.2e}  "
                  f"within1%={w}/{len(of_arr)} ({100*w/len(of_arr):.1f}%)")
        except Exception as e:
            print(f"  {field_name:10s}: ERROR — {e}")

    print("\n  With constant nu:")
    for field_name in ["k", "epsilon"]:
        try:
            of_arr, nn_arr = _solve_transport(env, field_name, nu)
            d = np.max(np.abs(nn_arr - of_arr))
            r = d / (np.max(np.abs(of_arr)) + 1e-30)
            w = np.sum(np.abs(nn_arr - of_arr) / (np.abs(of_arr) + 1e-30) < 0.01)
            print(f"  {field_name:10s}: abs={d:.2e}  rel={r:.2e}  "
                  f"within1%={w}/{len(of_arr)} ({100*w/len(of_arr):.1f}%)")
        except Exception as e:
            print(f"  {field_name:10s}: ERROR — {e}")
