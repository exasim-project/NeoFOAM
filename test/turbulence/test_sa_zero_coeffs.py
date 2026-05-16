# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Narrow down SA discrepancy by zeroing Cb1/Cb2 coefficients.

Modifies turbulenceProperties to override SA constants, then compares
OF turbulence.correct() vs NeoN correct() with the same zeroed constants.
This isolates which source terms cause the mismatch.

Runs as standalone (not using the session fixture) to get a clean OF state.
"""

import os
import shutil
import signal
from pathlib import Path

os.environ["FOAM_SIGFPE"] = ""
signal.signal(signal.SIGFPE, signal.SIG_IGN)

import ctypes
import ctypes.util
from typing import Any

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
    correct as sa_correct,
)

import sys

sys.path.insert(0, str(Path(__file__).parent))
from generate_fields import (  # noqa: E402
    compute_nuTilda,
    compute_velocity,
    compute_nut_from_nuTilda,
    write_scalar_field,
    write_vector_field,
    _read_boundary_block,
)

CASE_SOURCE = Path(__file__).parent / "pitzDaily_SA"


def _disable_fpe() -> None:
    libm = ctypes.CDLL(ctypes.util.find_library("m"))
    libm.fedisableexcept(0x3F)


def _of_internal(field: Any) -> np.ndarray:
    return np.array(field.internalField())


def _nn_internal(field: Any) -> np.ndarray:
    return np.asarray(field.internal_vector().__array__())


def _report(name: str, a: np.ndarray, b: np.ndarray, la: str = "OF", lb: str = "NeoN") -> None:
    abs_diff = np.max(np.abs(a - b))
    denom = np.max(np.abs(a)) + 1e-30
    print(f"\n  {name}:")
    print(f"    {la}: min={a.min():.6e}  max={a.max():.6e}  mean={a.mean():.6e}")
    print(f"    {lb}: min={b.min():.6e}  max={b.max():.6e}  mean={b.mean():.6e}")
    print(f"    diff: abs={abs_diff:.6e}  rel={abs_diff / denom:.6e}")


def _setup_case(
    test_case: Path,
    *,
    Cb1: float = 0.1355,
    Cb2: float = 0.622,
) -> None:
    """Copy case and override SA coefficients in turbulenceProperties."""
    if test_case.exists():
        shutil.rmtree(test_case)
    test_case.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(CASE_SOURCE, test_case)

    # Override coefficients in turbulenceProperties
    tp = test_case / "constant" / "turbulenceProperties"
    tp.write_text(f"""\
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      turbulenceProperties;
}}

simulationType      RAS;

RAS
{{
    RASModel        SpalartAllmaras;
    turbulence      on;
    printCoeffs     on;

    // Overridden coefficients
    Cb1             {Cb1};
    Cb2             {Cb2};
}}
""")


def _run_comparison(
    test_case: Path,
    cfg: SpalartAllmarasConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Run both backends and return (of_result, nn_result)."""
    original_dir = Path.cwd()
    os.chdir(test_case)
    _disable_fpe()

    try:
        args = pyf.argList(["test"])
        run_time = pyf.Time(args)
        _disable_fpe()
        block_mesh_dict = pyf.dictionary.read("system/blockMeshDict")
        mesh = generate_blockmesh(run_time, block_mesh_dict)
        _disable_fpe()

        # Generate non-uniform fields
        cc = np.array(mesh.C().internalField())
        nu = 1e-5
        zero_dir = test_case / "0"
        bc_U = _read_boundary_block(zero_dir / "U")
        bc_nuTilda = _read_boundary_block(zero_dir / "nuTilda")
        bc_nut = _read_boundary_block(zero_dir / "nut")
        nuTilda_vals = compute_nuTilda(cc, nu=nu)
        U_vals = compute_velocity(cc)
        nut_vals = compute_nut_from_nuTilda(nuTilda_vals, nu=nu)
        write_vector_field(zero_dir / "U", "U", "[0 1 -1 0 0 0 0]", U_vals, bc_U)
        write_scalar_field(zero_dir / "nuTilda", "nuTilda", "[0 2 -1 0 0 0 0]", nuTilda_vals, bc_nuTilda)
        write_scalar_field(zero_dir / "nut", "nut", "[0 2 -1 0 0 0 0]", nut_vals, bc_nut)

        # --- pybFoam: turbulence.correct() reads overridden coefficients ---
        U_of = volVectorField.read_field(mesh, "U")
        phi_of = pyf.createPhi(U_of)
        nuTilda_of = volScalarField.read_field(mesh, "nuTilda")
        nut_of = volScalarField.read_field(mesh, "nut")
        lam = singlePhaseTransportModel(U_of, phi_of)
        turb = incompressibleTurbulenceModel.New(U_of, phi_of, lam)
        lam.correct()
        turb.correct()
        of_result = _of_internal(nuTilda_of).copy()

        # --- NeoN: correct() with matching config ---
        rt = nfb.create_adapter_run_time(run_time)
        solvers = rt.fv_solution_dict.subDict("solvers")
        solvers.insert_dict("p", nfb.map_fv_solution(solvers.subDict("p")))
        solvers.insert_dict("U", nfb.map_fv_solution(solvers.subDict("U")))
        solvers.insert_dict("nuTilda", nfb.map_fv_solution(solvers.subDict("nuTilda")))
        rt.fv_schemes_dict = nfb.map_fv_schemes(rt.fv_schemes_dict)

        nuTilda_nn = nfb.read_scalar_volume_field(rt, "nuTilda")
        nut_nn = nfb.read_scalar_volume_field(rt, "nut")
        U_nn = nfb.read_vector_volume_field(rt, "U")
        phi_nn = nfb.create_phi(rt, "U")
        d_nn = nfb.compute_wall_distance(rt)
        nu_value = float(nfb.read_transport_viscosity(rt))

        sa_correct(cfg, rt, nuTilda_nn, nut_nn, U_nn, phi_nn, d_nn, nu_value)
        nn_result = _nn_internal(nuTilda_nn).copy()

        return of_result, nn_result
    finally:
        os.chdir(original_dir)


# ---------------------------------------------------------------------------
# Test 1: Cb1=0, Cb2=0 → no production, no nonConsDiff
#   OF equation becomes: ddt + div - laplacian == -Sp(Cw1*fw*nuTilda/d², nuTilda)
#   Cw1 = Cb1/kappa² + (1+Cb2)/sigma = 0 + 1/0.6666 = 1.5
# ---------------------------------------------------------------------------


def test_Cb1_zero_Cb2_zero() -> None:
    """With Cb1=Cb2=0: no production, no nonConsDiff, reduced Cw1."""
    test_case = Path(__file__).parent.parent.parent / "test_cases" / "sa_Cb1_0_Cb2_0"
    cfg = SpalartAllmarasConfig(Cb1=0.0, Cb2=0.0)

    try:
        _setup_case(test_case, Cb1=0.0, Cb2=0.0)
        of_result, nn_result = _run_comparison(test_case, cfg)

        _report("Cb1=0 Cb2=0", of_result, nn_result)

        match_1pct = np.sum(np.abs(nn_result - of_result) / (np.abs(of_result) + 1e-30) < 0.01)
        print(f"    within 1%: {match_1pct}/{len(of_result)} ({100*match_1pct/len(of_result):.1f}%)")

        np.testing.assert_allclose(nn_result, of_result, rtol=1e-3, atol=1e-10,
                                   err_msg="Cb1=0 Cb2=0: results differ")
    finally:
        if test_case.exists():
            shutil.rmtree(test_case)


# ---------------------------------------------------------------------------
# Test 2: Cb2=0 only → production active, no nonConsDiff
# ---------------------------------------------------------------------------


def test_Cb2_zero() -> None:
    """With Cb2=0: production active, no nonConsDiff, Cw1 = Cb1/kappa² + 1/sigma."""
    test_case = Path(__file__).parent.parent.parent / "test_cases" / "sa_Cb2_0"
    cfg = SpalartAllmarasConfig(Cb2=0.0)

    try:
        _setup_case(test_case, Cb2=0.0)
        of_result, nn_result = _run_comparison(test_case, cfg)

        _report("Cb2=0", of_result, nn_result)

        match_1pct = np.sum(np.abs(nn_result - of_result) / (np.abs(of_result) + 1e-30) < 0.01)
        print(f"    within 1%: {match_1pct}/{len(of_result)} ({100*match_1pct/len(of_result):.1f}%)")

        np.testing.assert_allclose(nn_result, of_result, rtol=1e-3, atol=1e-10,
                                   err_msg="Cb2=0: results differ")
    finally:
        if test_case.exists():
            shutil.rmtree(test_case)


# ---------------------------------------------------------------------------
# Test 3: Cb1=0 only → no production, nonConsDiff active
# ---------------------------------------------------------------------------


def test_Cb1_zero() -> None:
    """With Cb1=0: no production, nonConsDiff active, Cw1 = (1+Cb2)/sigma."""
    test_case = Path(__file__).parent.parent.parent / "test_cases" / "sa_Cb1_0"
    cfg = SpalartAllmarasConfig(Cb1=0.0)

    try:
        _setup_case(test_case, Cb1=0.0)
        of_result, nn_result = _run_comparison(test_case, cfg)

        _report("Cb1=0", of_result, nn_result)

        match_1pct = np.sum(np.abs(nn_result - of_result) / (np.abs(of_result) + 1e-30) < 0.01)
        print(f"    within 1%: {match_1pct}/{len(of_result)} ({100*match_1pct/len(of_result):.1f}%)")

        np.testing.assert_allclose(nn_result, of_result, rtol=1e-3, atol=1e-10,
                                   err_msg="Cb1=0: results differ")
    finally:
        if test_case.exists():
            shutil.rmtree(test_case)


# ---------------------------------------------------------------------------
# Test 4: Default coefficients (full SA)
# ---------------------------------------------------------------------------


def test_default_coeffs() -> None:
    """Full SA with default coefficients — baseline comparison."""
    test_case = Path(__file__).parent.parent.parent / "test_cases" / "sa_default"
    cfg = SpalartAllmarasConfig()

    try:
        _setup_case(test_case)
        of_result, nn_result = _run_comparison(test_case, cfg)

        _report("default", of_result, nn_result)

        match_1pct = np.sum(np.abs(nn_result - of_result) / (np.abs(of_result) + 1e-30) < 0.01)
        print(f"    within 1%: {match_1pct}/{len(of_result)} ({100*match_1pct/len(of_result):.1f}%)")
    finally:
        if test_case.exists():
            shutil.rmtree(test_case)
