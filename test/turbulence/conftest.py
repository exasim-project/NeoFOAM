# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

import os
import shutil
import signal
from pathlib import Path
from typing import Any, Generator

# Disable OpenFOAM FPE trapping BEFORE importing any OpenFOAM bindings
os.environ["FOAM_SIGFPE"] = ""
signal.signal(signal.SIGFPE, signal.SIG_IGN)

import ctypes
import ctypes.util

import numpy as np
import pytest

import neon._neon as nn
import pybFoam as pyf
from pybFoam import volScalarField, volVectorField
from pybFoam.meshing import generate_blockmesh
from pybFoam.turbulence import incompressibleTurbulenceModel, singlePhaseTransportModel

from neofoam import neofoam_bindings as nfb

import sys

# Ensure test/turbulence/ is on the path for local imports
sys.path.insert(0, str(Path(__file__).parent))

from generate_fields import (  # noqa: E402
    compute_nuTilda,
    compute_velocity,
    compute_nut_from_nuTilda,
    write_scalar_field,
    write_vector_field,
    _read_boundary_block,
)
from turbulence_models import MODELS  # noqa: E402

CASE_SOURCE = Path(__file__).parent / "pitzDaily_SA"
TEST_CASE_DIR = Path(__file__).parent.parent.parent / "test_cases" / "sa_turbulence_test"


def _disable_fpe() -> None:
    """Disable hardware FPE trapping set by OpenFOAM."""
    libm = ctypes.CDLL(ctypes.util.find_library("m"))
    libm.fedisableexcept(0x3F)


def _generate_nonuniform_fields(case_dir: Path, cc: np.ndarray) -> None:
    """Overwrite 0/ with non-uniform fields computed from cell centers."""
    zero_dir = case_dir / "0"
    nu = 1e-5

    # Read boundary blocks from original files before overwriting
    bc_U = _read_boundary_block(zero_dir / "U")
    bc_nuTilda = _read_boundary_block(zero_dir / "nuTilda")
    bc_nut = _read_boundary_block(zero_dir / "nut")

    # Compute analytical profiles
    U_vals = compute_velocity(cc)
    nuTilda_vals = compute_nuTilda(cc, nu=nu)
    nut_vals = compute_nut_from_nuTilda(nuTilda_vals, nu=nu)

    # Write non-uniform fields
    write_vector_field(
        zero_dir / "U", "U", "[0 1 -1 0 0 0 0]", U_vals, bc_U,
    )
    write_scalar_field(
        zero_dir / "nuTilda", "nuTilda", "[0 2 -1 0 0 0 0]", nuTilda_vals, bc_nuTilda,
    )
    write_scalar_field(
        zero_dir / "nut", "nut", "[0 2 -1 0 0 0 0]", nut_vals, bc_nut,
    )


@pytest.fixture(scope="session", autouse=True)
def neon_session() -> Generator[None, None, None]:  # type: ignore[misc]
    """Initialize NeoN once for all turbulence tests and disable hardware FPE trapping."""
    nn.initialize(["test"])
    _disable_fpe()
    yield


@pytest.fixture(scope="session", params=list(MODELS.keys()))
def turbulence_context(request: Any) -> Generator[dict[str, Any], None, None]:
    """Set up both backends with identical non-uniform fields.

    Parameterized over all registered turbulence models. Yields a dict
    with pybFoam and NeoN fields, runtime objects, and the model config.

    1. Copy case, create mesh, get cell centers
    2. Generate non-uniform fields and write to 0/
    3. Read into pybFoam and NeoN
    4. Yield context dict for tests
    """
    model_cfg = MODELS[request.param]

    if TEST_CASE_DIR.exists():
        shutil.rmtree(TEST_CASE_DIR)
    TEST_CASE_DIR.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(CASE_SOURCE, TEST_CASE_DIR)

    original_dir = Path.cwd()
    os.chdir(TEST_CASE_DIR)
    try:
        # --- Create mesh ---
        args = pyf.argList(["test"])
        run_time = pyf.Time(args)
        _disable_fpe()
        block_mesh_dict = pyf.dictionary.read("system/blockMeshDict")
        mesh = generate_blockmesh(run_time, block_mesh_dict)
        _disable_fpe()

        # --- Generate non-uniform fields ---
        cc: np.ndarray = np.array(mesh.C().internalField())
        _generate_nonuniform_fields(TEST_CASE_DIR, cc)

        # --- Read pybFoam fields (non-uniform from disk) ---
        U_of = volVectorField.read_field(mesh, "U")
        phi_of = pyf.createPhi(U_of)
        nuTilda_of = volScalarField.read_field(mesh, "nuTilda")
        nut_of = volScalarField.read_field(mesh, "nut")
        laminar_transport = singlePhaseTransportModel(U_of, phi_of)
        turbulence = incompressibleTurbulenceModel.New(U_of, phi_of, laminar_transport)

        # --- Create NeoN runtime and read fields ---
        rt = nfb.create_adapter_run_time(run_time)
        solvers = rt.fv_solution_dict.subDict("solvers")
        solvers.insert_dict("p", nfb.map_fv_solution(solvers.subDict("p")))
        solvers.insert_dict("U", nfb.map_fv_solution(solvers.subDict("U")))
        if solvers.contains("nuTilda"):
            solvers.insert_dict(
                "nuTilda", nfb.map_fv_solution(solvers.subDict("nuTilda")),
            )
        rt.fv_schemes_dict = nfb.map_fv_schemes(rt.fv_schemes_dict)

        nuTilda_nn = nfb.read_scalar_volume_field(rt, "nuTilda")
        nut_nn = nfb.read_scalar_volume_field(rt, "nut")
        U_nn = nfb.read_vector_volume_field(rt, "U")
        phi_nn = nfb.create_phi(rt, "U")
        d_nn = nfb.compute_wall_distance(rt)
        nu_value = float(nfb.read_transport_viscosity(rt))

        # Save initial field values (before any test mutates them via correct())
        initial_nuTilda_of = np.array(nuTilda_of.internalField()).copy()
        initial_nut_of = np.array(nut_of.internalField()).copy()
        initial_nut_nn = np.asarray(nut_nn.internal_vector().__array__()).copy()

        yield {
            "model_config": model_cfg,
            # pybFoam side
            "nuTilda_of": nuTilda_of,
            "nut_of": nut_of,
            "U_of": U_of,
            "laminar_transport": laminar_transport,
            "turbulence": turbulence,
            # NeoN side
            "nuTilda": nuTilda_nn,
            "nut": nut_nn,
            "U": U_nn,
            "phi": phi_nn,
            "d": d_nn,
            "rt": rt,
            "nu_value": nu_value,
            # Saved initial values (immutable numpy copies)
            "initial_nuTilda_of": initial_nuTilda_of,
            "initial_nut_of": initial_nut_of,
            "initial_nut_nn": initial_nut_nn,
        }
    finally:
        os.chdir(original_dir)
        if TEST_CASE_DIR.exists():
            shutil.rmtree(TEST_CASE_DIR)
