# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""incompressibleFluidNeoN with Boussinesq buoyancy against buoyantBoussinesqPimpleFoam.

Each case runs three times on identical copies: OpenFOAM's ``buoyantBoussinesqPimpleFoam``
(the reference, run here rather than checked in, so it always comes from the installed
OpenFOAM and the case files on disk), ``incompressibleFluidNeoN`` and the pybFoam
``incompressibleFluid`` with its Boussinesq model. Every solver runs in its own subprocess
(one ``Foam::Time``/Kokkos runtime per process); the fields of the last time step are read
back with foamlib.

The cases are OpenFOAM's hotRoom (laminar with the momentum predictor, k-epsilon with
``alphatJayatillekeWallFunction`` walls and no predictor, laminar with two outer
correctors, laminar with an open ceiling: ``prghPressure`` p_rgh, zeroGradient U and T)
and BenardCells (2-D, ``corrected`` laplacian/snGrad). They use only schemes
both backends discretise identically: BenardCells' ``limitedLinear`` is not in NeoN, so its
``div(phi,U)`` is ``linearUpwind grad(U)`` and ``div(phi,T)`` ``upwind``. All solvers run with
``tolerance 1e-12, relTol 0``, so each step converges to round-off and the runs differ by
round-off growth alone, measured in units of the field's peak (of ``T - TRef`` for T) at the
last step: hotRoom k-epsilon 2e-11 (U), 3e-11 (T), 6e-12 (p_rgh), 9e-11 (alphat);
BenardCells 1e-10; the laminar hotRooms, whose 600 K heater face makes the momentum
predictor amplify round-off, 3e-9 (U, T, one outer corrector) and 5e-9 (two); ``p_rgh`` and
``p`` stay below 2e-11 throughout. The bound is ``1e-8`` of the peak. The open ceiling lets
the heated air out, so its flow stays ~1000 times weaker (peak |U| 6e-4 m/s) and the same
absolute round-off is 5e-8 of its peak (U; T 2e-8): bound ``1e-7``. Serial runs of the same
binaries are deterministic, so the margins need to cover no run-to-run noise.

The pybFoam model matches in ``U``, ``T`` and ``alphat`` with the k-epsilon closure as
closely, but does not shift ``p`` to ``pRefValue`` in the reference cell as pEqn.H does: its
``p``/``p_rgh`` differ from the native ones by a constant, so they are compared relative to
their reference-cell value, and the other pressure level changes the round-off of the
laminar case to 3e-8 of the peak against OpenFOAM, hence ``1e-7`` for that comparison.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from foamlib import FoamFieldFile

_CASES = Path(__file__).parent / "cases"
T_REF = 300.0
#: The pRefCell of every case: where the reference value of p is set.
REF_CELL = 0
RTOL_OF_PEAK = 1e-8
#: Cases whose bound differs from RTOL_OF_PEAK (see the module docstring).
CASE_RTOL_OF_PEAK = {"hotRoomOpenCeiling": 1e-7}
PYBFOAM_RTOL_OF_PEAK = 1e-7

NEON = "from neofoam.solver.incompressibleFluidNeoN import run; run(['incompressibleFluidNeoN'])"
PYBFOAM = "from neofoam.solver.incompressibleFluid import run; run(['incompressibleFluid'])"

#: (case, field) pairs compared with buoyantBoussinesqPimpleFoam.
NATIVE = [
    *(
        (case, field)
        for case in ("hotRoomLaminar", "hotRoomOuterCorrectors", "benardCells")
        for field in ("U", "T", "p_rgh", "p")
    ),
    *(("hotRoomKEpsilon", field) for field in ("U", "T", "p_rgh", "p", "alphat")),
    *(("hotRoomOpenCeiling", field) for field in ("U", "T", "p_rgh", "p")),
]

#: (case, field) pairs compared with the pybFoam incompressibleFluid.
PYBFOAM_FIELDS = [
    *(("hotRoomLaminar", field) for field in ("U", "T", "p_rgh", "p")),
    *(("hotRoomKEpsilon", field) for field in ("U", "T", "p_rgh", "p", "alphat")),
]


def _run(command: list[str], case: Path, label: str) -> None:
    env = {**os.environ, "FOAM_SIGFPE": "false"}
    result = subprocess.run(
        command, cwd=case, env=env, capture_output=True, text=True, timeout=1200
    )
    assert result.returncode == 0, (
        f"{label} failed in {case} (rc={result.returncode}):\n"
        f"{result.stdout[-3000:]}\n{result.stderr[-3000:]}"
    )


def _mesh(case: Path) -> None:
    _run(["blockMesh"], case, "blockMesh")
    if (case / "system" / "setFieldsDict").is_file():
        _run(["setFields"], case, "setFields")


def _last_time(case: Path) -> Path:
    times = [d for d in case.iterdir() if d.is_dir() and d.name.replace(".", "").isdigit()]
    return max(times, key=lambda d: float(d.name))


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory) -> dict[tuple[str, str], Path]:
    """``{(case, solver): run directory}``, each run once per module on demand."""
    cache: dict[tuple[str, str], Path] = {}
    commands = {
        "openfoam": ["buoyantBoussinesqPimpleFoam"],
        "neon": [sys.executable, "-c", NEON],
        "pybfoam": [sys.executable, "-c", PYBFOAM],
    }

    class Runs(dict):  # type: ignore[type-arg]
        def __missing__(self, key: tuple[str, str]) -> Path:
            case, solver = key
            meshed = tmp_path_factory.getbasetemp() / case / "meshed"
            if not meshed.is_dir():
                shutil.copytree(_CASES / case, meshed)
                _mesh(meshed)
            run_dir = meshed.parent / solver
            shutil.copytree(meshed, run_dir)
            _run(commands[solver], run_dir, f"{solver} on {case}")
            self[key] = run_dir
            return run_dir

    return Runs(cache)


def _last(run_dir: Path, field: str) -> np.ndarray:
    return np.asarray(FoamFieldFile(_last_time(run_dir) / field).internal_field, dtype=float)


def _peak(field: str, reference: np.ndarray) -> float:
    return float(np.abs(reference - T_REF).max() if field == "T" else np.abs(reference).max())


@pytest.mark.parametrize(("case", "field"), NATIVE)
def test_neon_boussinesq_matches_openfoam(
    runs: dict[tuple[str, str], Path], case: str, field: str
) -> None:
    reference = _last(runs[case, "openfoam"], field)

    result = _last(runs[case, "neon"], field)

    np.testing.assert_allclose(
        result,
        reference,
        rtol=0.0,
        atol=CASE_RTOL_OF_PEAK.get(case, RTOL_OF_PEAK) * _peak(field, reference),
        err_msg=f"{case}: {field} after {_last_time(runs[case, 'neon']).name} s",
    )


@pytest.mark.parametrize(("case", "field"), PYBFOAM_FIELDS)
def test_neon_boussinesq_matches_incompressible_fluid(
    runs: dict[tuple[str, str], Path], case: str, field: str
) -> None:
    reference = _last(runs[case, "pybfoam"], field)

    result = _last(runs[case, "neon"], field)

    if field in ("p", "p_rgh"):
        reference = reference - reference[REF_CELL]
        result = result - result[REF_CELL]
    np.testing.assert_allclose(
        result,
        reference,
        rtol=0.0,
        atol=PYBFOAM_RTOL_OF_PEAK * _peak(field, reference),
        err_msg=f"{case}: {field}, incompressibleFluid vs incompressibleFluidNeoN",
    )
