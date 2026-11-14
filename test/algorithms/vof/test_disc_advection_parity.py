# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Advection-only VoF (``neofoam.algorithms.vof``) on a disc in uniform flow.

**Case.** ``cases/discUniformFlow``: a 32x32 2D box, a disc of radius 0.15
advected by the uniform velocity ``(1, 0.5, 0)`` for 40 steps of 0.005 (cell
Courant number 0.16 in x), written at 0.1 and 0.2. The walls carry zero
velocity, so ``phi = fvc::flux(U)`` is zero on the boundary: the box is closed
and the alpha volume must stay constant. The disc stays clear of the walls.
``neofoam preprocess`` meshes each staged copy and sets the disc through the
case's ``system/setAlphaField.py``.
Each regime changes one MULES control of that case through pybFoam's
dictionary writer; two also add advectorVoF's ``isoAdvector`` dict, which
makes the flux time-dependent (``phi = phi0*cos(...)``, ``period`` /
``spirallingFlow``, the flow reversed at ``reverseTime``) — the period of
0.3 and the spiralling time of 0.15 turn the flow around within the run.

**Oracles.**

* openfoam backend vs incompressibleVoF: the reference
  (``_incompressible_vof_alpha_worker.py``) runs the solver's own alpha solve
  with the real two-phase mixture and the flow frozen. The backend runs the
  same ``alpha_eqn`` with a stand-in for the mixture's interface terms, so the
  two only differ in how ``nHatf`` is computed; they agree to round-off
  (``atol = 1e-12`` on a field in [0, 1]).
* neon backend vs openfoam backend, per executor: the same algorithm in other
  kernels, so the agreement is round-off too, accumulated over 40 steps and
  with reordered atomic sums on the CPU/GPU executors (``atol = 1e-10``).
  The neon backend runs ``MULESCorr no`` only (see ``_neon_backend``), so
  the explicit regimes are the parity set; the adaptive one
  also checks that both backends take the same ``maxAlphaCo`` time steps.
  The time-dependent regimes are in the parity set: the neon backend scales
  its flux on the executor, the openfoam one in pybFoam, from the same factor.
* time-dependent flux vs a prescribed velocity: the openfoam backend with
  ``isoAdvector { period 0.3; }`` against the same backend driven by
  ``U(x, t) = U0 cos(2 pi t/0.3)`` evaluated at mid-step
  (``_prescribed_velocity_worker.py``), with ``maxAlphaCo`` time steps. Both
  must evaluate the cosine at ``t + dt/2`` of the step being taken and let
  ``setDeltaT`` see the previous step's flux; ``phi0*c`` and
  ``fvc::flux(c*U0)`` differ by round-off only (``atol = 1e-12``).
* volume and bounds: MULES is conservative, so the volume holds to round-off
  (``rtol = 1e-12``). The explicit path is bounded up to the limiter's
  unconverged remainder after ``nLimiterIter 3``: ``1e-8``, the bound
  TwoPhaseFlow's discUniFlow test asserts. ``MULESCorr`` is not: its second
  corrector averages the corrected field with the pre-correction one, which
  the limiter never saw, so it undershoots by O(1e-5) here (-2.8e-5), as
  interFoam does; it gets the 1e-4 slack ``test_mules_regimes.py`` uses for
  the same path, still four orders below an unlimited scheme's O(0.1).

Every run goes through a subprocess (``advect_in_subprocess`` or the reference
worker): one ``Foam::Time`` and one Kokkos runtime per process.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Union

import numpy as np
import pybFoam as pyf
import pytest
from numpy.testing import assert_allclose

from neofoam.algorithms.vof import AdvectionResult, advect_in_subprocess

CASE = Path(__file__).parent / "cases" / "discUniformFlow"
REFERENCE_WORKER = Path(__file__).parent / "_incompressible_vof_alpha_worker.py"
VELOCITY_WORKER = Path(__file__).parent / "_prescribed_velocity_worker.py"
# The case's uniform 0/U, which the prescribed-velocity worker modulates.
U0 = (1.0, 0.5, 0.0)

Controls = Mapping[str, Union[bool, float]]

# (alpha solver controls, controlDict entries) per regime.
REGIMES: dict[str, tuple[Controls, Controls]] = {
    "explicit_nAlphaCorr2": ({}, {}),
    "explicit_nAlphaCorr1": ({"nAlphaCorr": 1}, {}),
    "explicit_subCycles2": ({"nAlphaCorr": 1, "nAlphaSubCycles": 2}, {}),
    "MULESCorr": ({"MULESCorr": True}, {}),
    "explicit_adaptive": ({"nAlphaCorr": 1}, {"adjustTimeStep": True}),
    # maxDeltaT: where the flow turns, setDeltaT sees the previous step's
    # near-zero flux and would take one large step
    "explicit_period_adaptive": ({"nAlphaCorr": 1}, {"adjustTimeStep": True, "maxDeltaT": 0.01}),
    "explicit_spiralling_reversed": ({}, {}),
}
# The isoAdvector dict (advectorVoF's time-dependent flux) per regime that has one.
FLUX_MODULATION: dict[str, Controls] = {
    "explicit_period_adaptive": {"period": 0.3},
    "explicit_spiralling_reversed": {"spirallingFlow": 0.15, "reverseTime": 0.1},
}
BOUND_SLACK = {name: 1e-4 if name == "MULESCorr" else 1e-8 for name in REGIMES}
# incompressibleVoF holds the flux fixed and steps with a fixed deltaT.
FIXED_DT_REGIMES = [
    name for name in REGIMES if name != "explicit_adaptive" and name not in FLUX_MODULATION
]
NEON_REGIMES = [
    "explicit_nAlphaCorr2",
    "explicit_nAlphaCorr1",
    "explicit_subCycles2",
    "explicit_adaptive",
    "explicit_period_adaptive",
    "explicit_spiralling_reversed",
]
EXECUTORS = ["Serial", "CPU", "GPU"]


def _stage(regime: str, dest: Path) -> Path:
    """Copy the case, mesh it, and apply the regime through pybFoam's dictionary writer."""
    shutil.copytree(CASE, dest)
    neofoam = Path(sys.executable).with_name("neofoam")
    subprocess.run([str(neofoam), "preprocess", str(dest)], check=True, capture_output=True)
    alpha_controls, control_dict_entries = REGIMES[regime]
    fv_solution = pyf.dictionary.read(str(dest / "system" / "fvSolution"))
    alpha_dict = fv_solution.subDict("solvers").subDict("alpha.water")
    for key, value in alpha_controls.items():
        alpha_dict.set(key, value)
    if regime in FLUX_MODULATION:
        # subDictOrAdd hands out a copy: fill a dictionary and set it as a whole.
        modulation = pyf.dictionary()
        for key, value in FLUX_MODULATION[regime].items():
            modulation.set(key, value)
        fv_solution.set("isoAdvector", modulation)
    fv_solution.write(str(dest / "system" / "fvSolution"))
    control_dict = pyf.dictionary.read(str(dest / "system" / "controlDict"))
    for key, value in control_dict_entries.items():
        control_dict.set(key, value)
    control_dict.write(str(dest / "system" / "controlDict"))
    return dest


def _gpu_available() -> bool:
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import neon._neon as nn; nn.initialize(['probe']); nn.GPUExecutor(); nn.finalize()",
        ],
        env={**os.environ, "FOAM_SIGFPE": "false"},
        capture_output=True,
        timeout=120.0,
    )
    return probe.returncode == 0


class _Runs:
    """Each (regime, backend, executor) run once per module, in its own case copy."""

    def __init__(self, tmp_path_factory: pytest.TempPathFactory) -> None:
        self._tmp = tmp_path_factory
        self._results: dict[tuple[str, ...], AdvectionResult] = {}

    def advect(self, regime: str, backend: str, executor: str = "Serial") -> AdvectionResult:
        key = (regime, backend, executor)
        if key not in self._results:
            case = _stage(regime, self._tmp.mktemp("_".join(key)) / "case")
            self._results[key] = advect_in_subprocess(case, backend, executor)  # type: ignore[arg-type]
        return self._results[key]

    def incompressible_vof(self, regime: str) -> AdvectionResult:
        key = (regime, "incompressibleVoF")
        if key not in self._results:
            case = _stage(regime, self._tmp.mktemp("_".join(key)) / "case")
            out = case / "reference.npz"
            completed = subprocess.run(
                [sys.executable, str(REFERENCE_WORKER), str(case), str(out)],
                capture_output=True,
                text=True,
            )
            assert completed.returncode == 0, completed.stdout[-4000:] + completed.stderr[-4000:]
            with np.load(out) as saved:
                self._results[key] = AdvectionResult(saved["times"], saved["alpha"], np.empty(0))
        return self._results[key]

    def prescribed_velocity(self, regime: str) -> AdvectionResult:
        """The openfoam backend driven by ``U0 cos(2 pi t/period)`` instead of isoAdvector."""
        key = (regime, "prescribedVelocity")
        if key not in self._results:
            case = _stage(regime, self._tmp.mktemp("_".join(key)) / "case")
            fv_solution = pyf.dictionary.read(str(case / "system" / "fvSolution"))
            fv_solution.remove("isoAdvector")
            fv_solution.write(str(case / "system" / "fvSolution"))
            out = case / "velocity.npz"
            period = str(FLUX_MODULATION[regime]["period"])
            completed = subprocess.run(
                [sys.executable, str(VELOCITY_WORKER), str(case), period, *map(str, U0), str(out)],
                capture_output=True,
                text=True,
            )
            assert completed.returncode == 0, completed.stdout[-4000:] + completed.stderr[-4000:]
            with np.load(out) as saved:
                self._results[key] = AdvectionResult(saved["times"], saved["alpha"], np.empty(0))
        return self._results[key]


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory) -> _Runs:
    return _Runs(tmp_path_factory)


@pytest.mark.parametrize("regime", FIXED_DT_REGIMES)
def test_openfoam_backend_reproduces_incompressible_vof_alpha(runs: _Runs, regime: str) -> None:
    """The openfoam backend advances alpha exactly as incompressibleVoF's alpha solve does."""
    reference = runs.incompressible_vof(regime)

    result = runs.advect(regime, "openfoam")

    assert_allclose(result.times, [0.0, 0.1, 0.2], rtol=0, atol=1e-12, err_msg=regime)
    assert_allclose(result.times, reference.times, rtol=0, atol=1e-12, err_msg=regime)
    assert_allclose(result.alpha, reference.alpha, rtol=0, atol=1e-12, err_msg=regime)


@pytest.mark.parametrize("executor", EXECUTORS)
@pytest.mark.parametrize("regime", NEON_REGIMES)
def test_neon_backend_matches_openfoam_backend(runs: _Runs, regime: str, executor: str) -> None:
    """The neon backend reproduces the openfoam backend on every executor."""
    if executor == "GPU" and not _gpu_available():
        pytest.skip("no GPU executor available on this host")
    reference = runs.advect(regime, "openfoam")

    result = runs.advect(regime, "neon", executor)

    label = f"{regime} on {executor}"
    assert_allclose(result.times, reference.times, rtol=1e-12, atol=0, err_msg=label)
    assert_allclose(result.alpha, reference.alpha, rtol=0, atol=1e-10, err_msg=label)


@pytest.mark.parametrize("regime", list(REGIMES))
def test_openfoam_backend_conserves_alpha_volume(runs: _Runs, regime: str) -> None:
    """The alpha volume of the closed box is the same at every write time."""
    result = runs.advect(regime, "openfoam")

    volumes = result.alpha @ result.cell_volumes

    assert_allclose(volumes, volumes[0], rtol=1e-12, err_msg=regime)


@pytest.mark.parametrize("regime", list(REGIMES))
def test_openfoam_backend_keeps_alpha_bounded(runs: _Runs, regime: str) -> None:
    """alpha stays in [0, 1] up to the MULES limiter's remainder."""
    result = runs.advect(regime, "openfoam")

    slack = BOUND_SLACK[regime]
    assert result.alpha.min() >= -slack, f"{regime}: min alpha {result.alpha.min():.3e}"
    assert result.alpha.max() <= 1 + slack, f"{regime}: max alpha {result.alpha.max():.3e}"


def test_openfoam_backend_time_dependent_flux_matches_prescribed_velocity(runs: _Runs) -> None:
    """``isoAdvector { period }`` scales the flux as a velocity ``U0 cos(2 pi t/period)`` does."""
    regime = "explicit_period_adaptive"
    reference = runs.prescribed_velocity(regime)

    result = runs.advect(regime, "openfoam")

    assert_allclose(result.times, reference.times, rtol=1e-12, atol=0, err_msg=regime)
    assert_allclose(result.alpha, reference.alpha, rtol=0, atol=1e-12, err_msg=regime)
