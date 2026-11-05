# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""End-to-end run test for the NeoN-based ``neorhoPimpleFoam``.

Drives the transient compressible PIMPLE duct case (``test/setup_rhoPimple``)
through the Python bindings and asserts the flow is finite, develops in the
expected direction, and - the key correctness check - that rho == p/(R*T)
holds in the written fields, confirming the perfectGas thermo-update wiring
actually ran. It deliberately does NOT assert field-by-field parity against
OpenFOAM's ``rhoPimpleFoam`` - this is a port-runs-and-makes-physical-sense
smoke test, not a parity test, the same way ``test_neoSimpleFoam.py`` treats
``neoSimpleFoam``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from neofoam.solver.neorhoPimpleFoam import _energy_patch_types
from neofoam.tooling.casebuild import block_mesh, from_template, patch


def _run_solver(case_dir: Path) -> None:
    """Run ``NeoRhoPimpleFoam`` on a case in an isolated subprocess.

    NeoN/Kokkos + OpenFOAM hold per-process global state that does not survive a second
    in-process solver run.
    """
    env = {**os.environ, "FOAM_SIGFPE": "false"}
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from neofoam.solver.neorhoPimpleFoam import NeoRhoPimpleFoam;"
            " NeoRhoPimpleFoam(['neorhoPimpleFoam']).run()",
        ],
        cwd=str(case_dir),
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, (
        f"neorhoPimpleFoam failed (rc={result.returncode}):\n{result.stderr[-3000:]}"
    )


UNIVERSAL_GAS_CONSTANT = 8314.46261815324  # J/(kmol K)
MOL_WEIGHT = 28.9  # matches test/setup_rhoPimple/constant/thermophysicalProperties


def test_neorhoPimpleFoam_runs_and_develops_duct_flow(tmp_path: Path) -> None:
    """neorhoPimpleFoam runs end-to-end and the EOS stays self-consistent."""
    repo_root = Path(__file__).parents[2]
    source_case = repo_root / "test" / "setup_rhoPimple"
    case = (from_template(source_case) | block_mesh()).build_at(tmp_path / "rhopimple")

    _run_solver(case.path)

    U = case.read_field("U")
    p = case.read_field("p")
    T = case.read_field("T")
    rho = case.read_field("rho")

    assert np.all(np.isfinite(U)), "U contains non-finite values"
    assert np.all(np.isfinite(p)), "p contains non-finite values"
    assert np.all(np.isfinite(T)), "T contains non-finite values"
    assert np.all(np.isfinite(rho)), "rho contains non-finite values"

    # EOS consistency: proves the perfectGas thermo-update wiring (psi.assign/
    # rho.assign each outer iteration) actually ran, independent of how far the
    # flow itself developed.
    R = UNIVERSAL_GAS_CONSTANT / MOL_WEIGHT
    np.testing.assert_allclose(
        rho, p / (R * T), rtol=1e-8, err_msg="rho != p/(R*T); EOS update is not wired correctly"
    )

    # Directional sanity: the duct's inlet drives U, 1 m/s; mean flow should stay
    # close to that rather than collapsing to the zero initial condition.
    mean_ux = float(np.mean(U[:, 0]))
    assert mean_ux > 0.5, f"duct flow did not develop (mean Ux = {mean_ux:.3e})"
    print(
        f"neorhoPimpleFoam ran: mean Ux = {mean_ux:.3e}, "
        f"p range = [{float(np.min(p)):.1f}, {float(np.max(p)):.1f}], "
        f"T range = [{float(np.min(T)):.2f}, {float(np.max(T)):.2f}]"
    )


def test_energy_patch_types_follow_the_temperature_patch_types() -> None:
    """he's BC types are derived from T's, as OpenFOAM's heThermo::heBoundaryTypes does.

    fixedValue -> fixedEnergy, empty -> empty, every mixed type -> mixedEnergy (a
    gradientEnergy there would force valueFraction = 0 and lose the fixed inflow value of an
    inletOutlet patch), anything else -> gradientEnergy.
    """
    t_types = [
        "fixedValue",
        "empty",
        "zeroGradient",
        "inletOutlet",
        "outletInlet",
        "mixed",
        "freestream",
        "advective",
        "waveTransmissive",
        "calculated",
    ]
    assert _energy_patch_types(t_types) == [
        "fixedEnergy",
        "empty",
        "gradientEnergy",
        *["mixedEnergy"] * 6,
        "gradientEnergy",
    ]


@pytest.mark.parametrize(
    ("p_max", "p_min"),
    [
        pytest.param("101000", "99000", id="loose-integer-bounds"),
        pytest.param("100000", "99999", id="clipping-integer-bounds"),
    ],
)
def test_neorhoPimpleFoam_ras_cavity_runs_with_kOmegaSST_and_pressure_limits(
    tmp_path: Path, p_max: str, p_min: str
) -> None:
    """The compressible RAS cavity runs with kOmegaSST and keeps p inside [pMin, pMax].

    ``pMax``/``pMin`` are written as integers (OpenFOAM accepts both), which the settings
    reader must not reject. After 5 steps the unlimited run peaks at p = 100000.014, so the
    second parametrisation's ``pMax 100000`` is exceeded and the pressure control has to clip. Turbulence is proven to
    have run by nut leaving its zero initial value and by k, omega staying positive.
    """
    repo_root = Path(__file__).parents[2]
    source_case = repo_root / "test" / "setup_rhoPimpleRAS"
    case = (
        from_template(source_case)
        | patch("system/fvSolution", **{"PIMPLE.pMax": p_max, "PIMPLE.pMin": p_min})
        | block_mesh()
    ).build_at(tmp_path / "rhopimple_ras")

    _run_solver(case.path)

    fields = {name: case.read_field(name) for name in ("U", "p", "T", "rho", "k", "omega", "nut")}
    for name, values in fields.items():
        assert np.all(np.isfinite(values)), f"{name} contains non-finite values"

    assert np.all(fields["k"] > 0.0), "k must stay positive"
    assert np.all(fields["omega"] > 0.0), "omega must stay positive"
    assert float(np.max(fields["nut"])) > 0.0, "nut never left its zero initial value"

    slack = 1e-3  # Pa, below the 0.014 Pa the unlimited run overshoots pMax by
    assert float(np.max(fields["p"])) <= float(p_max) + slack
    assert float(np.min(fields["p"])) >= float(p_min) - slack

    R = UNIVERSAL_GAS_CONSTANT / MOL_WEIGHT
    np.testing.assert_allclose(fields["rho"], fields["p"] / (R * fields["T"]), rtol=1e-6)
