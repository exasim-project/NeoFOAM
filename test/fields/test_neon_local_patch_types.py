# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The NeoN field reader translates the four *local* OpenFOAM patch types.

``NeoFOAM::readVolBoundaryConditions`` translates every OpenFOAM ``fvPatchField``
type into a NeoN boundary dictionary through a whitelist keyed on the type word;
a type outside it kills the run during initialisation ("Unsupported boundary
condition type"). ``slip``, ``surfaceNormalFixedValue``, ``uniformTotalPressure``
and ``movingWallVelocity`` were missing, which is what stopped 8 of the
incompressibleFluid tutorials on the NeoN backend before a single time step.

The case (``cases/localPatchTypes``) is a unit cube of 2 x 2 x 2 cells whose six
axis-aligned patches carry one of those types each, written exactly as the
tutorials write them (``simpleCar``'s ramped intake, ``TJunction``'s ``p0`` time
table, ``oscillatingInletACMI2D``'s wall ``value``). Axis-aligned patches make the
face normals the axis unit vectors, so the expected values below are exact, not
mesh-dependent.

Two of the four are approximations and their expectation is written against the
approximation, not against OpenFOAM: ``movingWallVelocity`` becomes a stationary
no-slip wall — which is why the case gives it a non-zero ``value`` that the
translation must discard — and ``surfaceNormalFixedValue`` ignores the ``ramp``
Function1, which OpenFOAM evaluates to 0 at t=0. Each must announce itself, so the
notices are asserted too.

The read runs in a subprocess (see :mod:`_local_patch_types_worker`): one
``Foam::Time`` per process, and an unsupported type aborts the process rather than
raising in-process.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from neofoam.tooling.casebuild import from_template, patch

_HERE = Path(__file__).parent
_WORKER = _HERE / "_local_patch_types_worker.py"
_CASE = _HERE / "cases" / "localPatchTypes"

#: ``0/U`` internal field — every expectation below is derived from it.
INTERNAL_U = (1.0, 2.0, 3.0)

#: Per-patch boundary velocity the NeoN reader must produce for ``0/U``.
#:
#: * ``xMin``: surfaceNormalFixedValue, refValue 1.2 on the outward normal (-1 0 0).
#: * ``xMax``: zeroGradient — the internal value, the control.
#: * ``yMin``: slip on the outward normal (0 -1 0) — the normal component removed.
#: * ``yMax``: movingWallVelocity — the static-mesh approximation, *not* its ``value``.
#: * ``zMin`` / ``zMax``: noSlip.
EXPECTED_U = {
    "xMin": (-1.2, 0.0, 0.0),
    "xMax": INTERNAL_U,
    "yMin": (1.0, 0.0, 3.0),
    "yMax": (0.0, 0.0, 0.0),
    "zMin": (0.0, 0.0, 0.0),
    "zMax": (0.0, 0.0, 0.0),
}

#: Per-patch boundary pressure for ``0/p``: ``xMax`` is pinned at p0(t=0) = 10,
#: every other patch is zeroGradient on an internal field of 5.
EXPECTED_P = {
    "xMin": 5.0,
    "xMax": 10.0,
    "yMin": 5.0,
    "yMax": 5.0,
    "zMin": 5.0,
    "zMax": 5.0,
}

#: Field values are read and copied, never solved for, so they must be exact to
#: round-off — no linear-solver residual enters.
TOLERANCE = 1e-12


def _run_worker(role: str, case: Path) -> subprocess.CompletedProcess[str]:
    """Run one worker role in a fresh process (one ``Foam::Time`` per process)."""
    return subprocess.run(
        [sys.executable, str(_WORKER), role, str(case)],
        cwd=str(case.parent),
        capture_output=True,
        text=True,
        # A rejected patch type takes OpenFOAM's abort path, whose stack trace can carry
        # bytes that are not valid UTF-8; the message being asserted on is plain ASCII.
        errors="replace",
        timeout=600,
    )


def _read_fields(case: Path) -> subprocess.CompletedProcess[str]:
    """Mesh the case, then read its fields; fail loudly if either role aborts."""
    meshing = _run_worker("mesh", case)
    assert meshing.returncode == 0, f"blockMesh failed:\n{meshing.stdout}\n{meshing.stderr}"
    return _run_worker("fields", case)


@pytest.fixture
def staged_case(tmp_path: Path) -> Path:
    """A writable copy of ``cases/localPatchTypes`` (never mutate the checked-in one)."""
    return from_template(_CASE).build_at(tmp_path / "localPatchTypes").path


def test_local_patch_types_translate_to_neon_boundary_values(staged_case: Path) -> None:
    """slip, surfaceNormalFixedValue, uniformTotalPressure and movingWallVelocity all read."""
    result = _read_fields(staged_case)

    assert result.returncode == 0, f"field read failed:\n{result.stdout}\n{result.stderr}"
    velocity = np.load(staged_case / "U_boundary.npz")
    for name, expected in EXPECTED_U.items():
        np.testing.assert_allclose(
            velocity[name],
            np.broadcast_to(expected, velocity[name].shape),
            rtol=TOLERANCE,
            atol=TOLERANCE,
            err_msg=f"localPatchTypes: U on patch {name}",
        )
    pressure = np.load(staged_case / "p_boundary.npz")
    for name, expected_p in EXPECTED_P.items():
        np.testing.assert_allclose(
            pressure[name],
            expected_p,
            rtol=TOLERANCE,
            atol=TOLERANCE,
            err_msg=f"localPatchTypes: p on patch {name}",
        )


def test_approximated_patch_types_announce_the_approximation(staged_case: Path) -> None:
    """Every degraded translation prints a notice naming its patch and what it dropped."""
    result = _read_fields(staged_case)

    assert result.returncode == 0, f"field read failed:\n{result.stdout}\n{result.stderr}"
    output = result.stdout + result.stderr
    assert "movingWallVelocity on patch 'yMax'" in output
    assert "stationary no-slip wall" in output
    assert "surfaceNormalFixedValue on patch 'xMin'" in output
    assert "`ramp` Function1" in output
    assert "uniformTotalPressure on patch 'xMax'" in output


def test_untranslated_patch_type_still_fails_naming_the_type_and_patch(tmp_path: Path) -> None:
    """A type outside the table keeps failing with the supported-type list.

    ``pressureInletVelocity`` is an OpenFOAM patch type the table does not carry (and
    a near-miss for the ``pressureInletOutletVelocity`` it does), so the read gets
    past OpenFOAM's own run-time selection and reaches the translation gate.
    """
    case = (
        from_template(_CASE) | patch("0/U", **{"boundaryField.yMax.type": "pressureInletVelocity"})
    ).build_at(tmp_path / "untranslatedPatchType")

    result = _read_fields(case.path)

    assert result.returncode != 0, f"read unexpectedly succeeded:\n{result.stdout}"
    output = result.stdout + result.stderr
    assert "Unsupported boundary condition type 'pressureInletVelocity' on patch 'yMax'" in output
    assert "Supported types:" in output
