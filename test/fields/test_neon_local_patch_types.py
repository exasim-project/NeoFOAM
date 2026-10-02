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

A sibling case (``cases/freestreamPatchTypes``) covers the far-field family that
stopped ``simpleFoam/airFoil2D``: ``freestreamVelocity``, ``freestreamPressure`` and
scalar ``freestream``. Every one of them carries a ``freestreamValue`` away from its
internal field, so an expectation of the internal value can only come from the
Neumann behaviour the translation produces. All three are approximations and
announce themselves: ``freestream`` is OpenFOAM's ``inletOutlet`` under another key
name, but NeoN's ``inletOutlet`` degenerates to zeroGradient without a flux context,
so the value is pinned instead of lost.

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
_FREESTREAM_CASE = _HERE / "cases" / "freestreamPatchTypes"

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

#: The far-field velocities of ``cases/freestreamPatchTypes``, whose internal field is
#: also :data:`INTERNAL_U`.
#:
#: * ``xMin``: freestreamVelocity — exactly its ``freestreamValue``, which the case
#:   deliberately sets away from the internal field.
#: * ``xMax``: zeroGradient — the internal value, the control.
#: * ``yMin`` / ``yMax`` / ``zMin`` / ``zMax``: noSlip.
FREESTREAM_EXPECTED_U = {
    "xMin": (25.75, 3.62, 0.0),
    "xMax": INTERNAL_U,
    "yMin": (0.0, 0.0, 0.0),
    "yMax": (0.0, 0.0, 0.0),
    "zMin": (0.0, 0.0, 0.0),
    "zMax": (0.0, 0.0, 0.0),
}

#: The far-field pressures of ``cases/freestreamPatchTypes`` on an internal field of 5.
#:
#: * ``xMax``: freestreamPressure is translated to zeroGradient, so its
#:   ``freestreamValue`` of 11 must not appear — it holds the internal value.
#: * ``yMin``: scalar freestream is pinned at its ``freestreamValue`` of 13. Mapping it
#:   to NeoN's inletOutlet would read as exact but degenerates to zeroGradient without a
#:   flux context, which on airFoil2D let the far-field nuTilda grow from 4e-05 to ~41
#:   against a native peak of 0.29. The internal value of 5 is what that regression
#:   would show here.
#: * the rest: zeroGradient, holding the internal value.
FREESTREAM_EXPECTED_P = {
    "xMin": 5.0,
    "xMax": 5.0,
    "yMin": 13.0,
    "yMax": 5.0,
    "zMin": 5.0,
    "zMax": 5.0,
}

#: Substrings every approximated translation of a case must print — the patch it hit
#: and what it dropped.
LOCAL_NOTICES = (
    "movingWallVelocity on patch 'yMax'",
    "stationary no-slip wall",
    "surfaceNormalFixedValue on patch 'xMin'",
    "`ramp` Function1",
    "uniformTotalPressure on patch 'xMax'",
)

FREESTREAM_NOTICES = (
    "freestreamVelocity on patch 'xMin'",
    "fixedValue at the freestreamValue",
    "freestreamPressure on patch 'xMax'",
    "approximated as zeroGradient",
    "freestream on patch 'yMin'",
)

#: One entry per case: the case directory, its two expectation tables and the notices
#: its approximations owe the user.
CASES = [
    pytest.param(_CASE, EXPECTED_U, EXPECTED_P, LOCAL_NOTICES, id="localPatchTypes"),
    pytest.param(
        _FREESTREAM_CASE,
        FREESTREAM_EXPECTED_U,
        FREESTREAM_EXPECTED_P,
        FREESTREAM_NOTICES,
        id="freestreamPatchTypes",
    ),
]

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
def staged_case(tmp_path: Path, request: pytest.FixtureRequest) -> Path:
    """A writable copy of the requested case (never mutate the checked-in one)."""
    case = Path(request.param)
    return from_template(case).build_at(tmp_path / case.name).path


@pytest.mark.parametrize(
    "staged_case, expected_u, expected_p, expected_notices", CASES, indirect=["staged_case"]
)
def test_local_patch_types_translate_to_neon_boundary_values(
    staged_case: Path,
    expected_u: dict[str, tuple[float, float, float]],
    expected_p: dict[str, float],
    expected_notices: tuple[str, ...],
) -> None:
    """Every patch type in the case reads and lands on its documented boundary value."""
    result = _read_fields(staged_case)

    assert result.returncode == 0, f"field read failed:\n{result.stdout}\n{result.stderr}"
    velocity = np.load(staged_case / "U_boundary.npz")
    for name, expected in expected_u.items():
        np.testing.assert_allclose(
            velocity[name],
            np.broadcast_to(expected, velocity[name].shape),
            rtol=TOLERANCE,
            atol=TOLERANCE,
            err_msg=f"{staged_case.name}: U on patch {name}",
        )
    pressure = np.load(staged_case / "p_boundary.npz")
    for name, expected in expected_p.items():
        np.testing.assert_allclose(
            pressure[name],
            expected,
            rtol=TOLERANCE,
            atol=TOLERANCE,
            err_msg=f"{staged_case.name}: p on patch {name}",
        )


@pytest.mark.parametrize(
    "staged_case, expected_u, expected_p, expected_notices", CASES, indirect=["staged_case"]
)
def test_approximated_patch_types_announce_the_approximation(
    staged_case: Path,
    expected_u: dict[str, tuple[float, float, float]],
    expected_p: dict[str, float],
    expected_notices: tuple[str, ...],
) -> None:
    """Every degraded translation prints a notice naming its patch and what it dropped."""
    result = _read_fields(staged_case)

    assert result.returncode == 0, f"field read failed:\n{result.stdout}\n{result.stderr}"
    output = result.stdout + result.stderr
    for notice in expected_notices:
        assert notice in output, f"{staged_case.name}: missing notice {notice!r}"


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
