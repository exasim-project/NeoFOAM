# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Spec for ``compute_alpha_courant_number`` — the interface (alpha) Courant
number of ``incompressibleVoF``, a pure-Python transcription of OpenFOAM's
``alphaCourantNo.H``.

The reference is native ``interFoam`` itself, taken through its own log. No C++
oracle is bound for this: ``alphaCourantNo.H`` is a solver include, not a
library call, and the project rule is that composite algorithms live in Python
composed from pybFoam primitives — so adding a binding just to check a
transcription would invert that. Instead the ``interFoam`` fixture runs the
damBreak tutorial for five fixed time steps, parses the ``Interface Courant
Number mean: ... max: ...`` line the solver prints at the top of every step, and
re-evaluates the Python function on exactly the state that line was computed
from (``alpha.water`` + ``phi`` read back from the matching time directory, the
run's fixed ``deltaT`` re-imposed). The only disagreement possible is the log's
6 significant digits, hence ``rtol=1e-5``; the fields themselves are written
``binary`` and read back bit-identical.

That reference is sharp about the thing worth pinning: the mask is
``interfaceProperties::nearInterface()`` on **cell** values, applied outside the
face sum. At ``t = 0.001`` the MULES-advected interface is still sharp, no cell
lies in ``[0.01, 0.99]`` and interFoam reports exactly 0 — while a mask built on
``interpolate(alpha1)`` would see 0.5 on every interface face and report 0.075.

The remaining scenarios need no reference at all: on the ``row4`` mesh every
expected number is a hand-derived literal. ``row4`` is a unit cube cut into 4
cells along x — cell volume exactly 0.25, x-face area exactly 1 — so with
``U = (1 0 0)`` every face flux is exactly +-1 on the x-faces and 0 elsewhere,
``surfaceSum(mag(phi))`` is exactly 2 for every cell (interior cells from their
two internal x-faces, end cells from one internal and one boundary x-face), and
with ``deltaT = 0.25`` the expectations ``0.5*max(sumPhi/V)*deltaT`` and
``0.5*(sum(sumPhi)/sum(V))*deltaT`` are exact decimals; the 1e-12 tolerance only
covers OpenFOAM's geometric cell-volume computation.

Every scenario runs in a worker subprocess: one ``Foam::Time`` (and one mesh)
per process is a hard OpenFOAM constraint. ``_alpha_courant_worker.py`` takes
one synthetic mesh and evaluates all its scenarios in one pass;
``_native_alpha_courant_worker.py`` takes one time directory of the interFoam
case (the start time is itself a ``controlDict`` entry, so it cannot be varied
within a process).

End-to-end, the consequence of this number — the adaptive-dt sequence it drives
— is already covered by ``test_damBreak_comparison.py`` and
``test_damBreak_isoAdvector_comparison.py``, which require field agreement with
interFoam/interIsoFoam to 1e-10 and would diverge on the first differing step.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from numpy.testing import assert_allclose

_HERE = Path(__file__).parent
_CASES = _HERE / "cases" / "alphaCourant"
_TUTORIAL = _HERE.parents[2] / "tutorials" / "damBreak"
_WORKER = _HERE / "_alpha_courant_worker.py"
_NATIVE_WORKER = _HERE / "_native_alpha_courant_worker.py"

# Uniform U = (1 0 0) on the row4 mesh: |phi| = 1 on every x-face, 0 elsewhere.
_UNIT_X = [1.0, 0.0, 0.0]

# alpha.water per cell, along the 4-cell row. Only the band
# 0.01 <= alpha <= 0.99 (pos0 is ">= 0") marks a cell as near-interface.
ROW4_SCENARIOS: dict[str, dict[str, Any]] = {
    "uniform_dry": {"alpha": [0.0, 0.0, 0.0, 0.0], "U": _UNIT_X},
    "uniform_wet": {"alpha": [1.0, 1.0, 1.0, 1.0], "U": _UNIT_X},
    "sharp_step": {"alpha": [0.0, 1.0, 1.0, 1.0], "U": _UNIT_X},
    "band_edges": {"alpha": [0.01, 0.99, 0.009, 0.991], "U": _UNIT_X},
    "graded_interface": {"alpha": [0.0, 0.5, 0.5, 1.0], "U": _UNIT_X},
    "lone_interface_cell": {"alpha": [0.0, 0.5, 1.0, 1.0], "U": _UNIT_X},
}

SINGLE_CELL_SCENARIOS: dict[str, dict[str, Any]] = {
    # alpha = 0.5 is squarely inside the band, so only the nInternalFaces()
    # early return can make this zero.
    "interface_cell": {"alpha": [0.5], "U": _UNIT_X},
}

# system/controlDict of cases/alphaCourant/interFoam: fixed dt, every step
# written. The state behind the line printed before "Time = t + deltaT" is the
# time directory t, so the last written time (0.005) has no printed counterpart
# and time 0 has no phi on disk (interFoam creates it from U at startup).
NATIVE_DELTA_T = 0.001
NATIVE_TIMES = ["0.001", "0.002", "0.003", "0.004"]

_INTERFACE_COURANT = re.compile(r"^Interface Courant Number mean: (\S+)\s+max: (\S+)\s*$")


def _stage(mesh: str, dest: Path) -> Path:
    """Copy the checked-in case inputs for ``mesh``; the worker meshes on top."""
    shutil.copytree(_CASES / "common", dest)
    shutil.copy(_CASES / mesh / "blockMeshDict", dest / "system")
    shutil.copytree(dest / "0.orig", dest / "0")
    return dest


def _evaluate(mesh: str, scenarios: dict[str, Any], tmp_path: Path) -> dict[str, Any]:
    """Run every scenario of ``mesh`` in one worker process; return its results."""
    case = _stage(mesh, tmp_path / "case")
    request = case / "request.json"
    request.write_text(json.dumps({"scenarios": scenarios}))
    subprocess.run(
        [sys.executable, str(_WORKER), str(case), str(request)],
        check=True,
        capture_output=True,
        text=True,
    )
    return dict(json.loads((case / "result.json").read_text()))


def _run_interfoam(case: Path) -> Path:
    """Stage the damBreak tutorial with our controlDict and run it natively."""
    shutil.copytree(_TUTORIAL, case)
    shutil.copytree(case / "0.orig", case / "0")
    shutil.copy(_CASES / "interFoam" / "system" / "controlDict", case / "system")
    for tool in ("blockMesh", "setFields", "interFoam"):
        completed = subprocess.run(
            [tool, "-case", str(case)],
            check=True,
            capture_output=True,
            text=True,
            timeout=600,
        )
        (case / f"log.{tool}").write_text(completed.stdout)
    return case


def _parse_interface_courant(log: str) -> dict[str, list[float]]:
    """Map each time directory to the (max, mean) interFoam printed *from* it.

    ``alphaCourantNo.H`` runs at the top of the step, before ``++runTime``, so a
    printed line belongs to the time named by the *previous* ``Time =`` line —
    the run's start time for the first one.
    """
    printed: dict[str, list[float]] = {}
    state_time = "0"
    for line in log.splitlines():
        match = _INTERFACE_COURANT.match(line)
        if match:
            printed[state_time] = [float(match[2]), float(match[1])]
        elif line.startswith("Time = "):
            state_time = line.split("=", 1)[1].strip()
    return printed


def _evaluate_native(case: Path, time_name: str) -> list[float]:
    """Run ``compute_alpha_courant_number`` on one written time of ``case``."""
    subprocess.run(
        [
            sys.executable,
            str(_NATIVE_WORKER),
            str(case),
            time_name,
            str(NATIVE_DELTA_T),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return list(json.loads((case / f"alphaCo.{time_name}.json").read_text()))


@pytest.fixture(scope="module")
def row4(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    return _evaluate("row4", ROW4_SCENARIOS, tmp_path_factory.mktemp("row4"))


@pytest.fixture(scope="module")
def single_cell(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    return _evaluate("single_cell", SINGLE_CELL_SCENARIOS, tmp_path_factory.mktemp("single_cell"))


@pytest.fixture(scope="module")
def interfoam(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, tuple[list[float], list[float]]]:
    """Per written time: what interFoam printed, and what Python computes."""
    case = _run_interfoam(tmp_path_factory.mktemp("interFoam") / "case")
    printed = _parse_interface_courant((case / "log.interFoam").read_text())
    return {t: (printed[t], _evaluate_native(case, t)) for t in NATIVE_TIMES}


@pytest.mark.parametrize("scenario", ["uniform_dry", "uniform_wet"])
def test_uniform_alpha_yields_exactly_zero(row4: dict[str, Any], scenario: str) -> None:
    # No cell is inside [0.01, 0.99], so no face contributes at all — the two
    # numbers are structurally zero, not merely small.
    assert row4[scenario] == [0.0, 0.0]


def test_sharp_step_between_saturated_cells_yields_zero(row4: dict[str, Any]) -> None:
    # alpha = [0, 1, 1, 1]: the interface sits on a face, but no *cell* value is
    # inside the band, so alphaCourantNo.H reports nothing.
    assert row4["sharp_step"] == [0.0, 0.0]


def test_band_edges_are_inclusive(row4: dict[str, Any]) -> None:
    # alpha = [0.01, 0.99, 0.009, 0.991]: pos0 is ">= 0", so the two cells
    # exactly on the edges count and the two just outside do not.
    # max: 0.5*(2/0.25)*0.25 = 1.0; mean: 0.5*((2+2)/1.0)*0.25 = 0.5.
    assert_allclose(
        row4["band_edges"],
        [1.0, 0.5],
        rtol=1e-12,
        err_msg="row4/band_edges: pos0 band must include 0.01 and 0.99 exactly",
    )


def test_known_alpha_courant_number_on_the_four_cell_row(row4: dict[str, Any]) -> None:
    # alpha = [0, 0.5, 0.5, 1]: cells 1 and 2 are near-interface, each summing
    # |phi| = 1 over two x-faces. max: 0.5*(2/0.25)*0.25 = 1.0;
    # mean: 0.5*((0+2+2+0)/1.0)*0.25 = 0.5.
    assert_allclose(
        row4["graded_interface"],
        [1.0, 0.5],
        rtol=1e-12,
        err_msg="row4/graded_interface: hand-computed 0.5*max(sumPhi/V)*deltaT",
    )


def test_interface_mask_is_cell_based_not_face_based(row4: dict[str, Any]) -> None:
    # alpha = [0, 0.5, 1, 1]: cell 1 alone is inside the band and contributes
    # its full sumPhi = 2. max: 0.5*(2/0.25)*0.25 = 1.0;
    # mean: 0.5*((0+2+0+0)/1.0)*0.25 = 0.25.
    # A band on interpolate(alpha1) instead would light up the faces 0|1
    # (alphaf = 0.25) and 1|2 (alphaf = 0.75), handing cells 0 and 2 a share
    # each and doubling the mean to 0.5.
    assert_allclose(
        row4["lone_interface_cell"],
        [1.0, 0.25],
        rtol=1e-12,
        err_msg="row4/lone_interface_cell: nearInterface() masks cells, not faces",
    )


def test_mesh_without_internal_faces_yields_zero(single_cell: dict[str, Any]) -> None:
    # One cell, six boundary faces: surfaceSum over the boundary alone would
    # give 0.5*(2/1.0)*0.25 = 0.25, so this pins the early return.
    assert single_cell["interface_cell"] == [0.0, 0.0]


def test_interfoam_reference_is_not_vacuous(
    interfoam: dict[str, tuple[list[float], list[float]]],
) -> None:
    # Guards the comparison below against passing on all-zero data: the damBreak
    # column really does put cells inside the band within these five steps.
    assert max(printed[0] for printed, _ in interfoam.values()) > 0.0


@pytest.mark.parametrize("time_name", NATIVE_TIMES)
def test_matches_interfoam_on_damBreak(
    interfoam: dict[str, tuple[list[float], list[float]]], time_name: str
) -> None:
    printed, computed = interfoam[time_name]
    assert_allclose(
        computed,
        printed,
        rtol=1e-5,
        err_msg=f"damBreak t={time_name}: must reproduce alphaCourantNo.H",
    )
