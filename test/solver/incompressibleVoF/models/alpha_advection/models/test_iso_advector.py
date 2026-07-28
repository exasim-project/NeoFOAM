# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Spec for the isoAdvector member: the porosity field it reads, and its alpha pass.

Part one — **build**. ``interIsoFoam``'s ``createFields.H`` includes
``createPorosity.H``, which reads ``0/porosity`` and registers it on the mesh
*before* the advector is built:
``Foam::isoAdvection``'s constructor looks that field up in the object registry
whenever ``constant/porosityProperties`` sets ``porosityEnabled``, and calls
``FatalError`` when it is missing. So the field is not decoration — without it a
porous case cannot construct at all.

Two levels, mirroring ``test/solver/incompressibleVoF/test_create_fields.py``:

* the **step list** (is the read there, and is the advector ordered behind it?)
  is pure description and is asserted in-process, against the two real
  fvSolutions under ``../cases/`` — the porosity-free ``damBreak_isoAdvector``
  pins that a case without ``constant/porosityProperties`` gets no extra step;
* the **field it produces** comes from one executed pipeline of
  ``cases/vofRow4Porous`` (see ``test/solver/incompressibleVoF/conftest.py``:
  one ``Foam::Time`` per process, so it runs in a worker subprocess). That the
  worker returns at all is the proof that ``isoAdvection`` constructed.

``cases/vofRow4Porous`` is the 4-cell row of ``cases/vofRow4`` switched to
``advectionScheme isoAdvector`` plus ``constant/porosityProperties``
(``porosityEnabled true``) and a per-cell ``0/porosity`` of ``(1 0.75 0.5
0.25)`` — read back exactly, so no tolerance is needed.

Part two — **the alpha pass** (``alphaEqn.H`` + ``alphaEqnSubCycle.H``), through
``_iso_advector_worker.py``, which swaps the advector for a recorder and dumps
the ``Foam::Time`` state and the velocity field each ``advect()`` is entered
with. Two cases, one per behaviour, run as data through the same worker:

* ``cases/vofRow4IsoAdvectorMoving`` — ``cases/vofRow4Moving``'s oscillating
  4-cell row switched to ``advectionScheme isoAdvector`` with
  ``nAlphaSubCycles 3``. It is the only combination that exercises both gaps:
  isoAdvection interpolates ``U`` onto the iso-face centres for the interface
  normal velocity, so on a moving mesh ``alphaEqn.H`` has to hand it
  ``U - fvc::reconstruct(mesh.phi())``; and the pass has to run three times over
  a sub-cycled ``Foam::Time``. ``deltaT 0.25`` with ``adjustTimeStep`` off makes
  the expected sub-step times exact thirds.
* ``cases/vofRow4Porous`` — static mesh, ``nAlphaSubCycles 1``: the
  no-sub-cycle, no-mesh-motion direction, so neither behaviour can be a
  constant.

Velocity tolerances are ``atol=1e-15`` on a field whose mesh-motion peak is
0.0765 m/s: the assertion recomputes the *same* IEEE subtraction the code does,
so only double round-off is in play.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from neofoam.solver.incompressibleVoF.create_fields import create_init
from neofoam.solver.incompressibleVoF.models.alpha_advection.models.iso_advector import (
    read_n_alpha_sub_cycles,
)

from ....conftest import BuiltCase

_ADVECTION_CASES = Path(__file__).parents[1] / "cases"
_POROUS_CASE = Path(__file__).parents[3] / "cases" / "vofRow4Porous"
_MOVING_CASE = Path(__file__).parents[3] / "cases" / "vofRow4IsoAdvectorMoving"
_ISO_ADVECTOR_WORKER = Path(__file__).parent / "_iso_advector_worker.py"


def _build_steps(case: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Names of the init steps the staged pipeline describes for *case*."""
    monkeypatch.chdir(case)
    runner = create_init(case_dir=case)
    runner.run_load()
    return [step.name for step in runner.run_build()]


def test_build_reads_porosity_when_the_case_enables_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert "fields.porosity" in _build_steps(_POROUS_CASE, monkeypatch)


def test_build_omits_porosity_when_the_case_has_no_porosity_properties(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _ADVECTION_CASES / "damBreak_isoAdvector"
    assert not (case / "constant" / "porosityProperties").exists()
    assert "fields.porosity" not in _build_steps(case, monkeypatch)


def test_advector_depends_on_the_porosity_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The dependency edge is what orders the read ahead of the construction —
    # the graph is topologically sorted, step order in the list is not enough.
    monkeypatch.chdir(_POROUS_CASE)
    runner = create_init(case_dir=_POROUS_CASE)
    runner.run_load()
    step = next(s for s in runner.run_build() if s.name == "models.advector")
    assert step.depends_on == [
        "fields.alpha1",
        "fields.phi",
        "fields.U",
        "fields.porosity",
    ]


def test_isoadvection_is_constructed_on_a_porous_case(
    vof_row4_porous: BuiltCase,
) -> None:
    # isoAdvection aborts the process when porosity is enabled and the field is
    # not in the registry, so a live advector model is the whole point.
    assert "advector" in vof_row4_porous.result["model_keys"]


def test_porosity_is_read_from_the_cases_zero_directory(
    vof_row4_porous: BuiltCase,
) -> None:
    # 0/porosity is the literal (1 0.75 0.5 0.25), registered under its own name
    # (isoAdvection looks it up as "porosity", not by the Context key).
    assert vof_row4_porous.internal("porosity") == [1.0, 0.75, 0.5, 0.25]
    assert vof_row4_porous.result["registered_names"]["porosity"] == "porosity"


# --- alphaControls.H: the one control interIsoFoam reads -------------------


@pytest.mark.parametrize(
    "case, expected",
    [
        pytest.param(_MOVING_CASE, 3, id="nAlphaSubCycles_3"),
        pytest.param(_POROUS_CASE, 1, id="nAlphaSubCycles_1"),
    ],
)
def test_n_alpha_sub_cycles_is_read_from_the_cases_alpha_solver_dict(
    monkeypatch: pytest.MonkeyPatch, case: Path, expected: int
) -> None:
    # Resolved through OpenFOAM's own regex dict-key matching, so the real
    # "alpha.water.*" entry answers for the field named alpha.water.
    monkeypatch.chdir(case)
    assert read_n_alpha_sub_cycles("alpha.water") == expected


# --- alphaEqn.H + alphaEqnSubCycle.H: one live alpha_advection call --------


@pytest.fixture(scope="module")
def iso_advector_runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, dict[str, Any]]:
    """One ``alpha_advection`` call per case; what the advector and Time saw."""
    runs: dict[str, dict[str, Any]] = {}
    for source in (_MOVING_CASE, _POROUS_CASE):
        case = tmp_path_factory.mktemp(source.name) / "case"
        shutil.copytree(source, case)
        subprocess.run(
            ["blockMesh", "-case", str(case)],
            check=True,
            capture_output=True,
            text=True,
            timeout=300,
        )
        subprocess.run(
            [sys.executable, str(_ISO_ADVECTOR_WORKER), str(case)],
            check=True,
            capture_output=True,
            text=True,
            timeout=300,
        )
        runs[source.name] = json.loads((case / "iso_advector.json").read_text())
    return runs


@pytest.mark.parametrize(
    "case_name, expected_sub_step_times",
    [
        pytest.param("vofRow4IsoAdvectorMoving", [0.25 / 3, 0.5 / 3, 0.25], id="sub_cycled"),
        pytest.param("vofRow4Porous", [0.25], id="not_sub_cycled"),
    ],
)
def test_the_advector_runs_once_per_alpha_sub_cycle(
    iso_advector_runs: dict[str, dict[str, Any]],
    case_name: str,
    expected_sub_step_times: list[float],
) -> None:
    # nAlphaSubCycles was ignored on this path: one advect() per time step at the
    # full deltaT, however many the case asked for.
    times = [step["time"] for step in iso_advector_runs[case_name]["sub_steps"]]
    assert times == pytest.approx(expected_sub_step_times)


def test_each_sub_step_advects_over_a_fraction_of_the_time_step(
    iso_advector_runs: dict[str, dict[str, Any]],
) -> None:
    # isoAdvection reads mesh.time().deltaT() itself, so the sub-step length is
    # what actually shortens the advection — not a scaled flux.
    deltas = [step["deltaT"] for step in iso_advector_runs["vofRow4IsoAdvectorMoving"]["sub_steps"]]
    assert deltas == pytest.approx([0.25 / 3] * 3)


def test_the_sub_cycle_restores_the_time_state(
    iso_advector_runs: dict[str, dict[str, Any]],
) -> None:
    # endSubCycle() puts back time, deltaT and index, so the momentum/pressure
    # stage after the alpha solve runs on the real time step.
    run = iso_advector_runs["vofRow4IsoAdvectorMoving"]
    assert run["after"] == run["before"]


def test_the_advector_sees_the_velocity_relative_to_the_mesh_motion(
    iso_advector_runs: dict[str, dict[str, Any]],
) -> None:
    """``alphaEqn.H``'s ``U -= fvc::reconstruct(mesh.phi())`` around ``advect()``.

    isoAdvection interpolates ``U`` onto the iso-face centres to get the
    interface normal velocity ``Un0``; on a moving mesh that has to be the
    velocity relative to the mesh, or the interface is transported with the
    mesh's own motion on top of the flow.
    """
    run = iso_advector_runs["vofRow4IsoAdvectorMoving"]
    expected = np.asarray(run["U_before"]) - np.asarray(run["mesh_velocity"])
    np.testing.assert_allclose(
        np.asarray(run["sub_steps"][0]["U"]),
        expected,
        rtol=0,
        atol=1e-15,
        err_msg="vofRow4IsoAdvectorMoving: advect() saw the absolute velocity",
    )
    # Non-vacuous: the mesh really is moving, so relative != absolute.
    assert np.abs(np.asarray(run["mesh_velocity"])).max() > 0.07


def test_the_velocity_is_absolute_again_after_the_alpha_pass(
    iso_advector_runs: dict[str, dict[str, Any]],
) -> None:
    # alphaEqn.H undoes the subtraction before UEqn.H assembles the momentum
    # equation, which needs the absolute velocity.
    run = iso_advector_runs["vofRow4IsoAdvectorMoving"]
    # Non-vacuous: U really was relative while the advector held it, so
    # "restored" is a round trip and not a field that never moved.
    assert not np.allclose(np.asarray(run["U_after"]), np.asarray(run["sub_steps"][-1]["U"]))
    np.testing.assert_allclose(
        np.asarray(run["U_after"]),
        np.asarray(run["U_before"]),
        rtol=0,
        atol=1e-15,
        err_msg="vofRow4IsoAdvectorMoving: U left relative to the mesh motion",
    )


def test_a_static_mesh_hands_the_advector_the_velocity_untouched(
    iso_advector_runs: dict[str, dict[str, Any]],
) -> None:
    # fvc::reconstruct(mesh.phi()) is not merely zero on a static mesh — there is
    # no mesh flux at all — so the bracket must not run.
    run = iso_advector_runs["vofRow4Porous"]
    assert run["moving"] is False
    assert run["sub_steps"][0]["U"] == run["U_before"]
