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
  the porous case (see ``test/solver/incompressibleVoF/conftest.py``:
  one ``Foam::Time`` per process, so it runs in a worker subprocess). That the
  worker returns at all is the proof that ``isoAdvection`` constructed.

The porous case is the 4-cell row of ``cases/vofRow4/common`` switched to
``advectionScheme isoAdvector`` (the conftest's ``iso_advector`` step) plus the
``porous`` overlay's ``constant/porosityProperties`` (``porosityEnabled true``)
and per-cell ``0/porosity`` of ``(1 0.75 0.5 0.25)`` — read back exactly, so no
tolerance is needed.

Part two — **the alpha pass** (``alphaEqn.H`` + ``alphaEqnSubCycle.H``), through
``_iso_advector_worker.py``, which swaps the advector for a recorder and dumps
the ``Foam::Time`` state and the velocity field each ``advect()`` is entered
with. Two cases, one per behaviour, run as data through the same worker:

* the moving case under isoAdvector with ``nAlphaSubCycles 3`` — the oscillating
  4-cell row. It is the only combination that exercises both gaps:
  isoAdvection interpolates ``U`` onto the iso-face centres for the interface
  normal velocity, so on a moving mesh ``alphaEqn.H`` has to hand it
  ``U - fvc::reconstruct(mesh.phi())``; and the pass has to run three times over
  a sub-cycled ``Foam::Time``. ``deltaT 0.25`` with ``adjustTimeStep`` off makes
  the expected sub-step times exact thirds.
* the porous case — static mesh, ``nAlphaSubCycles 1``: the
  no-sub-cycle, no-mesh-motion direction, so neither behaviour can be a
  constant.

Velocity tolerances are ``atol=1e-15`` on a field whose mesh-motion peak is
0.0765 m/s: the assertion recomputes the *same* IEEE subtraction the code does,
so only double round-off is in play.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from neofoam.solver.incompressibleVoF.create_fields import create_init
from neofoam.solver.incompressibleVoF.models.alpha_advection.models.iso_advector import (
    PorosityPropertiesConfig,
    _porosity_enabled,
    read_n_alpha_sub_cycles,
)
from neofoam.tooling.casebuild import Step

from ....conftest import VOF_ROW4, BuiltCase, build_case, iso_advector, moving, overlay

_ADVECTION_CASES = Path(__file__).parents[1] / "cases"
_ISO_ADVECTOR_WORKER = Path(__file__).parent / "_iso_advector_worker.py"

#: The case variants this module runs, as casebuild steps on ``vofRow4/common``.
_POROUS: tuple[Step, ...] = (overlay(VOF_ROW4 / "porous"), iso_advector())
_MOVING: tuple[Step, ...] = (moving(), iso_advector(n_alpha_sub_cycles=3))


@pytest.fixture(scope="module")
def porous_case(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The composed porous case, for the tests that read its dictionaries."""
    return build_case(tmp_path_factory.mktemp("porous") / "case", *_POROUS)


@pytest.fixture(scope="module")
def moving_case(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The composed sub-cycled moving isoAdvector case."""
    return build_case(tmp_path_factory.mktemp("isoAdvectorMoving") / "case", *_MOVING)


def _build_steps(case: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Names of the init steps the staged pipeline describes for *case*."""
    monkeypatch.chdir(case)
    runner = create_init(case_dir=case)
    runner.run_load()
    return [step.name for step in runner.run_build()]


def test_build_reads_porosity_when_the_case_enables_it(
    porous_case: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # ``constant/porosityProperties`` is the file createPorosity.H reads; the
    # porous overlay sets the switch on, and BUILD then carries the read.
    assert PorosityPropertiesConfig.load(case_dir=porous_case, validate=False).porosityEnabled
    assert "fields.porosity" in _build_steps(porous_case, monkeypatch)


def test_porosity_is_off_and_unread_when_the_case_has_no_porosity_properties(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The dictionary is optional, so the read is guarded: an absent file is the
    # same answer as ``porosityEnabled no``, not a FileNotFoundError — and BUILD
    # then emits no read step at all.
    case = _ADVECTION_CASES / "damBreak_isoAdvector"
    assert not (case / "constant" / "porosityProperties").exists()
    monkeypatch.chdir(case)
    assert _porosity_enabled() is False
    assert "fields.porosity" not in _build_steps(case, monkeypatch)


def test_advector_depends_on_the_porosity_field(
    porous_case: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The dependency edge is what orders the read ahead of the construction —
    # the graph is topologically sorted, step order in the list is not enough.
    monkeypatch.chdir(porous_case)
    runner = create_init(case_dir=porous_case)
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
    # not in the registry, so a live advector model is the whole point; and
    # 0/porosity is the literal (1 0.75 0.5 0.25), registered under its own name
    # (isoAdvection looks it up as "porosity", not by the Context key).
    assert "advector" in vof_row4_porous.result["model_keys"]
    assert vof_row4_porous.internal("porosity") == [1.0, 0.75, 0.5, 0.25]
    assert vof_row4_porous.result["registered_names"]["porosity"] == "porosity"


# --- alphaControls.H: the one control interIsoFoam reads -------------------


@pytest.mark.parametrize(
    "case_fixture, expected",
    [
        pytest.param("moving_case", 3, id="nAlphaSubCycles_3"),
        pytest.param("porous_case", 1, id="nAlphaSubCycles_1"),
    ],
)
def test_n_alpha_sub_cycles_is_read_from_the_cases_alpha_solver_dict(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    case_fixture: str,
    expected: int,
) -> None:
    # Resolved through OpenFOAM's own regex dict-key matching, so the real
    # "alpha.water.*" entry answers for the field named alpha.water.
    monkeypatch.chdir(request.getfixturevalue(case_fixture))
    assert read_n_alpha_sub_cycles("alpha.water") == expected


# --- alphaEqn.H + alphaEqnSubCycle.H: one live alpha_advection call --------


@pytest.fixture(scope="module")
def iso_advector_runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, dict[str, Any]]:
    """One ``alpha_advection`` call per case; what the advector and Time saw."""
    runs: dict[str, dict[str, Any]] = {}
    for name, steps in (("isoAdvectorMoving", _MOVING), ("porous", _POROUS)):
        case = build_case(tmp_path_factory.mktemp(name) / "case", *steps)
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
        runs[name] = json.loads((case / "iso_advector.json").read_text())
    return runs


@pytest.mark.parametrize(
    "case_name, expected_sub_step_times",
    [
        pytest.param("isoAdvectorMoving", [0.25 / 3, 0.5 / 3, 0.25], id="sub_cycled"),
        pytest.param("porous", [0.25], id="not_sub_cycled"),
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


def test_each_sub_step_advects_over_a_fraction_of_the_restored_time_step(
    iso_advector_runs: dict[str, dict[str, Any]],
) -> None:
    # isoAdvection reads mesh.time().deltaT() itself, so the sub-step length is
    # what actually shortens the advection — not a scaled flux. endSubCycle()
    # then puts back time, deltaT and index, so the momentum/pressure stage
    # after the alpha solve runs on the real time step.
    run = iso_advector_runs["isoAdvectorMoving"]
    deltas = [step["deltaT"] for step in run["sub_steps"]]
    assert deltas == pytest.approx([0.25 / 3] * 3)
    assert run["after"] == run["before"]


def test_the_advector_sees_the_velocity_relative_to_the_mesh_motion(
    iso_advector_runs: dict[str, dict[str, Any]],
) -> None:
    """``alphaEqn.H``'s ``U -= fvc::reconstruct(mesh.phi())`` around ``advect()``.

    isoAdvection interpolates ``U`` onto the iso-face centres to get the
    interface normal velocity ``Un0``; on a moving mesh that has to be the
    velocity relative to the mesh, or the interface is transported with the
    mesh's own motion on top of the flow. The subtraction is then undone before
    UEqn.H assembles the momentum equation, which needs the absolute velocity.
    """
    run = iso_advector_runs["isoAdvectorMoving"]
    expected = np.asarray(run["U_before"]) - np.asarray(run["mesh_velocity"])
    np.testing.assert_allclose(
        np.asarray(run["sub_steps"][0]["U"]),
        expected,
        rtol=0,
        atol=1e-15,
        err_msg="isoAdvectorMoving: advect() saw the absolute velocity",
    )
    # Non-vacuous: the mesh really is moving, so relative != absolute, and
    # "restored" below is a round trip and not a field that never moved.
    assert np.abs(np.asarray(run["mesh_velocity"])).max() > 0.07
    assert not np.allclose(np.asarray(run["U_after"]), np.asarray(run["sub_steps"][-1]["U"]))
    np.testing.assert_allclose(
        np.asarray(run["U_after"]),
        np.asarray(run["U_before"]),
        rtol=0,
        atol=1e-15,
        err_msg="isoAdvectorMoving: U left relative to the mesh motion",
    )


def test_a_static_mesh_hands_the_advector_the_velocity_untouched(
    iso_advector_runs: dict[str, dict[str, Any]],
) -> None:
    # fvc::reconstruct(mesh.phi()) is not merely zero on a static mesh — there is
    # no mesh flux at all — so the bracket must not run.
    run = iso_advector_runs["porous"]
    assert run["moving"] is False
    assert run["sub_steps"][0]["U"] == run["U_before"]
