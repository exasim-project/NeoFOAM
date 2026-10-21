# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Reopening a case: what is read off disk and what it does to the wizard forms.

The input is the checked-in ``simulationType laminar`` PIMPLE case; a test that
changes it works on a copy. A case's fvSchemes validate as the Pimple *and* the
Simple slices, so both models count as filled and Python's set order used to pick
the algorithm: the decision is therefore asserted with the filled models in both
orders, and the end-to-end tests start from the *other* algorithm.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

pytest.importorskip("pybFoam")
pytest.importorskip("trame")
pytest.importorskip("trame_flow")  # the wizard fixture renders the sweep canvas

from trame.app import get_server  # noqa: E402

from neofoam.mcp.registry import resolve_solver  # noqa: E402
from neofoam.ui import build_app  # noqa: E402
from neofoam.ui.case_load import (  # noqa: E402
    apply_configs_to_forms,
    case_advection_model,
    case_algorithm,
    models_to_select,
    read_case_configs,
)
from neofoam.ui.steps import (  # noqa: E402
    build_model_families,
    select_model_state,
    selection_key,
)

#: A case whose ``system/snappyHexMeshDict`` includes OpenFOAM's shipped
#: ``snappyHexMeshDict.cfg``, which ends in ``#include "<system>/meshQualityDict"``.
_ETC_INCLUDE_CASE = Path(__file__).resolve().parent / "cases" / "etc_include"

#: A checked-in ``simulationType laminar`` case (read-only).
_LAMINAR_CASE = Path(__file__).resolve().parents[1] / "setup_pimple"
_TEST_ROOT = Path(__file__).resolve().parents[1]
#: A checked-in steady SIMPLE case (read-only): its fvSolution has no ``*Final`` solvers.
_STEADY_CASE = _TEST_ROOT / "solver/incompressibleFluidNeoN/cases/pitzDailySteady"

#: Checked-in cases (read-only) → the algorithm their fvSolution control block names.
_ALGORITHM_CASES = [
    pytest.param("setup_pimple", "Pimple", "Simple", id="PIMPLE"),
    pytest.param(
        "solver/incompressibleFluidNeoN/cases/pitzDailySteady", "Simple", "Pimple", id="SIMPLE"
    ),
    pytest.param(
        "solver/incompressibleFluid/models/pressure_velocity/cases/piso_cavity",
        "Pimple",
        "Simple",
        id="PISO",
    ),
]

#: Checked-in VoF cases (read-only) → the advection model their fvSolution is evidence of:
#: the ``advectionScheme`` key, else isoAdvector-only controls in the alpha solver block,
#: else MULES — the solver's own fallback.
_VOF_CASES = "solver/incompressibleVoF"
_ADVECTION_CASES = [
    pytest.param(
        f"{_VOF_CASES}/models/alpha_advection/cases/damBreak_isoAdvector", "isoAdvector", id="key"
    ),
    pytest.param(
        f"{_VOF_CASES}/cases/interIsoFoam_discInConstantFlow", "isoAdvector", id="controls"
    ),
    pytest.param(f"{_VOF_CASES}/models/alpha_advection/cases/damBreak_mules", "MULES", id="none"),
]


def _form_data(server, cls_name: str) -> dict:
    entries = {e.cls_name: e for e in server.controller.get_entries() if e.kind == "dict"}
    return server.state[entries[cls_name].state_key]


def _turbulence_properties(server) -> dict:
    return _form_data(server, "TurbulencePropertiesConfig")


def test_read_case_configs_reads_a_case_that_includes_the_shipped_snappy_cfg(solver):
    # The reported failure: the .cfg's `<system>` tag expands from $FOAM_CASE, which
    # argList sets for a solver and nothing sets here — unset, the include resolved
    # against OpenFOAM's own etc/ and the read aborted the process outright.
    read_case_configs(_ETC_INCLUDE_CASE, solver)


def test_read_case_configs_leaves_out_a_file_the_case_does_not_have(tmp_path, solver):
    case = shutil.copytree(_LAMINAR_CASE, tmp_path / "case")
    (case / "constant" / "transportProperties").unlink()

    configs = read_case_configs(case, solver)

    names = {type(config).__name__ for config in configs}
    assert "TransportPropertiesConfig" not in names
    assert {"ControlDictConfig", "TurbulencePropertiesConfig"} <= names


def _boussinesq_entry(wizard):
    return next(e for e in wizard.controller.get_entries() if e.cls_name == "BoussinesqConfig")


def test_loading_a_case_deselects_an_optional_model_it_does_not_fill(wizard, solver):
    # Editing a buoyant case and then loading a non-buoyant one must not leave the
    # Boussinesq data live and selected: the next Save would write the gravity and
    # p_rgh configs of the case before into the case just loaded.
    families = build_model_families(solver)
    entries = wizard.controller.get_entries()
    wizard.state.update(select_model_state(families, "boussinesq"))
    boussinesq = _boussinesq_entry(wizard)
    wizard.state[boussinesq.state_key] = {"beta": 3e-3, "TRef": 300}

    apply_configs_to_forms(
        wizard.state, entries, families, read_case_configs(_LAMINAR_CASE, solver), _LAMINAR_CASE
    )

    assert wizard.state[selection_key("boussinesq")] is False
    assert wizard.state[boussinesq.state_key] == dict(boussinesq.defaults)


def test_an_incremental_fill_keeps_the_forms_it_does_not_cover(wizard, solver):
    # Without a case_dir the configs are one partial fill (the AI's own output), not a
    # whole case, so the forms it does not name keep what the user put in them.
    families = build_model_families(solver)
    entries = wizard.controller.get_entries()
    wizard.state.update(select_model_state(families, "boussinesq"))
    boussinesq = _boussinesq_entry(wizard)
    wizard.state[boussinesq.state_key] = {"beta": 3e-3, "TRef": 300}

    apply_configs_to_forms(
        wizard.state, entries, families, read_case_configs(_LAMINAR_CASE, solver)
    )

    assert wizard.state[selection_key("boussinesq")] is True
    assert wizard.state[boussinesq.state_key] == {"beta": 3e-3, "TRef": 300}


def test_a_load_leaves_every_pick_one_family_with_one_member_selected(wizard, solver):
    # A family member is gated like an optional model but must never be switched off by
    # the replacement pass: "no algorithm" is not a case the wizard can save.
    families = build_model_families(solver)
    configs = read_case_configs(_LAMINAR_CASE, solver)

    apply_configs_to_forms(
        wizard.state, wizard.controller.get_entries(), families, configs, _LAMINAR_CASE
    )

    for family in families:
        selected = [c.name for c in family.members if wizard.state[selection_key(c.name)]]
        assert len(selected) == 1, f"{family.name}: {selected}"


def test_loaded_models_include_the_turbulence_choice_the_load_moved(wizard, solver):
    # turbulenceProperties has no owning model, so the move is invisible to
    # models_to_select — the summary the caller prints still has to name it.
    families = build_model_families(solver)
    wizard.state.update(select_model_state(families, "kEpsilon"))
    configs = read_case_configs(_LAMINAR_CASE, solver)

    selected = apply_configs_to_forms(
        wizard.state, wizard.controller.get_entries(), families, configs, _LAMINAR_CASE
    )

    assert wizard.state.choice_momentumTransportModel == "laminar"
    assert "laminar" in selected


def test_loaded_case_fills_the_forms_with_the_values_on_disk(wizard, solver):
    configs = read_case_configs(_LAMINAR_CASE, solver)

    apply_configs_to_forms(
        wizard.state, wizard.controller.get_entries(), build_model_families(solver), configs
    )

    assert _form_data(wizard, "TransportPropertiesConfig") == {
        "transportModel": "Newtonian",
        "nu": 0.01,
    }
    assert _form_data(wizard, "ControlDictConfig")["deltaT"] == 0.005


def test_loaded_steady_case_fills_the_simple_fv_solution_form(wizard, solver):
    # A SIMPLE case has no <field>Final solvers: they belong to PIMPLE's final iteration.
    configs = read_case_configs(_STEADY_CASE, solver)

    apply_configs_to_forms(
        wizard.state, wizard.controller.get_entries(), build_model_families(solver), configs
    )

    assert _form_data(wizard, "Simple_fvSolution")["solvers"]["U"]["solver"] == "PBiCGStab"


def test_loaded_case_overrides_the_default_turbulence_properties(wizard, solver):
    configs = read_case_configs(_LAMINAR_CASE, solver)

    apply_configs_to_forms(
        wizard.state, wizard.controller.get_entries(), build_model_families(solver), configs
    )

    assert _turbulence_properties(wizard) == {"simulationType": "laminar"}


def test_loaded_case_moves_the_turbulence_choice_to_the_loaded_model(wizard, solver):
    # turbulenceProperties is owned by no single model, so filling it selects none:
    # the radio group has to be moved to the model the loaded file names.
    configs = read_case_configs(_LAMINAR_CASE, solver)

    apply_configs_to_forms(
        wizard.state, wizard.controller.get_entries(), build_model_families(solver), configs
    )

    assert wizard.state.choice_momentumTransportModel == "laminar"
    assert wizard.state.sel_laminar is True
    assert wizard.state.sel_kEpsilon is False


@pytest.mark.parametrize("filled", [["Pimple", "Simple"], ["Simple", "Pimple"]])
def test_models_to_select_follows_the_case_algorithm_in_either_fill_order(solver, filled):
    selected = models_to_select(filled, build_model_families(solver), "Pimple")

    assert selected == ["Pimple"]


def test_models_to_select_leaves_an_undecided_family_alone(solver):
    selected = models_to_select(
        ["Simple", "boussinesq", "Pimple"], build_model_families(solver), None
    )

    assert selected == ["boussinesq"]


def test_models_to_select_takes_the_only_filled_member_without_a_case(solver):
    selected = models_to_select(["Simple"], build_model_families(solver), None)

    assert selected == ["Simple"]


@pytest.mark.parametrize(("case", "algorithm", "_other"), _ALGORITHM_CASES)
def test_case_algorithm_reads_the_fv_solution_control_block(wizard, case, algorithm, _other):
    assert case_algorithm(_TEST_ROOT / case, wizard.controller.get_entries()) == algorithm


def test_case_algorithm_is_none_without_an_fv_solution(tmp_path, wizard):
    assert case_algorithm(tmp_path, wizard.controller.get_entries()) is None


@pytest.mark.parametrize(("case", "algorithm", "other"), _ALGORITHM_CASES)
def test_loaded_case_moves_the_algorithm_choice_to_its_control_block(
    wizard, solver, case, algorithm, other
):
    families = build_model_families(solver)
    wizard.state.update(select_model_state(families, other))
    configs = read_case_configs(_TEST_ROOT / case, solver)

    selected = apply_configs_to_forms(
        wizard.state, wizard.controller.get_entries(), families, configs, _TEST_ROOT / case
    )

    assert wizard.state.choice_PressureVelocityAlgorithm == algorithm
    assert other not in selected


def test_loaded_configs_without_a_case_keep_the_current_algorithm(wizard, solver):
    families = build_model_families(solver)
    wizard.state.update(select_model_state(families, "Simple"))
    configs = read_case_configs(_LAMINAR_CASE, solver)

    apply_configs_to_forms(wizard.state, wizard.controller.get_entries(), families, configs)

    assert wizard.state.choice_PressureVelocityAlgorithm == "Simple"


@pytest.mark.parametrize(("case", "advection"), _ADVECTION_CASES)
def test_case_advection_model_reads_the_fv_solution_evidence(case, advection):
    assert case_advection_model(_TEST_ROOT / case) == advection


def test_case_advection_model_is_mules_without_an_fv_solution(tmp_path):
    assert case_advection_model(tmp_path) == "MULES"


@pytest.mark.parametrize(
    ("case", "current", "advection"),
    [
        pytest.param(_ADVECTION_CASES[0].values[0], "MULES", "isoAdvector", id="key"),
        pytest.param(_ADVECTION_CASES[1].values[0], "MULES", "isoAdvector", id="controls"),
        pytest.param(_ADVECTION_CASES[2].values[0], "isoAdvector", "MULES", id="none"),
    ],
)
def test_loaded_vof_case_moves_the_advection_choice_to_its_evidence(
    request, case, current, advection
):
    vof = resolve_solver("incompressibleVoF")
    server = build_app(
        server=get_server(request.node.name), solver_name="incompressibleVoF", plugins=[]
    )
    families = build_model_families(vof)
    server.state.update(select_model_state(families, current))
    configs = read_case_configs(_TEST_ROOT / case, vof)

    apply_configs_to_forms(
        server.state, server.controller.get_entries(), families, configs, _TEST_ROOT / case
    )

    assert server.state.choice_advectionModel == advection
    assert server.state.sel_MULES is (advection == "MULES")
