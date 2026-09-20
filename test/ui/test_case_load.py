# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Reopening a case: what is read off disk and what it does to the wizard forms.

The input is the checked-in ``simulationType laminar`` PIMPLE case; a test that
changes it works on a copy. Which pressure-velocity algorithm a load selects is
left out on purpose: the case's fvSchemes/fvSolution validate as the Pimple *and*
the Simple slices, so both count as filled.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

pytest.importorskip("pybFoam")
pytest.importorskip("trame")
pytest.importorskip("trame_flow")  # the wizard fixture renders the sweep canvas

from neofoam.ui.case_load import apply_configs_to_forms, read_case_configs  # noqa: E402
from neofoam.ui.steps import build_model_families  # noqa: E402

#: A checked-in ``simulationType laminar`` case (read-only).
_LAMINAR_CASE = Path(__file__).resolve().parents[1] / "setup_pimple"


def _form_data(server, cls_name: str) -> dict:
    entries = {e.cls_name: e for e in server.controller.get_entries() if e.kind == "dict"}
    return server.state[entries[cls_name].state_key]


def _turbulence_properties(server) -> dict:
    return _form_data(server, "TurbulencePropertiesConfig")


def test_read_case_configs_leaves_out_a_file_the_case_does_not_have(tmp_path, solver):
    case = shutil.copytree(_LAMINAR_CASE, tmp_path / "case")
    (case / "constant" / "transportProperties").unlink()

    configs = read_case_configs(case, solver)

    names = {type(config).__name__ for config in configs}
    assert "TransportPropertiesConfig" not in names
    assert {"ControlDictConfig", "TurbulencePropertiesConfig"} <= names


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
