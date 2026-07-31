# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Tests for the ``incompressibleVoFModel`` optional-model plugin interface.

Mirrors ``incompressibleFluidModel``. ``all_specs`` lists every member that has
joined the family (``mrf`` and ``fvOptions``, both shared with
``incompressibleFluid``), while
``detect_models`` runs each member's ``detect`` against the case — so a case that
asks for none of them still gets an empty list rather than a crash. The family is
registered with the ``PluginSystem`` regardless of its membership.
"""

from pathlib import Path

import pytest

from neofoam.core.plugin_system import PluginSystem
from neofoam.framework.model import ModelRuntime
from neofoam.solver.incompressibleVoF.models.incompressibleVoFModel import (
    incompressibleVoFModel,
)

_CASES = Path(__file__).parent.parent / "cases"


def test_incompressibleVoFModel_is_registered_in_the_plugin_system() -> None:
    assert PluginSystem.get_registered("incompressibleVoFModel") is not None


def test_all_specs_lists_every_registered_optional_model() -> None:
    assert [spec.name for spec in incompressibleVoFModel.all_specs()] == ["mrf", "fvOptions"]


def test_detect_models_returns_empty_list_for_the_real_damBreak_case() -> None:
    # damBreak carries neither constant/MRFProperties nor an fvOptions
    # dictionary, so both registered members detect themselves out and the
    # case's model set stays empty.
    runtimes: list[ModelRuntime] = incompressibleVoFModel.detect_models(_CASES / "damBreak")
    assert runtimes == []


def test_detect_models_defaults_case_dir_to_the_cwd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(_CASES / "damBreak")
    assert incompressibleVoFModel.detect_models() == []
