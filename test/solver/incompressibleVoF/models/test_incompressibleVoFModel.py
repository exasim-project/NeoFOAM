# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Tests for the ``incompressibleVoFModel`` optional-model plugin interface.

This is a minimal port (mirrors ``incompressibleFluidModel``): nothing in the
current tree registers an optional model with this family, so ``all_specs``
and ``detect_models`` are always empty. The point of these tests is to pin
that "empty registry, not a crash" behaviour, and that the family is
registered with the ``PluginSystem`` regardless of whether any member ever
joins it.
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


def test_all_specs_is_empty_when_no_optional_model_is_registered() -> None:
    assert incompressibleVoFModel.all_specs() == []


def test_detect_models_returns_empty_list_for_the_real_damBreak_case() -> None:
    runtimes: list[ModelRuntime] = incompressibleVoFModel.detect_models(
        _CASES / "damBreak"
    )
    assert runtimes == []


def test_detect_models_defaults_case_dir_to_the_cwd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(_CASES / "damBreak")
    assert incompressibleVoFModel.detect_models() == []
