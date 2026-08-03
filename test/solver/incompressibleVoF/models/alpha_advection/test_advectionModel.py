# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Tests for the ``advectionModel`` plugin interface and native registration.

Importing ``neofoam.solver.incompressibleVoF.incompressibleVoF`` (the solver
entrypoint) imports ``.models.alpha_advection`` with a
``# noqa: F401 (registers members)`` comment — that import runs the alpha
package's ``__init__``, which registers both bundled schemes (``MULES``,
``isoAdvector``) with the ``advectionModel`` family as an import side effect.
This module proves that side effect really did happen (not just that one
scheme registered) and exercises the read-only registry queries +
case-detection factory built on top of it.
"""

from pathlib import Path

import pytest

# Importing the solver entrypoint registers both bundled advection schemes
# with the advectionModel family (import side effect - the whole point of
# this test module).
import neofoam.solver.incompressibleVoF.incompressibleVoF  # noqa: F401
from neofoam.framework.model import ModelSpec
from neofoam.solver.incompressibleVoF.models.alpha_advection.advectionModel import (
    advectionModel,
)

_CASES = Path(__file__).parent / "cases"


def test_registered_names_includes_both_bundled_schemes() -> None:
    assert {"MULES", "isoAdvector"} <= set(advectionModel.registered_names())


def test_all_specs_returns_model_spec_objects_for_both_schemes() -> None:
    specs = advectionModel.all_specs()
    names = {spec.name for spec in specs}
    assert {"MULES", "isoAdvector"} <= names
    assert all(isinstance(spec, ModelSpec) for spec in specs)


@pytest.mark.parametrize("name", ["MULES", "isoAdvector"])
def test_find_spec_returns_the_matching_registered_spec(name: str) -> None:
    spec = advectionModel.find_spec(name)
    assert isinstance(spec, ModelSpec)
    assert spec.name == name


def test_find_spec_returns_none_for_an_unknown_name() -> None:
    assert advectionModel.find_spec("bogusScheme") is None


def test_detect_and_create_selects_mules_from_the_real_damBreak_mules_case(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The interFoam damBreak fvSolution has no ``advectionScheme`` key."""
    monkeypatch.chdir(_CASES / "damBreak_mules")
    spec = advectionModel.detect_and_create()
    assert spec.name == "MULES"


def test_detect_and_create_selects_isoadvector_from_the_real_case(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The damBreak_isoAdvector fvSolution carries ``advectionScheme isoAdvector;``."""
    monkeypatch.chdir(_CASES / "damBreak_isoAdvector")
    spec = advectionModel.detect_and_create()
    assert spec.name == "isoAdvector"
