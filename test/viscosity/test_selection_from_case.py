# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Integration: read a real transportProperties case and select the model.

Ties config reading to selection using the bundled cases under
``test/viscosity/cases`` (no dict content is encoded here). The OpenFOAM factory
is injected so the fallback path needs no live mesh/fields. Carries
``requires_openfoam`` because reading the dictionary uses the pybFoam-backed
strategy.
"""

from pathlib import Path
from unittest.mock import MagicMock

from neofoam.framework.model import ModelSpec
from neofoam.viscosity.fallback import OpenFOAMViscosityModel
from neofoam.viscosity.interface import Model, viscosityModel
from neofoam.viscosity.selection import select_from_case

# Register the bundled natives (Newtonian).
import neofoam.viscosity.models  # noqa: F401

from viscosity.conftest import requires_openfoam


@requires_openfoam
def test_select_from_case_dispatches_native_newtonian(newtonian_case: Path) -> None:
    # Newtonian IS a registered native → dispatched, not falling back.
    selected = select_from_case(newtonian_case)
    assert isinstance(selected, ModelSpec)
    assert selected.name == "Newtonian"


@requires_openfoam
def test_select_from_case_falls_back_for_unported_model(
    cross_power_law_case: Path,
) -> None:
    # CrossPowerLaw has no native NeoFOAM model in the skeleton → OpenFOAM fallback.
    selected = select_from_case(cross_power_law_case, of_factory=MagicMock())
    assert isinstance(selected, OpenFOAMViscosityModel)


@requires_openfoam
def test_select_from_case_dispatches_registered_model(
    cross_power_law_case: Path, clean_viscosity_registry: None
) -> None:
    # Registering a native for the configured transportModel makes selection
    # dispatch to it instead of falling back to OpenFOAM.
    cross = Model("CrossPowerLaw").register_with(viscosityModel)
    selected = select_from_case(cross_power_law_case)
    assert selected is cross
