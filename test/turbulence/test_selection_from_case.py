# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Integration: read a real turbulenceProperties case and select the model.

Ties config reading to selection using the bundled cases under
``test/turbulence/cases`` (no dict content is encoded here). The OpenFOAM
factory is injected so the fallback path needs no live mesh/fields. Carries
``requires_openfoam`` because reading the dictionary uses the pybFoam-backed
strategy.
"""

from pathlib import Path
from unittest.mock import MagicMock

from neofoam.framework.model import ModelSpec
from neofoam.turbulence.fallback import OpenFOAMTurbulenceModel
from neofoam.turbulence.interface import Model, turbulenceModel
from neofoam.turbulence.selection import select_from_case

# Register the bundled natives (laminar).
import neofoam.turbulence.models  # noqa: F401

from turbulence.conftest import requires_openfoam


@requires_openfoam
def test_select_from_case_falls_back_for_unported_model(ras_case: Path) -> None:
    # kEpsilon has no native NeoFOAM model in the skeleton → OpenFOAM fallback.
    selected = select_from_case(ras_case, of_factory=MagicMock())
    assert isinstance(selected, OpenFOAMTurbulenceModel)


@requires_openfoam
def test_select_from_case_dispatches_native_laminar(laminar_case: Path) -> None:
    # laminar IS a registered native → dispatched, not falling back.
    selected = select_from_case(laminar_case)
    assert isinstance(selected, ModelSpec)
    assert selected.name == "laminar"


@requires_openfoam
def test_select_from_case_dispatches_registered_ras_model(
    ras_case: Path, clean_turbulence_registry: None
) -> None:
    # Registering a native for the configured RASModel makes selection dispatch
    # to it instead of falling back to OpenFOAM.
    k_epsilon = Model("kEpsilon").register_with(turbulenceModel)
    selected = select_from_case(ras_case)
    assert selected is k_epsilon
