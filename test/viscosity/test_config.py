# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Tests for reading ``constant/transportProperties`` via BaseConfig.

These read real OpenFOAM dictionaries from the bundled cases under
``test/viscosity/cases`` (no dict content is encoded here). They touch the
OpenFOAM reading strategy (pybFoam), so they carry the ``requires_openfoam``
marker.
"""

from pathlib import Path

from neofoam.viscosity.config import TransportPropertiesConfig
from neofoam.viscosity.selection import model_name

from viscosity.conftest import requires_openfoam


@requires_openfoam
def test_reads_newtonian_transport_model(newtonian_case: Path) -> None:
    cfg = TransportPropertiesConfig.load(case_dir=newtonian_case)
    assert cfg.transportModel == "Newtonian"
    assert cfg.nu == 1e-05


@requires_openfoam
def test_reads_non_newtonian_transport_model(cross_power_law_case: Path) -> None:
    cfg = TransportPropertiesConfig.load(case_dir=cross_power_law_case)
    assert cfg.transportModel == "CrossPowerLaw"
    assert cfg.nu is None  # coefficients live in their own sub-dictionary


@requires_openfoam
def test_model_name_from_loaded_config(
    newtonian_case: Path, cross_power_law_case: Path
) -> None:
    newtonian_cfg = TransportPropertiesConfig.load(case_dir=newtonian_case)
    assert model_name(newtonian_cfg) == "Newtonian"

    cross_cfg = TransportPropertiesConfig.load(case_dir=cross_power_law_case)
    assert model_name(cross_cfg) == "CrossPowerLaw"
