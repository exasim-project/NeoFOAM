# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Tests for reading ``constant/turbulenceProperties`` via BaseConfig.

These read real OpenFOAM dictionaries from the bundled cases under
``test/turbulence/cases`` (no dict content is encoded here). They touch the
OpenFOAM reading strategy (pybFoam), so they carry the ``requires_openfoam``
marker.
"""

from pathlib import Path

from neofoam.turbulence.config import TurbulencePropertiesConfig
from neofoam.turbulence.selection import model_name

from turbulence.conftest import requires_openfoam


@requires_openfoam
def test_reads_ras_simulation_type(ras_case: Path) -> None:
    cfg = TurbulencePropertiesConfig.load(case_dir=ras_case)
    assert cfg.simulationType == "RAS"


@requires_openfoam
def test_reads_nested_ras_subdict(ras_case: Path) -> None:
    cfg = TurbulencePropertiesConfig.load(case_dir=ras_case)
    assert cfg.RAS is not None
    assert cfg.RAS.RASModel == "kEpsilon"
    assert cfg.RAS.turbulence is True
    assert cfg.RAS.printCoeffs is True


@requires_openfoam
def test_reads_nested_les_subdict(les_case: Path) -> None:
    cfg = TurbulencePropertiesConfig.load(case_dir=les_case)
    assert cfg.simulationType == "LES"
    assert cfg.LES is not None
    assert cfg.LES.LESModel == "Smagorinsky"
    assert cfg.LES.delta == "cubeRootVol"


@requires_openfoam
def test_laminar_simulation_type_has_no_subdicts(laminar_case: Path) -> None:
    cfg = TurbulencePropertiesConfig.load(case_dir=laminar_case)
    assert cfg.simulationType == "laminar"
    assert cfg.RAS is None
    assert cfg.LES is None


@requires_openfoam
def test_model_name_from_loaded_config(ras_case: Path, laminar_case: Path) -> None:
    ras_cfg = TurbulencePropertiesConfig.load(case_dir=ras_case)
    assert model_name(ras_cfg) == "kEpsilon"

    laminar_cfg = TurbulencePropertiesConfig.load(case_dir=laminar_case)
    assert model_name(laminar_cfg) == "laminar"
