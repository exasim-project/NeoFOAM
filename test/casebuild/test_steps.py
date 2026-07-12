# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Unit tests for the built-in steps: ``patch``, ``configs`` (meshing → ``test_meshing``).

Each step is exercised against a materialized :class:`CaseDir` staged from the
committed ``cavity`` template. Configs are built the way production does, from
``configurations(incompressibleFluid)``.
"""

import shutil
from pathlib import Path

import pybFoam as pyf
import pytest

from neofoam.casebuild import CaseDir, configs, patch
from neofoam.framework.solver.configurations import configurations
from neofoam.solver.incompressibleFluid.incompressibleFluid import incompressibleFluid

CAVITY = Path(__file__).parent / "cases" / "cavity"


def _staged(tmp_path: Path) -> CaseDir:
    dst = tmp_path / "case"
    shutil.copytree(CAVITY, dst)
    return CaseDir(dst)


def test_patch_merges_dict_and_kwargs_with_kwargs_winning(tmp_path: Path) -> None:
    case = _staged(tmp_path)
    patch("system/controlDict", {"endTime": 1.0, "deltaT": 0.5}, endTime=2.0)(case)
    d = pyf.dictionary.read(str(case.path / "system" / "controlDict"))
    assert d.get_scalar("endTime") == pytest.approx(2.0)  # kwargs override the dict
    assert d.get_scalar("deltaT") == pytest.approx(0.5)


def test_patch_missing_file_raises(tmp_path: Path) -> None:
    case = _staged(tmp_path)
    with pytest.raises(FileNotFoundError):
        patch("system/doesNotExist", endTime=1.0)(case)


def test_configs_writes_config_file(tmp_path: Path) -> None:
    case = _staged(tmp_path)
    cfgs = configurations(incompressibleFluid)
    transport = cfgs["TransportPropertiesConfig"].model_validate(
        {"transportModel": "Newtonian", "nu": 0.01}
    )
    configs(transport)(case)
    assert (case.path / "constant" / "transportProperties").is_file()
