# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""``system/regionCoupling.yaml`` names the mapping and acceleration of a coupling plugin.

The fixture imports ``fake_plugin.py``, which registers on import, and unregisters it
after the test, so no other test sees its types. The mapping is the identity (both
patches lie at x = 1), so the run reaches the profile of ``test_coupled_slabs`` within
its 1e-8 K.
"""

import importlib
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from neofoam.core.plugin_system import PluginSystem
from neofoam.framework.coupling import RegionCouplingConfig
from neofoam.solver.multiRun import run
from solver.multiRun.case_files import copy_case

PLUGIN = "solver.multiRun.fake_plugin"
EXPECTED = {
    "left": [390.625, 371.875, 353.125, 334.375],
    "right": [321.875, 315.625, 309.375, 303.125],
}


@pytest.fixture
def plugin() -> Iterator[ModuleType]:
    module = importlib.import_module(PLUGIN)
    yield module
    PluginSystem.remove_plugin_model("CouplingMapping", module.RadialBasis)
    PluginSystem.remove_plugin_model("CouplingAcceleration", module.QuasiNewton)
    del sys.modules[PLUGIN]


def _case(tmp_path: Path, overlay: str) -> Path:
    return copy_case("coupled_slabs", tmp_path / "case", f"plugin/{overlay}")


def test_multiRun_couples_through_the_strategies_of_a_plugin(
    plugin: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(_case(tmp_path, "strategies"))

    regions = run()

    assert plugin.calls["fake-rbf"] > 0
    assert plugin.calls["fake-iqn"] > 0
    for region, expected in EXPECTED.items():
        np.testing.assert_allclose(
            regions[region].fields["T"], expected, rtol=0, atol=1e-8, err_msg=f"region {region}"
        )


def test_region_coupling_reads_the_options_of_a_plugin_strategy(
    plugin: ModuleType, tmp_path: Path
) -> None:
    config = RegionCouplingConfig.load(case_dir=_case(tmp_path, "strategies"))

    mapping: Any = config.exchanges[0].mapping
    acceleration: Any = config.couplings[0].exchanges[0].acceleration
    assert (type(mapping), mapping.radius) == (plugin.RadialBasis, 0.25)
    assert (type(acceleration), acceleration.columns) == (plugin.QuasiNewton, 3)


def test_region_coupling_naming_an_unregistered_strategy_raises(tmp_path: Path) -> None:
    with pytest.raises(
        ValidationError,
        match=r"exchanges.0.mapping.mapping\n.*Input tag 'fake-rbf' .* expected tags: "
        r"'identity', 'nearest-neighbor'",
    ):
        RegionCouplingConfig.load(case_dir=_case(tmp_path, "strategies"))


def test_a_wrong_option_of_a_plugin_strategy_raises_naming_it(
    plugin: ModuleType, tmp_path: Path
) -> None:
    with pytest.raises(
        ValidationError,
        match=r"acceleration.fake-iqn.columns\n.*Field required.*\n.*\n.*acceleration.fake-iqn."
        r"column\n.*Extra inputs",
    ):
        RegionCouplingConfig.load(case_dir=_case(tmp_path, "option_typo"))
