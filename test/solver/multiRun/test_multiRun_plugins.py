# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""A ``system/regionCoupling.yaml`` names the mapping and acceleration of a coupling plugin.

The plugin is a fake entry point of ``neofoam.coupling``
(``framework.coupling.plugins``) whose module registers the mapping ``fake-rbf``
(the identity: both patches of ``cases/coupled_slabs`` lie at x = 1) and the
acceleration ``fake-iqn`` (a constant relaxation), each with an option of its
own and counting its calls. The case's file names them with their options; it
validates with the plugin only, reading it loads the plugins (outside multiRun
too), and multiRun, which loads them before the regions start, reaches the
steady profile of ``test_multiRun_region_coupling`` through them.
"""

import importlib
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from framework.coupling.plugins import installed
from pydantic import ValidationError

from neofoam.framework.context import Context
from neofoam.framework.coupling import load_plugins
from neofoam.framework.coupling.kernels import ENTRY_POINT_GROUP
from neofoam.framework.solver.registry import resolve_solver
from neofoam.solver.multiRun import run
from neofoam.solver.multiRun.configs import RegionCouplingConfig
from solver.multiRun.slab_solver import slab

CASE = Path(__file__).parent / "cases" / "coupled_slabs"
# The package exports the multiRun solver under the module's name.
MULTI_RUN_MODULE = importlib.import_module("neofoam.solver.multiRun.multiRun")
EXPECTED = {
    "left": [390.625, 371.875, 353.125, 334.375],
    "right": [321.875, 315.625, 309.375, 303.125],
}
REGION_COUPLING = """\
exchanges:
  - name: interface
    sides: {left: right, right: left}
    mapping: {type: fake-rbf, radius: 0.25}
acceleration: {type: fake-iqn, relaxation: 0.7, columns: 3}
convergence:
  - {type: relative, data: T, limit: 1.0e-12}
"""
PRECICE = """
from typing import Any, Literal

from pydantic import Field

from neofoam.framework.coupling import ConstantRelaxation, CouplingAcceleration, CouplingMapping

calls = {"fake-rbf": 0, "fake-iqn": 0}


@CouplingMapping.register
class RadialBasis(CouplingMapping):
    type: Literal["fake-rbf"] = "fake-rbf"
    radius: float = Field(gt=0)

    def bind(self, source: Any, target: Any) -> Any:
        def mapper(data: Any) -> Any:
            calls["fake-rbf"] += 1
            return data

        return mapper


@CouplingAcceleration.register
class QuasiNewton(ConstantRelaxation):
    type: Literal["fake-iqn"] = "fake-iqn"
    columns: int = Field(gt=0)

    def __call__(self, sent: Any, iteration: int) -> Any:
        calls["fake-iqn"] += 1
        return super().__call__(sent, iteration)
"""


def _with_slab(name: str) -> Any:
    return slab if name == "slab" else resolve_solver(name)


def _case(tmp_path: Path, text: str = REGION_COUPLING) -> Path:
    shutil.copytree(CASE, tmp_path / "case")
    (tmp_path / "case" / "system" / "regionCoupling.yaml").write_text(text)
    return tmp_path / "case"


def test_multiRun_couples_through_the_strategies_of_a_plugin_with_their_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _case(tmp_path)
    monkeypatch.setattr(MULTI_RUN_MODULE, "resolve_solver", _with_slab)
    monkeypatch.chdir(case)

    with installed(monkeypatch, tmp_path, precice=PRECICE):
        regions: dict[str, Context] = run()
        config = RegionCouplingConfig.load(case_dir=case)
        calls = importlib.import_module("fake_coupling_plugin_precice").calls

    mapping, acceleration = config.exchanges[0].mapping, config.acceleration
    assert (mapping.type, mapping.radius) == ("fake-rbf", 0.25)  # type: ignore[attr-defined]
    assert acceleration is not None
    assert (acceleration.type, acceleration.columns) == ("fake-iqn", 3)  # type: ignore[attr-defined]
    assert calls["fake-rbf"] > 0
    assert calls["fake-iqn"] > 0
    for region, expected in EXPECTED.items():
        np.testing.assert_allclose(regions[region].fields["T"], expected, rtol=0, atol=1e-8)


def test_reading_region_coupling_outside_multiRun_loads_the_plugins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tool or the UI reads the file without ``run()``: the plugin's strategies are selected."""
    case = _case(tmp_path)

    with installed(monkeypatch, tmp_path, precice=PRECICE) as asked:
        config = RegionCouplingConfig.load(case_dir=case)
        RegionCouplingConfig.load(case_dir=case)

    mapping: Any = config.exchanges[0].mapping
    acceleration: Any = config.acceleration
    assert (type(mapping).__name__, mapping.radius) == ("RadialBasis", 0.25)
    assert (type(acceleration).__name__, acceleration.columns) == ("QuasiNewton", 3)
    assert asked == [ENTRY_POINT_GROUP]  # loaded once


def test_region_coupling_naming_a_strategy_of_no_installed_plugin_raises(tmp_path: Path) -> None:
    with pytest.raises(
        ValidationError,
        match=r"exchanges.0.mapping\n.*unknown mapping type 'fake-rbf'; registered mapping "
        r"types: \['identity', 'nearest-neighbor'\]",
    ):
        RegionCouplingConfig.load(case_dir=_case(tmp_path))


def test_a_wrong_option_of_a_plugin_strategy_raises_naming_its_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _case(tmp_path, REGION_COUPLING.replace("columns: 3", "column: 3"))

    with installed(monkeypatch, tmp_path, precice=PRECICE):
        load_plugins()
        with pytest.raises(
            ValidationError,
            match=r"acceleration\n.*acceleration 'fake-iqn': missing option 'columns'; unknown "
            r"option 'column'; its options: \['relaxation', 'columns'\]",
        ):
            RegionCouplingConfig.load(case_dir=case)
