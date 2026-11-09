# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""A ``system/regionCoupling.yaml`` names the mapping and acceleration of a coupling plugin.

The plugin is a fake entry point of ``neofoam.coupling``
(``framework.coupling.plugins``) that registers the mapping ``fake-rbf`` (the
identity: both patches of ``cases/coupled_slabs`` lie at x = 1) and the
acceleration ``fake-iqn`` (a constant relaxation), each counting its calls. The
case's file names them; it validates with the plugin only, and multiRun reaches
the steady profile of ``test_multiRun_region_coupling`` through them.
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
from neofoam.framework.coupling import BoundMapping, ConstantRelaxation, Mapping
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
    mapping: fake-rbf
acceleration: {type: fake-iqn, relaxation: 0.7}
convergence:
  - {type: relative, data: T, limit: 1.0e-12}
"""
calls = {"fake-rbf": 0, "fake-iqn": 0}


class _RadialBasis(Mapping):
    kind = "fake-rbf"

    def bind(self, source: Any, target: Any) -> BoundMapping:
        def mapper(data: Any) -> Any:
            calls["fake-rbf"] += 1
            return data

        return mapper


class _QuasiNewton(ConstantRelaxation):
    def __call__(self, sent: Any, iteration: int) -> Any:
        calls["fake-iqn"] += 1
        return super().__call__(sent, iteration)


def _register(mappings: dict[str, Any], accelerations: dict[str, Any], _: dict[str, Any]) -> None:
    mappings[_RadialBasis.kind] = _RadialBasis
    accelerations["fake-iqn"] = _QuasiNewton


def _with_slab(name: str) -> Any:
    return slab if name == "slab" else resolve_solver(name)


def _case(tmp_path: Path) -> Path:
    shutil.copytree(CASE, tmp_path / "case")
    (tmp_path / "case" / "system" / "regionCoupling.yaml").write_text(REGION_COUPLING)
    return tmp_path / "case"


def test_multiRun_couples_through_the_kernels_of_a_plugin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _case(tmp_path)
    monkeypatch.setattr(MULTI_RUN_MODULE, "resolve_solver", _with_slab)
    monkeypatch.chdir(case)
    monkeypatch.setitem(calls, "fake-rbf", 0)
    monkeypatch.setitem(calls, "fake-iqn", 0)

    with installed(monkeypatch, precice=_register):
        config = RegionCouplingConfig.load(case_dir=case)
        regions: dict[str, Context] = run()

    assert (config.exchanges[0].mapping, config.acceleration.type) == ("fake-rbf", "fake-iqn")
    assert calls["fake-rbf"] > 0
    assert calls["fake-iqn"] > 0
    for region, expected in EXPECTED.items():
        np.testing.assert_allclose(regions[region].fields["T"], expected, rtol=0, atol=1e-8)


def test_region_coupling_naming_a_kernel_of_no_installed_plugin_raises(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="unknown mapping 'fake-rbf'"):
        RegionCouplingConfig.load(case_dir=_case(tmp_path))
