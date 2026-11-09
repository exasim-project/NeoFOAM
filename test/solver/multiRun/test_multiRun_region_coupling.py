# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""multiRun couples two slab regions as ``system/regionCoupling.yaml`` describes.

``cases/coupled_slabs`` is the case of ``test_multiRun_coupling``: slab ``left``
(x in [0, 1], conductivity 1, 400 K at x = 0) and slab ``right`` (x in [1, 2],
conductivity 3, 300 K at x = 2), coupled at x = 1. Its regionCoupling.yaml
couples left's patch ``right`` with right's patch ``left``, parallel with Aitken
relaxation, until the exchanged data of ``T`` changes by < 1e-12 relative. Each
scheme/acceleration variant rewrites that file with its config class, as one
coupling node around each region's ``solve_T`` (a serial node must leave the
time step outside).

Every variant must reach the same analytic steady profile (linear in each slab,
325 K at the interface; see ``test_multiRun_coupling``). The parallel scheme
contracts by ~0.73 per pass, so a relative change of 1e-12 (3e-10 K) leaves at
most ~1e-9 K; 1e-8 K bounds every variant.
"""

import importlib
import shutil
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pytest

from neofoam.framework.context import Context
from neofoam.framework.coupling import Acceleration, AitkenRelaxation, ConstantRelaxation
from neofoam.framework.solver.registry import resolve_solver
from neofoam.solver.multiRun import run
from neofoam.solver.multiRun.configs import (
    CouplingNodeConfig,
    ExchangeConfig,
    NodeExchangeConfig,
    PlacementConfig,
    RegionCouplingConfig,
)
from solver.multiRun.slab_solver import slab

CASE = Path(__file__).parent / "cases" / "coupled_slabs"
# The package exports the multiRun solver under the module's name.
MULTI_RUN_MODULE = importlib.import_module("neofoam.solver.multiRun.multiRun")
Scheme = Literal["serial", "parallel"]
EXPECTED = {
    "left": [390.625, 371.875, 353.125, 334.375],
    "right": [321.875, 315.625, 309.375, 303.125],
}


def _with_slab(name: str) -> Any:
    return slab if name == "slab" else resolve_solver(name)


def _case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **update: Any) -> None:
    """The case in ``tmp_path`` as the working directory, its coupling config updated."""
    shutil.copytree(CASE, tmp_path / "case")
    config = RegionCouplingConfig.load(case_dir=tmp_path / "case")
    config.model_copy(update=update).save(case_dir=tmp_path / "case")
    monkeypatch.setattr(MULTI_RUN_MODULE, "resolve_solver", _with_slab)
    monkeypatch.chdir(tmp_path / "case")


def _around_solve_T(scheme: Scheme, acceleration: Acceleration) -> dict[str, Any]:
    """The case's coupling as one node of ``scheme`` that wraps each region's ``solve_T``."""
    config = RegionCouplingConfig.load(case_dir=CASE)
    node = CouplingNodeConfig(
        name="coupling",
        scheme=scheme,
        maxIterations=config.maxIterations or 100,
        exchanges=[NodeExchangeConfig(name="interface", acceleration=acceleration)],
        convergence=config.convergence or [],
        wraps={region: PlacementConfig(operations=["solve_T"]) for region in EXPECTED},
    )
    return {"acceleration": None, "convergence": None, "maxIterations": None, "couplings": [node]}


@pytest.mark.parametrize(
    ("scheme", "acceleration"),
    [
        ("serial", AitkenRelaxation(relaxation=0.5)),
        ("serial", ConstantRelaxation(relaxation=1.0)),
        ("parallel", AitkenRelaxation(relaxation=0.5)),
        ("parallel", ConstantRelaxation(relaxation=0.7)),
    ],
    ids=["serial-aitken", "serial-constant", "parallel-aitken", "parallel-constant"],
)
def test_multiRun_region_coupling_reaches_the_steady_profile(
    scheme: Scheme,
    acceleration: Acceleration,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _case(tmp_path, monkeypatch, **_around_solve_T(scheme, acceleration))

    regions: dict[str, Context] = run()

    for region, expected in EXPECTED.items():
        np.testing.assert_allclose(
            regions[region].fields["T"],
            expected,
            rtol=0,
            atol=1e-8,
            err_msg=f"coupled_slabs {scheme}/{acceleration.type} region {region}",
        )


def test_multiRun_region_coupling_of_an_unknown_region_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _case(
        tmp_path,
        monkeypatch,
        exchanges=[ExchangeConfig(sides={"left": "right", "fluid": "interface"})],
    )

    with pytest.raises(ValueError, match=r"couples the regions \['fluid'\]"):
        run()


def test_multiRun_region_coupling_convergence_on_data_no_side_sends_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = RegionCouplingConfig.load(case_dir=CASE)
    convergence = [*config.convergence, config.convergence[0].model_copy(update={"data": "U"})]
    _case(tmp_path, monkeypatch, convergence=convergence)

    with pytest.raises(
        ValueError, match=r"convergence RelativeConvergence on U, which no side of the node sends"
    ):
        run()


def test_multiRun_region_coupling_of_a_region_without_a_coupled_patch_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """right's patch left is fixed, not coupledConduction: right has no coupled side."""
    _case(tmp_path, monkeypatch)
    temperature = Path("0/right/T.yaml")
    temperature.write_text(
        temperature.read_text().replace(
            "left:\n  type: coupledConduction\n  value: 300.0", "left: 300.0"
        )
    )

    with pytest.raises(
        ValueError, match="exchange 'interface' connects right.left, which has no coupled side"
    ):
        run()
