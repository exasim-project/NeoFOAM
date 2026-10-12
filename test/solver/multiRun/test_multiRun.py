# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""multiRun steps two uncoupled regions in one case and one time loop.

Every region-capable solver of the package needs NeoN or FEniCSx, so the regions
run ``slab`` (``slab_solver``), a test-only numpy solver; multiRun finds region
solvers by package name, so the test makes ``resolve_solver`` of the multiRun
module return it. ``cases/two_regions`` holds the single-region cases
``slab_left`` and ``slab_right`` as its regions ``left`` and ``right``, under one
controlDict with the same time steps. The regions differ in mesh, diffusivity and
temperatures, so a region that read the other's files, missed a step or ran on
the driver's context would not reproduce its single-region run. Both runs do the
same floating-point operations in the same order, hence the round-off tolerance.
One multiRun run serves every test, hence the module-scoped fixture.
"""

import importlib
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from neofoam.framework.context import Context
from neofoam.framework.solver.registry import resolve_solver
from neofoam.solver.multiRun import run
from solver.multiRun.slab_solver import run_alone, slab

CASES = Path(__file__).parent / "cases"
# The package exports the multiRun solver under the module's name.
MULTI_RUN_MODULE = importlib.import_module("neofoam.solver.multiRun.multiRun")


def _with_slab(name: str) -> Any:
    return slab if name == "slab" else resolve_solver(name)


@pytest.fixture(scope="module")
def ctx(tmp_path_factory: pytest.TempPathFactory) -> Context:
    run_dir = tmp_path_factory.mktemp("multiRun") / "two_regions"
    shutil.copytree(CASES / "two_regions", run_dir)
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(MULTI_RUN_MODULE, "resolve_solver", _with_slab)
        monkeypatch.chdir(run_dir)
        return run()


@pytest.mark.parametrize(("region", "case"), [("left", "slab_left"), ("right", "slab_right")])
def test_multiRun_region_matches_its_single_region_run(
    ctx: Context, region: str, case: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shutil.copytree(CASES / case, tmp_path / case)
    monkeypatch.chdir(tmp_path / case)
    alone = run_alone()

    np.testing.assert_allclose(
        ctx.models["regions"][region][1].fields["T"],
        alone.fields["T"],
        rtol=1e-14,
        err_msg=f"region {region} vs case {case}",
    )


def test_multiRun_regions_share_the_time(ctx: Context) -> None:
    assert ctx.time.index == 10
    for _runtime, region_ctx in ctx.models["regions"].values():
        assert region_ctx.time is ctx.time
