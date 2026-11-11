# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""multiRun runs two uncoupled regions, each as its own single-region run would.

``cases/two_regions`` holds ``cases/slab_left`` and ``cases/slab_right`` as its
regions under one controlDict; they differ in mesh, diffusivity and
temperatures, so a region reading the other's files or missing a step would
differ. Both runs do the same floating-point operations in the same order,
hence round-off as the tolerance.
"""

from pathlib import Path

import numpy as np
import pytest

from neofoam.framework.context import Context
from neofoam.framework.coupling import Coupling
from neofoam.solver.multiRun import run
from solver.multiRun.case_files import copy_case
from solver.multiRun.slab_solver import slab


@pytest.fixture(scope="module")
def regions(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Context]:
    run_dir = copy_case("two_regions", tmp_path_factory.mktemp("multiRun") / "case")
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.chdir(run_dir)
        return run()


@pytest.mark.parametrize(("region", "case"), [("left", "slab_left"), ("right", "slab_right")])
def test_uncoupled_region_matches_its_single_region_run(
    regions: dict[str, Context],
    region: str,
    case: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(copy_case(case, tmp_path / case))

    alone = slab.instantiate().run()

    np.testing.assert_allclose(
        regions[region].fields["T"],
        alone.fields["T"],
        rtol=1e-14,
        err_msg=f"region {region} vs case {case}",
    )


def test_uncoupled_regions_step_the_same_times(regions: dict[str, Context]) -> None:
    assert [(ctx.region, ctx.time.index) for ctx in regions.values()] == [
        ("left", 10),
        ("right", 10),
    ]


def test_uncoupled_regions_run_without_coupling_ops(regions: dict[str, Context]) -> None:
    for ctx in regions.values():
        assert len(Coupling.of(ctx).operations()) == 0
