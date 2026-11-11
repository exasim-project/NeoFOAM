# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""A block of ``cases/coupledBlocks`` run alone: its coupled patch has no partner, so it raises."""

from pathlib import Path

import pytest

from solver.heatConductionNeoN._run_case import mesh_copy, run_worker


# left's interface is coupledHeatFlux, right's coupledTemperature
@pytest.mark.parametrize("region", ["left", "right"])
def test_uncoupled_block_with_a_coupled_patch_raises(tmp_path: Path, region: str) -> None:
    case_dir = mesh_copy("coupledBlocks", tmp_path / "blocks") / region

    done = run_worker(case_dir, "single")

    assert done.returncode != 0
    assert "ValueError" in done.stderr, done.stderr[-3000:]
    assert "the case does not couple its region" in done.stderr, done.stderr[-3000:]
