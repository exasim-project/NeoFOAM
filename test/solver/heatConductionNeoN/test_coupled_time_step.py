# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The coupled blocks with one parallel node around each region's whole time step.

``cases/coupledBlocks`` with an overlay ``regionCoupling.yaml`` without
``wraps``: the time-step ops and ``start_step`` are marked ``first`` and
``write_output`` ``final``, so however many passes a time step takes, the time
advances and the fields are written once. ``convergedStart`` starts from the
steady profile, where more passes change nothing beyond the linear solver's
round-off (1e-9 K). A serial node may not hold the time step.
"""

from pathlib import Path

import numpy as np

from solver.heatConductionNeoN._run_case import mesh_copy, result, run_worker
from solver.heatConductionNeoN.test_coupled_blocks import ATOL_K, STEADY_T

END_TIME, DELTA_T = 2.0e6, 1.0e6


def test_coupled_time_step_advances_once_however_many_passes(tmp_path: Path) -> None:
    case_dir = mesh_copy("coupledBlocks", tmp_path / "case", "wholeStep")

    computed = result(case_dir, "multi")

    # right leads: maxIterations 3 passes in each of the two time steps
    assert computed["passes"] == {"right": [3, 3]}
    for region in ("left", "right"):
        time = computed["regions"][region]["time"]
        assert (time["value"], time["index"], time["deltaT"]) == (END_TIME, 2, DELTA_T), region


def test_coupled_time_step_writes_once_however_many_passes(tmp_path: Path) -> None:
    case_dir = mesh_copy("coupledBlocks", tmp_path / "case", "wholeStep")

    done = run_worker(case_dir, "multi")

    assert done.returncode == 0, done.stderr[-3000:]
    # write_output reports the execution time each time it runs: per region and step
    assert done.stdout.count("ExecutionTime =") == 2 * 2
    for region in ("left", "right"):
        written = sorted(float(path.name) for path in (case_dir / region).glob("[1-9]*"))
        assert written == [DELTA_T, END_TIME], region
        assert (case_dir / region / f"{END_TIME:g}" / "T").is_file(), region


def test_coupled_time_step_more_passes_on_the_converged_state_change_nothing(
    tmp_path: Path,
) -> None:
    one = result(
        mesh_copy("coupledBlocks", tmp_path / "one", "wholeStepOnePass", "convergedStart"), "multi"
    )
    three = result(
        mesh_copy("coupledBlocks", tmp_path / "three", "wholeStep", "convergedStart"), "multi"
    )

    for region in ("left", "right"):
        one_T, three_T = one["regions"][region]["T"], three["regions"][region]["T"]
        np.testing.assert_allclose(
            one_T, STEADY_T[region], rtol=0, atol=ATOL_K, err_msg=f"{region}: one pass"
        )
        np.testing.assert_allclose(
            three_T, one_T, rtol=0, atol=1e-9, err_msg=f"{region}: three passes against one"
        )
        assert three["regions"][region]["time"] == one["regions"][region]["time"], region


def test_coupled_time_step_held_by_a_serial_node_raises(tmp_path: Path) -> None:
    case_dir = mesh_copy("coupledBlocks", tmp_path / "case", "wholeStepSerial")

    done = run_worker(case_dir, "multi")

    assert done.returncode != 0
    assert (
        "coupling 'coupling' is serial but holds the time step of its regions; make it "
        "parallel or place it around named operations (wraps."
    ) in done.stderr + done.stdout
