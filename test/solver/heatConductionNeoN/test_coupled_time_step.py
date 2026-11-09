# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The coupled blocks repeat a time step's passes, not its start (``first``).

``cases/coupledBlocks`` (see ``test_coupled_blocks.py``) without ``couplings``,
parallel: each region's coupling loop wraps its whole time step,
``set_time_step``, the time-step op, ``increment_time`` and ``start_step``
included; they are marked ``first`` and run in the loop's first pass only. So
with ``maxIterations: 3`` the time still advances once per time step, and on the
converged state one pass or three give the same temperature. A serial loop may
not wrap the time step: the case raises before it runs.
"""

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from solver.heatConductionNeoN._run_case import cell_centres, mesh_copy, result, run_worker
from solver.heatConductionNeoN.test_coupled_blocks import T_INTERFACE, _analytic

END_TIME, DELTA_T = 2.0e6, 1.0e6


def _replaced(text: str, old: str, new: str) -> str:
    assert text.count(old) == 1, old
    return text.replace(old, new)


def _case(tmp_path: Path, scheme: str, max_iterations: int, converged: bool = False) -> Path:
    """``coupledBlocks`` without ``couplings`` (one node around the whole time step), with
    ``scheme`` and ``max_iterations``.

    ``converged``: starting from the solution, the interface at its temperature.
    """
    case_dir = mesh_copy("coupledBlocks", tmp_path / "case")
    coupling_file = case_dir / "system" / "regionCoupling.yaml"
    coupling = yaml.safe_load(coupling_file.read_text())
    (node,) = coupling.pop("couplings")
    (exchange,) = node["exchanges"]
    coupling.update(
        scheme=scheme,
        maxIterations=max_iterations,
        acceleration=exchange["acceleration"],
        convergence=node["convergence"],
    )
    coupling_file.write_text(yaml.safe_dump(coupling))
    if converged:
        for region in ("left", "right"):
            x = cell_centres(case_dir / region)[:, 0]
            values = "\n".join(f"{value:.15g}" for value in _analytic(region, x))
            field = case_dir / region / "0" / "T"
            text = _replaced(
                field.read_text(),
                "internalField   uniform 350;",
                f"internalField   nonuniform List<scalar> {len(x)}\n(\n{values}\n);",
            )
            # the interface starts at the interface temperature
            text = _replaced(
                text, "value           uniform 350;", f"value           uniform {T_INTERFACE};"
            )
            field.write_text(text)
    return case_dir


def _run(tmp_path: Path, scheme: str, max_iterations: int, converged: bool = False) -> Any:
    return result(_case(tmp_path, scheme, max_iterations, converged), "multi")


def test_the_time_advances_once_per_time_step_however_many_passes(tmp_path: Path) -> None:
    computed = _run(tmp_path, "parallel", max_iterations=3)

    # right leads: three passes in each of the two time steps
    assert computed["passes"] == {"right": [3, 3]}
    for region in ("left", "right"):
        time = computed["regions"][region]["time"]
        assert (time["value"], time["index"], time["deltaT"]) == (END_TIME, 2, DELTA_T)


def test_more_passes_on_the_converged_state_change_nothing(tmp_path: Path) -> None:
    one = _run(tmp_path / "one", "parallel", max_iterations=1, converged=True)
    three = _run(tmp_path / "three", "parallel", max_iterations=3, converged=True)

    for region in ("left", "right"):
        np.testing.assert_allclose(
            three["regions"][region]["T"], one["regions"][region]["T"], rtol=0, atol=1e-9
        )
        assert three["regions"][region]["time"] == one["regions"][region]["time"]


def test_a_serial_loop_around_the_whole_time_step_raises(tmp_path: Path) -> None:
    done = run_worker(_case(tmp_path, "serial", max_iterations=3), "multi")

    assert done.returncode != 0
    assert (
        "coupling 'coupling' is serial but holds the time step of its regions; make it "
        "parallel or place it around named operations (wraps."
    ) in done.stderr + done.stdout
