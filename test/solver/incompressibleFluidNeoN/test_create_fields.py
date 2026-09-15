# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The NeoN runtime init step builds the executor the configuration asks for.

``create_fields`` used to call ``create_adapter_run_time`` with no executor at
all, so every run silently took the binding's ``Serial`` default and there was no
way to ask for another one. The executor now comes from the ``executor`` entry of
``system/controlDict`` (:func:`neofoam.solver.neon_runtime.requested_executor`),
the same entry the C++ solvers read.

The name reaches NeoFOAM's C++ ``createExecutor``, which logs ``Creating Executor
<name>`` before resolving it. A name no backend answers to is the discriminating
case: it reaches that log only if the configured string was passed through, and
the failed run proves it was really resolved. The resolving names are pinned
cheaply by ``test/solver/test_neon_runtime.py`` and by every other case here. A
subprocess is needed: NeoN/Kokkos and OpenFOAM globals do not survive a re-run.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from neofoam.tooling.casebuild import block_mesh, patch
from solver.incompressibleFluidNeoN._regex_case import regex_solver_keys_case

# One time step (deltaT 0.005) — the executor is built during initialization, so
# the run only has to get past the first step.
END_TIME = 0.005


def _run_solver(case_path: Path) -> subprocess.CompletedProcess[str]:
    """Run the framework NeoN solver in ``case_path``."""
    env = {**os.environ, "FOAM_SIGFPE": "false"}
    return subprocess.run(
        [
            sys.executable,
            "-c",
            "from neofoam.solver.incompressibleFluidNeoN import run;"
            " run(['incompressibleFluidNeoN'])",
        ],
        cwd=str(case_path),
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )


@pytest.mark.slow
def test_neon_runtime_uses_the_requested_executor(tmp_path: Path) -> None:
    """The configured name reaches createExecutor, and an unknown one fails the run."""
    case = (
        regex_solver_keys_case()
        | patch("system/controlDict", endTime=END_TIME, executor="NoSuchExecutor")
        | block_mesh()
    ).build_at(tmp_path / "executorSelection")

    result = _run_solver(case.path)

    output = result.stdout + result.stderr
    assert "Creating Executor NoSuchExecutor" in output, (
        f"the configured executor did not reach createExecutor:\n{output[-3000:]}"
    )
    assert result.returncode != 0, (
        f"an unresolvable executor did not fail the run:\n{output[-3000:]}"
    )
