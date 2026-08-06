# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The NeoN runtime init step builds the executor the configuration asks for.

``create_fields`` used to call ``create_adapter_run_time`` with no executor at
all, so every run silently took the binding's ``Serial`` default and there was no
way to ask for another one. The executor now comes from ``NEOFOAM_EXECUTOR``
(:func:`neofoam.solver.neon_runtime.requested_executor`).

The name reaches NeoFOAM's C++ ``createExecutor``, which logs ``Creating Executor
<name>`` before it resolves it — so the log line is the observation, and the
run's exit status says whether the name resolved. A name no backend answers to
is the discriminating case: it can only appear in that log if the configured
string was passed through rather than dropped, and the failed run proves the
name was really resolved and not merely printed. The resolving names (unset ->
``Serial``, and ``Serial``) are covered where they are cheap:
``test/solver/test_neon_runtime.py`` pins the environment read in-process, and
every other case in this directory runs on the default executor.

The selection happens inside the C++ runtime during a real init, and NeoN/Kokkos
plus OpenFOAM keep per-process global state that does not survive a second
in-process run, so the case runs in its own subprocess.
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


def _run_solver(case_path: Path, executor: str) -> subprocess.CompletedProcess[str]:
    """Run the framework NeoN solver in ``case_path`` with ``NEOFOAM_EXECUTOR`` set."""
    env = {**os.environ, "FOAM_SIGFPE": "false", "NEOFOAM_EXECUTOR": executor}
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
        regex_solver_keys_case() | patch("system/controlDict", endTime=END_TIME) | block_mesh()
    ).build_at(tmp_path / "executorSelection")

    result = _run_solver(case.path, "NoSuchExecutor")

    output = result.stdout + result.stderr
    assert "Creating Executor NoSuchExecutor" in output, (
        f"the configured executor did not reach createExecutor:\n{output[-3000:]}"
    )
    assert result.returncode != 0, (
        f"an unresolvable executor did not fail the run:\n{output[-3000:]}"
    )
