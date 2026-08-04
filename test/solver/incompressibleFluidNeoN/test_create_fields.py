# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The NeoN runtime init step builds the executor the configuration asks for.

``create_fields`` used to call ``create_adapter_run_time`` with no executor at
all, so every run silently took the binding's ``Serial`` default and there was no
way to ask for another one. The executor now comes from ``NEOFOAM_EXECUTOR``
(:func:`neofoam.solver.neon_runtime.requested_executor`).

The name reaches NeoFOAM's C++ ``createExecutor``, which logs ``Creating Executor
<name>`` before it resolves it — so the log line is the observation for both a
name that resolves (``Serial``) and one that does not, and the run's exit status
separates the two. The selection happens inside the C++ runtime during a real
init, and NeoN/Kokkos plus OpenFOAM keep per-process global state that does not
survive a second in-process run, so each case runs in its own subprocess.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from neofoam.tooling.casebuild import block_mesh, from_template, patch

CASES = Path(__file__).parent / "cases"

# One time step (deltaT 0.005) — the executor is built during initialization, so
# the run only has to get past the first step.
END_TIME = 0.005


def _run_solver(case_path: Path, executor: str | None) -> subprocess.CompletedProcess[str]:
    """Run the framework NeoN solver in ``case_path`` with ``NEOFOAM_EXECUTOR`` set."""
    env = {**os.environ, "FOAM_SIGFPE": "false"}
    env.pop("NEOFOAM_EXECUTOR", None)
    if executor is not None:
        env["NEOFOAM_EXECUTOR"] = executor
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


@pytest.mark.parametrize(
    ("executor", "expected_name", "expect_success"),
    [
        (None, "Serial", True),
        ("Serial", "Serial", True),
        ("NoSuchExecutor", "NoSuchExecutor", False),
    ],
)
def test_neon_runtime_uses_the_requested_executor(
    tmp_path: Path, executor: str | None, expected_name: str, expect_success: bool
) -> None:
    """The run builds the executor NEOFOAM_EXECUTOR names, and fails when it cannot."""
    case = (
        from_template(CASES / "regexSolverKeys")
        | patch("system/controlDict", endTime=END_TIME)
        | block_mesh()
    ).build_at(tmp_path / "executorSelection")

    result = _run_solver(case.path, executor)

    output = result.stdout + result.stderr
    assert f"Creating Executor {expected_name}" in output, (
        f"executor {executor!r} did not reach createExecutor:\n{output[-3000:]}"
    )
    assert (result.returncode == 0) is expect_success, (
        f"executor {executor!r} gave rc={result.returncode}:\n{output[-3000:]}"
    )
