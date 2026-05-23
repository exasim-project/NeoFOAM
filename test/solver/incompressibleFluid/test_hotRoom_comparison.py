# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Log-level comparison: incompressibleFluid + boussinesq vs native solver.

Runs both the NeoFOAM solver and OpenFOAM's
``buoyantBoussinesqPimpleFoam`` on the bundled ``tutorials/hotRoom`` and
checks that the same equations are solved in the same order — same
"Solving for <field>" lines, same total count. A field-by-field
numerical comparison is too tight a tolerance for buoyancy-driven flow
(non-linear coupling amplifies small ordering/relaxation differences),
so the log-equivalence check is the source-branch's chosen acceptance
criterion.
"""

import os
import re
import shutil
import subprocess
from pathlib import Path

from neofoam.solver.incompressibleFluid import run

from .comparison_helpers import (
    requires_openfoam,
    setup_case,
)

# Disable OpenFOAM floating-point exception trapping for in-process runs;
# the solver constructs intermediate fields whose deltas legitimately
# underflow on the first iteration.
os.environ["FOAM_SIGFPE"] = ""


def _extract_solving_lines(log: str) -> list[str]:
    return [line.strip() for line in log.splitlines() if "Solving for" in line]


def _extract_residual_fields(log: str) -> list[str]:
    pattern = re.compile(r"Solving for (\w+),")
    return pattern.findall(log)


@requires_openfoam
def test_hotRoom_solver_comparison() -> None:
    """Compare incompressibleFluid+boussinesq vs buoyantBoussinesqPimpleFoam."""
    repo_root = Path(__file__).parent.parent.parent.parent
    source_case = repo_root / "tutorials" / "hotRoom"

    test_case_custom = repo_root / "test_cases" / "hotRoom_custom_solver"
    test_case_native = repo_root / "test_cases" / "hotRoom_native_solver"

    end_time = 1000.0
    write_interval = 200.0

    try:
        setup_case(
            source_case,
            test_case_custom,
            end_time,
            write_interval,
            run_setfields=True,
        )
        setup_case(
            source_case,
            test_case_native,
            end_time,
            write_interval,
            run_setfields=True,
        )

        original_dir = Path.cwd()
        os.chdir(test_case_custom)
        custom_log = test_case_custom / "solver.log"
        try:
            run(["incompressibleFluid"], log_file=custom_log)
        finally:
            os.chdir(original_dir)

        custom_output = custom_log.read_text() if custom_log.exists() else ""

        native_log = test_case_native / "solver.log"
        with open(native_log, "w") as f:
            result = subprocess.run(
                ["buoyantBoussinesqPimpleFoam", "-case", str(test_case_native)],
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=120,
            )
        assert result.returncode == 0, (
            f"buoyantBoussinesqPimpleFoam failed with returncode {result.returncode}"
        )
        native_output = native_log.read_text()

        custom_fields = _extract_residual_fields(custom_output)
        native_fields = _extract_residual_fields(native_output)

        assert len(custom_fields) > 0, (
            f"Custom solver did not solve any equations.\nLog:\n{custom_output}"
        )

        assert custom_fields == native_fields, (
            f"Solved fields differ.\n"
            f"Custom: {custom_fields}\n"
            f"Native: {native_fields}\n\n"
            f"Custom log:\n{custom_output}\n\n"
            f"Native log:\n{native_output}"
        )

        custom_solving = _extract_solving_lines(custom_output)
        native_solving = _extract_solving_lines(native_output)

        assert len(custom_solving) == len(native_solving), (
            f"Number of solving lines differs.\n"
            f"Custom ({len(custom_solving)}):\n"
            + "\n".join(custom_solving)
            + f"\n\nNative ({len(native_solving)}):\n"
            + "\n".join(native_solving)
        )

    finally:
        for tc in [test_case_custom, test_case_native]:
            if tc.exists():
                shutil.rmtree(tc)
