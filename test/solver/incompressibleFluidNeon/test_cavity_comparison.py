# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""
Comparison test for cavity case: incompressibleFluidNeon vs neoIcoFoam.

Both solvers use NeoN bindings and PISO, so results should match exactly.
This validates that the framework-based solver reproduces the standalone solver.
"""

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest


def _check_openfoam_available() -> bool:
    try:
        result = subprocess.run(["blockMesh", "-help"], capture_output=True, timeout=5)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


requires_openfoam = pytest.mark.skipif(
    not _check_openfoam_available(),
    reason="OpenFOAM not available",
)


def _setup_case(
    source_case: Path,
    test_case: Path,
    end_time: float,
    write_interval: float,
) -> None:
    """Setup a test case from source tutorial."""
    if test_case.exists():
        shutil.rmtree(test_case)

    test_case.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_case, test_case)

    # Restore 0 from 0.orig
    orig_dir = test_case / "0.orig"
    zero_dir = test_case / "0"
    if orig_dir.exists():
        if zero_dir.exists():
            shutil.rmtree(zero_dir)
        shutil.copytree(orig_dir, zero_dir)

    # Run blockMesh
    result = subprocess.run(
        ["blockMesh", "-case", str(test_case)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"blockMesh failed: {result.stderr}"

    # Modify controlDict for short test run
    control_dict = test_case / "system" / "controlDict"
    content = control_dict.read_text()
    lines = content.split("\n")
    new_lines = []
    for line in lines:
        if line.strip().startswith("endTime"):
            new_lines.append(f"endTime         {end_time};")
        elif line.strip().startswith("writeControl"):
            new_lines.append("writeControl    adjustable;")
        elif line.strip().startswith("writeInterval"):
            new_lines.append(f"writeInterval   {write_interval};")
        else:
            new_lines.append(line)
    control_dict.write_text("\n".join(new_lines))


def _read_field_values(time_dir: Path, field_name: str) -> Any:
    """Read field values directly from OpenFOAM ASCII file."""
    field_path = time_dir / field_name
    if not field_path.exists():
        raise FileNotFoundError(f"Field file not found: {field_path}")

    content = field_path.read_text()

    # Find the internalField section
    idx = content.find("internalField")
    if idx == -1:
        raise ValueError(f"No internalField found in {field_path}")

    # Find the data block (between parentheses after nonuniform)
    start = content.find("(", idx)
    end = content.find(")", start)
    if start == -1 or end == -1:
        raise ValueError(f"Could not parse field data in {field_path}")

    data_str = content[start + 1 : end].strip()
    lines = [line.strip() for line in data_str.split("\n") if line.strip()]

    values = []
    for line in lines:
        if line.startswith("(") and line.endswith(")"):
            components = line[1:-1].split()
            values.append([float(c) for c in components])
        else:
            try:
                values.append(float(line))
            except ValueError:
                continue

    return np.array(values)


def _get_final_time(case_dir: Path) -> Path:
    """Get the last output time directory."""
    time_dirs = sorted(
        [
            d
            for d in case_dir.iterdir()
            if d.is_dir()
            and d.name.replace(".", "").replace("-", "").isdigit()
            and float(d.name) > 0
        ],
        key=lambda x: float(x.name),
    )
    assert len(time_dirs) > 0, f"No output time directories in {case_dir}"
    return time_dirs[-1]


@requires_openfoam
def test_cavity_neon_vs_neoicofoam() -> None:
    """Compare incompressibleFluidNeon against neoIcoFoam on cavity case."""

    repo_root = Path(__file__).parent.parent.parent.parent
    source_case = repo_root / "tutorials" / "cavity"

    test_case_framework = repo_root / "test_cases" / "cavity_framework_solver"
    test_case_standalone = repo_root / "test_cases" / "cavity_standalone_solver"

    end_time = 0.05
    write_interval = 0.05

    try:
        print("\n=== Setting up cavity test cases ===")
        _setup_case(source_case, test_case_framework, end_time, write_interval)
        _setup_case(source_case, test_case_standalone, end_time, write_interval)

        # Run the framework solver (incompressibleFluidNeon)
        print("\n=== Running incompressibleFluidNeon ===")
        original_dir = Path.cwd()
        os.chdir(test_case_framework)
        try:
            from neofoam.solver.incompressibleFluidNeon import run as run_framework

            run_framework(["."])
        finally:
            os.chdir(original_dir)

        # Run the standalone solver (neoIcoFoam)
        print("\n=== Running neoIcoFoam ===")
        os.chdir(test_case_standalone)
        try:
            from neofoam.solver.neoIcoFoam import NeoIcoFoam

            solver = NeoIcoFoam(["neoIcoFoam"])
            solver.run()
        finally:
            os.chdir(original_dir)

        # Compare final output
        final_framework = _get_final_time(test_case_framework)
        final_standalone = _get_final_time(test_case_standalone)

        print(f"\nComparing: {final_framework.name} vs {final_standalone.name}")

        fields_to_compare = ["U", "p"]
        all_match = True

        for field_name in fields_to_compare:
            vals_fw = _read_field_values(final_framework, field_name)
            vals_sa = _read_field_values(final_standalone, field_name)

            assert vals_fw.shape == vals_sa.shape, (
                f"{field_name}: shape mismatch {vals_fw.shape} vs {vals_sa.shape}"
            )

            max_abs_diff = float(np.max(np.abs(vals_fw - vals_sa)))
            max_val = float(np.max(np.abs(vals_fw)))
            rel_diff = max_abs_diff / (max_val + 1e-15) if max_val > 0 else max_abs_diff

            print(
                f"  {field_name}: max_abs_diff={max_abs_diff:.2e}, "
                f"rel_diff={rel_diff:.2e}"
            )

            if not np.allclose(vals_fw, vals_sa, rtol=1e-10, atol=1e-15):
                all_match = False
                print(f"  {field_name}: MISMATCH")

        assert all_match, "Field values differ between framework and standalone solvers"
        print("\n=== Test PASSED: Results match ===")

    finally:
        for test_case in [test_case_framework, test_case_standalone]:
            if test_case.exists():
                shutil.rmtree(test_case)
                print(f"Cleaned up: {test_case}")
