# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
Shared helper functions for solver comparison tests.

This module provides utilities for comparing OpenFOAM solver outputs,
including field loading, numerical comparison, and test case setup.
"""

import os
import shutil
import subprocess
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pytest
from pybFoam import volScalarField, volVectorField


def check_openfoam_available():
    """Check if OpenFOAM is available and properly configured."""
    try:
        result = subprocess.run(
            ["blockMesh", "-help"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


requires_openfoam = pytest.mark.skipif(
    not check_openfoam_available(),
    reason="OpenFOAM not available",
)


def load_field(case_dir: Path, time_dir: Path, field_name: str):
    """
    Load a volScalarField or volVectorField from disk.

    Args:
        case_dir: Path to the OpenFOAM case directory
        time_dir: Path to the specific time directory
        field_name: Name of the field to load

    Returns:
        Loaded field object (volScalarField or volVectorField)
    """
    import pybFoam as pyf

    # We need to temporarily make the target time appear as the "latest" time
    # Create a temporary directory structure
    temp_case = case_dir.parent / f"{case_dir.name}_temp_read"
    try:
        # Copy only the essential structure: system, constant, and the specific time dir as "0"
        if temp_case.exists():
            shutil.rmtree(temp_case)
        temp_case.mkdir()

        # Copy system and constant
        shutil.copytree(case_dir / "system", temp_case / "system")
        shutil.copytree(case_dir / "constant", temp_case / "constant")

        # Copy the target time directory as "0" so it's the only time available
        shutil.copytree(time_dir, temp_case / "0")

        # Now change to temp case and read
        original_dir = Path.cwd()
        os.chdir(temp_case)

        # Create argList and Time
        args = pyf.argList(["test"])
        runTime = pyf.Time(args)

        # Read mesh
        mesh = pyf.fvMesh(runTime)

        # Determine field type by checking the file header and load the field
        field_path = temp_case / "0" / field_name
        with open(field_path) as f:
            content = f.read()
            if "volScalarField" in content:
                field = volScalarField.read_field(mesh, field_name)
            elif "volVectorField" in content:
                field = volVectorField.read_field(mesh, field_name)
            else:
                raise ValueError(f"Unknown field type for {field_name}")

        return field
    finally:
        os.chdir(original_dir)
        if temp_case.exists():
            shutil.rmtree(temp_case)


def compare_fields_numerically(
    case1_dir: Path,
    case2_dir: Path,
    time1_dir: Path,
    time2_dir: Path,
    field_name: str,
    rtol: float = 1e-10,
    atol: float = 1e-15,
) -> bool:
    """
    Compare two OpenFOAM fields by loading actual field values and comparing numerically.

    Args:
        case1_dir: Path to first case directory
        case2_dir: Path to second case directory
        time1_dir: Path to time directory in first case
        time2_dir: Path to time directory in second case
        field_name: Name of the field to compare
        rtol: Relative tolerance for comparison
        atol: Absolute tolerance for comparison

    Returns:
        True if fields match within tolerance, False otherwise
    """
    try:
        field1 = load_field(case1_dir, time1_dir, field_name)
        field2 = load_field(case2_dir, time2_dir, field_name)

        # Get internal field values - try different possible method names
        try:
            internal1 = np.array(field1.internalField())
            internal2 = np.array(field2.internalField())
        except AttributeError:
            # Maybe it's a property or has different name
            try:
                internal1 = np.array(field1.primitiveField())
                internal2 = np.array(field2.primitiveField())
            except AttributeError:
                # Last resort - try as attribute
                internal1 = np.array(field1)
                internal2 = np.array(field2)

        # Compare shapes
        if internal1.shape != internal2.shape:
            print(
                f"  {field_name}: ✗ Shape mismatch: {internal1.shape} vs {internal2.shape}"
            )
            return False

        # Compute differences
        abs_diff = np.abs(internal1 - internal2)
        max_abs_diff = np.max(abs_diff)

        # Compute relative difference
        max_val = np.max(np.abs(internal1))
        rel_diff = max_abs_diff / (max_val + 1e-15) if max_val > 0 else max_abs_diff

        # Check if within tolerance
        if np.allclose(internal1, internal2, rtol=rtol, atol=atol):
            print(
                f"  {field_name}: ✓ Max abs diff: {max_abs_diff:.2e}, Max rel diff: {rel_diff:.2e}"
            )
            return True
        else:
            print(
                f"  {field_name}: ✗ Max abs diff: {max_abs_diff:.2e}, Max rel diff: {rel_diff:.2e}"
            )
            print(f"    Tolerance: rtol={rtol}, atol={atol}")
            return False
    except Exception as e:
        print(f"  {field_name}: ✗ Error loading/comparing: {str(e)}")
        return False


def setup_case(
    source_case: Path,
    test_case: Path,
    end_time: float = 0.01,
    write_interval: float = 0.01,
    run_setfields: bool = False,
):
    """
    Setup a test case from source.

    Args:
        source_case: Path to source tutorial case
        test_case: Path where test case should be created
        end_time: End time for simulation
        write_interval: Write interval for output
        run_setfields: Whether to run setFields after blockMesh (e.g., for hotRoom)
    """
    # Clean up any existing test case
    if test_case.exists():
        shutil.rmtree(test_case)

    # Copy the case
    test_case.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_case, test_case)

    # Setup initial conditions
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

    # Run setFields if needed (for cases like hotRoom)
    if run_setfields:
        result = subprocess.run(
            ["setFields", "-case", str(test_case)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, f"setFields failed: {result.stderr}"

    # Modify controlDict for test run
    control_dict = test_case / "system" / "controlDict"
    control_dict_content = control_dict.read_text()

    # Update timing parameters
    lines = control_dict_content.split("\n")
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


def get_time_directories(case_dir: Path) -> List[Path]:
    """
    Get sorted list of time directories in a case.

    Args:
        case_dir: Path to the OpenFOAM case directory

    Returns:
        Sorted list of time directory paths
    """
    return sorted(
        [
            d
            for d in case_dir.iterdir()
            if d.is_dir() and d.name.replace(".", "").replace("-", "").isdigit()
        ],
        key=lambda x: float(x.name),
    )


def compare_solver_fields(
    test_case_custom: Path,
    test_case_native: Path,
    fields_to_compare: List[Tuple[str, str]],
    rtol: float = 1e-10,
    atol: float = 1e-15,
) -> Tuple[bool, List[str]]:
    """
    Compare fields between two solver outputs.

    Args:
        test_case_custom: Path to custom solver case
        test_case_native: Path to native solver case
        fields_to_compare: List of (field_name, field_type) tuples
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        Tuple of (all_match: bool, failed_fields: List[str])
    """
    # Find final time directories
    time_dirs_custom = get_time_directories(test_case_custom)
    time_dirs_native = get_time_directories(test_case_native)

    assert len(time_dirs_custom) >= 2, "Custom solver: no output time directories"
    assert len(time_dirs_native) >= 2, "Native solver: no output time directories"

    # Compare final time step
    print("\n=== Comparing results ===")
    final_time_custom = time_dirs_custom[-1]
    final_time_native = time_dirs_native[-1]

    print("\n=== Comparing fields ===")
    print(
        f"Fields to compare: {', '.join([f'{name} ({ftype})' for name, ftype in fields_to_compare])}"
    )

    all_match = True
    failed_fields = []

    for field_name, field_type in fields_to_compare:
        field_path_custom = final_time_custom / field_name
        field_path_native = final_time_native / field_name

        if field_path_custom.exists() and field_path_native.exists():
            print(f"\nComparing {field_name} ({field_type})...")
            if not compare_fields_numerically(
                test_case_custom,
                test_case_native,
                final_time_custom,
                final_time_native,
                field_name,
                rtol=rtol,
                atol=atol,
            ):
                all_match = False
                failed_fields.append(field_name)
        else:
            print(f"  {field_name}: Skipped (field not found in both cases)")

    return all_match, failed_fields
