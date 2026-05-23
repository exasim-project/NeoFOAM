# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Shared helpers for incompressibleFluid solver comparison tests.

Ported from ``feat/python_solvers`` to support the minimal incompressibleFluid
port. Provides an ``@requires_openfoam`` skip marker, case setup, field
loading via pybFoam, and field-level numerical comparison.
"""

import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pytest
from pybFoam import volScalarField, volVectorField


def check_openfoam_available() -> bool:
    """Return True iff ``blockMesh`` is on PATH and runnable."""
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


def load_field(
    case_dir: Path, time_dir: Path, field_name: str
) -> Union[volScalarField, volVectorField]:
    """Load a vol*Field from disk by staging it as the only time directory.

    pybFoam discovers the "latest" time on construction, so we copy the
    target time directory into a temporary case as ``0/`` and read from
    there. Returns a volScalarField or volVectorField depending on the
    file header.
    """
    import pybFoam as pyf

    temp_case = case_dir.parent / f"{case_dir.name}_temp_read"
    original_dir = Path.cwd()
    try:
        if temp_case.exists():
            shutil.rmtree(temp_case)
        temp_case.mkdir()
        shutil.copytree(case_dir / "system", temp_case / "system")
        shutil.copytree(case_dir / "constant", temp_case / "constant")
        shutil.copytree(time_dir, temp_case / "0")

        os.chdir(temp_case)
        args = pyf.argList(["test"])
        runTime = pyf.Time(args)
        mesh = pyf.fvMesh(runTime)

        field_path = temp_case / "0" / field_name
        content = field_path.read_text()
        field: Union[volScalarField, volVectorField]
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


def _internal_field_values(
    field: Union[volScalarField, volVectorField],
) -> np.ndarray:
    """Extract the internal field as a numpy array.

    pybFoam exposes the values via either ``internalField()`` or
    ``primitiveField()`` depending on the binding generation, so we try
    each in turn.
    """
    try:
        return np.array(field.internalField())
    except AttributeError:
        try:
            return np.array(field.primitiveField())  # type: ignore[union-attr]
        except AttributeError:
            return np.array(field)


def compare_fields_numerically(
    case1_dir: Path,
    case2_dir: Path,
    time1_dir: Path,
    time2_dir: Path,
    field_name: str,
    rtol: float = 1e-10,
    atol: float = 1e-15,
) -> Tuple[bool, float, float]:
    """Compare two fields elementwise; return (match, max_abs_diff, max_rel_diff)."""
    try:
        field1 = load_field(case1_dir, time1_dir, field_name)
        field2 = load_field(case2_dir, time2_dir, field_name)

        internal1 = _internal_field_values(field1)
        internal2 = _internal_field_values(field2)

        if internal1.shape != internal2.shape:
            print(
                f"  {field_name}: shape mismatch {internal1.shape} vs {internal2.shape}"
            )
            return False, float("inf"), float("inf")

        abs_diff = np.abs(internal1 - internal2)
        max_abs_diff = float(np.max(abs_diff))
        max_val = float(np.max(np.abs(internal1)))
        rel_diff = max_abs_diff / (max_val + 1e-15) if max_val > 0 else max_abs_diff

        match = bool(np.allclose(internal1, internal2, rtol=rtol, atol=atol))
        marker = "OK" if match else "FAIL"
        print(
            f"  {field_name}: {marker} max abs={max_abs_diff:.2e}, "
            f"max rel={rel_diff:.2e}"
        )
        return match, max_abs_diff, rel_diff
    except Exception as exc:
        print(f"  {field_name}: error loading/comparing: {exc}")
        return False, float("inf"), float("inf")


def setup_case(
    source_case: Path,
    test_case: Path,
    end_time: float = 0.01,
    write_interval: float = 0.01,
    run_setfields: bool = False,
) -> None:
    """Copy a tutorial case, restore ``0/``, run blockMesh, override timings.

    ``run_setfields=True`` invokes ``setFields`` after blockMesh — needed
    by cases like hotRoom that initialize a temperature blob via
    ``system/setFieldsDict``.
    """
    if test_case.exists():
        shutil.rmtree(test_case)
    test_case.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_case, test_case)

    orig_dir = test_case / "0.orig"
    zero_dir = test_case / "0"
    if orig_dir.exists():
        if zero_dir.exists():
            shutil.rmtree(zero_dir)
        shutil.copytree(orig_dir, zero_dir)

    result = subprocess.run(
        ["blockMesh", "-case", str(test_case)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"blockMesh failed: {result.stderr}"

    if run_setfields:
        result = subprocess.run(
            ["setFields", "-case", str(test_case)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, f"setFields failed: {result.stderr}"

    control_dict = test_case / "system" / "controlDict"
    lines = control_dict.read_text().split("\n")
    new_lines: List[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("endTime"):
            new_lines.append(f"endTime         {end_time};")
        elif stripped.startswith("writeControl"):
            new_lines.append("writeControl    adjustable;")
        elif stripped.startswith("writeInterval"):
            new_lines.append(f"writeInterval   {write_interval};")
        else:
            new_lines.append(line)
    control_dict.write_text("\n".join(new_lines))


def get_time_directories(case_dir: Path) -> List[Path]:
    """Return time directories in a case, sorted by numeric time value."""
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
) -> Tuple[bool, List[str], dict[str, Tuple[float, float]]]:
    """Compare each named field between two cases at their final time directory."""
    time_dirs_custom = get_time_directories(test_case_custom)
    time_dirs_native = get_time_directories(test_case_native)

    assert len(time_dirs_custom) >= 2, "Custom solver: no output time directories"
    assert len(time_dirs_native) >= 2, "Native solver: no output time directories"

    final_custom = time_dirs_custom[-1]
    final_native = time_dirs_native[-1]

    all_match = True
    failed_fields: List[str] = []
    failed_details: dict[str, Tuple[float, float]] = {}

    for field_name, field_type in fields_to_compare:
        field_path_custom = final_custom / field_name
        field_path_native = final_native / field_name
        assert field_path_custom.exists(), (
            f"Field '{field_name}' not found in custom solver output: {field_path_custom}"
        )
        assert field_path_native.exists(), (
            f"Field '{field_name}' not found in native solver output: {field_path_native}"
        )

        print(f"\nComparing {field_name} ({field_type})...")
        match, max_abs_diff, max_rel_diff = compare_fields_numerically(
            test_case_custom,
            test_case_native,
            final_custom,
            final_native,
            field_name,
            rtol=rtol,
            atol=atol,
        )
        if not match:
            all_match = False
            failed_fields.append(field_name)
            failed_details[field_name] = (max_abs_diff, max_rel_diff)

    return all_match, failed_fields, failed_details
