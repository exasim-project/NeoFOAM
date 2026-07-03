# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Shared helpers for incompressibleFluid solver comparison tests.

Ported from ``feat/python_solvers`` to support the minimal incompressibleFluid
port. Provides case setup, field loading via pybFoam, and field-level numerical
comparison. OpenFOAM/pybFoam are treated as always available.

Field reads run in an **isolated subprocess** (``pyfoam_field_reader.py``):
constructing ``Foam::Time`` more than once per process corrupts OpenFOAM's
global state — an in-process second read silently returns the *previous*
case's fields, which made every comparison a false pass.
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

_FIELD_READER = Path(__file__).parent.parent / "pyfoam_field_reader.py"


def read_internal_fields(
    case_dir: Path, time_dir: Path, field_names: List[str]
) -> Dict[str, np.ndarray]:
    """Read vol*Field internal fields from ``time_dir`` of a case.

    One fresh interpreter per call — exactly one ``Foam::Time`` per process.
    """
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        result = subprocess.run(
            [
                sys.executable,
                str(_FIELD_READER),
                str(case_dir),
                str(time_dir),
                str(out_dir),
                *field_names,
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, (
            f"field reader failed for {case_dir}:\n{result.stdout}\n{result.stderr}"
        )
        return {name: np.load(out_dir / f"{name}.npy") for name in field_names}


def compare_fields_numerically(
    values1: np.ndarray,
    values2: np.ndarray,
    field_name: str,
    rtol: float = 1e-10,
    atol: float = 1e-15,
) -> Tuple[bool, float, float]:
    """Compare two internal fields elementwise; (match, max_abs_diff, max_rel_diff)."""
    if values1.shape != values2.shape:
        print(f"  {field_name}: shape mismatch {values1.shape} vs {values2.shape}")
        return False, float("inf"), float("inf")

    abs_diff = np.abs(values1 - values2)
    max_abs_diff = float(np.max(abs_diff))
    max_val = float(np.max(np.abs(values1)))
    rel_diff = max_abs_diff / (max_val + 1e-15) if max_val > 0 else max_abs_diff

    match = bool(np.allclose(values1, values2, rtol=rtol, atol=atol))
    marker = "OK" if match else "FAIL"
    print(
        f"  {field_name}: {marker} max abs={max_abs_diff:.2e}, max rel={rel_diff:.2e}"
    )
    return match, max_abs_diff, rel_diff


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

    for field_name, _field_type in fields_to_compare:
        field_path_custom = final_custom / field_name
        field_path_native = final_native / field_name
        assert field_path_custom.exists(), (
            f"Field '{field_name}' not found in custom solver output: {field_path_custom}"
        )
        assert field_path_native.exists(), (
            f"Field '{field_name}' not found in native solver output: {field_path_native}"
        )

    field_names = [name for name, _ in fields_to_compare]
    custom_fields = read_internal_fields(test_case_custom, final_custom, field_names)
    native_fields = read_internal_fields(test_case_native, final_native, field_names)

    for field_name, field_type in fields_to_compare:
        print(f"\nComparing {field_name} ({field_type})...")
        match, max_abs_diff, max_rel_diff = compare_fields_numerically(
            custom_fields[field_name],
            native_fields[field_name],
            field_name,
            rtol=rtol,
            atol=atol,
        )
        if not match:
            all_match = False
            failed_fields.append(field_name)
            failed_details[field_name] = (max_abs_diff, max_rel_diff)

    return all_match, failed_fields, failed_details
