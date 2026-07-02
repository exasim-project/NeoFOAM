# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""End-to-end cavity run test for the framework ``incompressibleFluidNeoN`` solver.

Copies the lid-driven-cavity case (``test/setup_pimple``), meshes it, and runs
the framework solver in an isolated subprocess (NeoN/Kokkos + OpenFOAM keep
per-process global state that does not survive a second in-process run), then
asserts the run completed with finite ``p`` / ``U`` output fields.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

# Short transient: fixed steps (deltaT 0.005) over 100 steps exercises the full
# outer-PIMPLE / inner-PISO machinery while staying fast.
END_TIME = 0.5
N_STEPS = 100  # END_TIME / deltaT


def _set_timings(case: Path) -> None:
    control_dict = case / "system" / "controlDict"
    new_lines = []
    for line in control_dict.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("endTime "):
            new_lines.append(f"endTime {END_TIME};")
        elif stripped.startswith("writeControl"):
            new_lines.append("writeControl timeStep;")
        elif stripped.startswith("writeInterval"):
            new_lines.append(f"writeInterval {N_STEPS};")
        else:
            new_lines.append(line)
    control_dict.write_text("\n".join(new_lines) + "\n")


def _prepare_case(source: Path, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(source, dest)
    _set_timings(dest)
    result = subprocess.run(
        ["blockMesh", "-case", str(dest)], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, f"blockMesh failed: {result.stderr}"


def _final_time_dir(case: Path) -> Path:
    times = sorted(
        (
            d
            for d in case.iterdir()
            if d.is_dir() and d.name.replace(".", "").isdigit() and float(d.name) > 0
        ),
        key=lambda d: float(d.name),
    )
    assert times, f"no output time directories in {case}"
    return times[-1]


_FLOAT = r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?"


def _read_internal(time_dir: Path, field_name: str) -> np.ndarray:
    """Parse an OpenFOAM field's internalField values (uniform or nonuniform)."""
    txt = (time_dir / field_name).read_text()
    m = re.search(r"internalField\s+nonuniform[^(]*\((.*?)\)\s*;", txt, re.S)
    if m:
        return np.array([float(x) for x in re.findall(_FLOAT, m.group(1))])
    m = re.search(rf"internalField\s+uniform\s+\(?\s*((?:{_FLOAT}\s*)+)\)?\s*;", txt)
    assert m, f"could not parse internalField of {field_name}"
    return np.array([float(x) for x in re.findall(_FLOAT, m.group(1))])


def test_incompressibleFluidNeoN_cavity_runs(tmp_path: Path) -> None:
    """The framework NeoN solver completes the cavity case with finite fields."""
    repo_root = Path(__file__).parent.parent.parent.parent
    source_case = repo_root / "test" / "setup_pimple"
    case = tmp_path / "cavity"
    _prepare_case(source_case, case)

    # Isolated subprocess: NeoN/Kokkos + OpenFOAM per-process state. FOAM_SIGFPE
    # is disabled so the signal handler does not abort on a benign denormal.
    env = {**os.environ, "FOAM_SIGFPE": "false"}
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from neofoam.solver.incompressibleFluidNeoN import run;"
            " run(['incompressibleFluidNeoN'])",
        ],
        cwd=str(case),
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, (
        f"incompressibleFluidNeoN failed (rc={result.returncode}):\n"
        f"{result.stdout[-2000:]}\n{result.stderr[-3000:]}"
    )
    assert "End" in result.stdout

    final = _final_time_dir(case)
    assert final.name == str(END_TIME), (
        f"expected final time dir {END_TIME}, got {final.name}"
    )
    p = _read_internal(final, "p")
    u = _read_internal(final, "U")
    assert np.all(np.isfinite(p)), "p contains non-finite values"
    assert np.all(np.isfinite(u)), "U contains non-finite values"
    assert float(np.max(np.abs(u))) > 0.0, "U stayed identically zero"
