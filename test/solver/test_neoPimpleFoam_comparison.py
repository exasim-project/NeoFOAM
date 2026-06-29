# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""End-to-end OpenFOAM-parity test for the NeoN-based ``neoPimpleFoam`` port.

The Python analogue of ``test/pimpleParity.cpp``: run the real OpenFOAM
``pimpleFoam`` binary and the in-process ``NeoPimpleFoam`` solver on byte-identical
copies of the lid-driven-cavity case (``test/setup_pimple``), then assert the
converged INTERNAL ``p``/``U`` fields agree.

The two stacks are independent (OpenFOAM PCG/PBiCGStab vs NeoN/Ginkgo), and the
cavity has a pressure singularity at the moving-lid corners that amplifies the
per-step solver difference, so a bit-level match is unattainable. The tolerance
(5e-3, ~0.1% of peak |p|) mirrors the C++ acceptance bar: it absorbs the
singularity-amplified drift while still failing loudly on any gross regression.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Union

import numpy as np
import pytest


def run_neopimplefoam(case: Path) -> None:
    """Run the NeoN neoPimpleFoam solver in an isolated subprocess in ``case``.

    NeoN/Kokkos and OpenFOAM keep per-process global state that does not survive a
    second in-process solver run, so each invocation gets its own interpreter.
    FOAM_SIGFPE is disabled so OpenFOAM's signal handler does not abort the NeoN
    solve on a benign denormal in a boundary cell.
    """
    env = {**os.environ, "FOAM_SIGFPE": "false"}
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from neofoam.solver.neoPimpleFoam import NeoPimpleFoam;"
            " NeoPimpleFoam(['neoPimpleFoam']).run()",
        ],
        cwd=str(case),
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, (
        f"neoPimpleFoam failed (rc={result.returncode}):\n{result.stderr[-3000:]}"
    )


def _openfoam_available() -> bool:
    try:
        return (
            subprocess.run(
                ["blockMesh", "-help"], capture_output=True, timeout=5
            ).returncode
            == 0
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


requires_openfoam = pytest.mark.skipif(
    not _openfoam_available(), reason="OpenFOAM not available"
)

# Short transient: both solvers take identical fixed steps (deltaT 0.005), so a
# shorter run keeps the test fast while exercising the full outer-PIMPLE /
# inner-PISO machinery against pimpleFoam.
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


def _load_internal(case: Path, time_dir: Path, field_name: str) -> np.ndarray:
    """Load a vol*Field internal field as a numpy array via pybFoam.

    pybFoam discovers the latest time on construction, so the target time dir is
    staged as ``0/`` in a temporary case and read from there.
    """
    import pybFoam as pyf
    from pybFoam import volScalarField, volVectorField

    temp_case = case.parent / f"{case.name}_read_{field_name}"
    original_dir = Path.cwd()
    try:
        if temp_case.exists():
            shutil.rmtree(temp_case)
        temp_case.mkdir()
        shutil.copytree(case / "system", temp_case / "system")
        shutil.copytree(case / "constant", temp_case / "constant")
        shutil.copytree(time_dir, temp_case / "0")

        os.chdir(temp_case)
        runTime = pyf.Time(pyf.argList(["test"]))
        mesh = pyf.fvMesh(runTime)

        content = (temp_case / "0" / field_name).read_text()
        field: Union[volScalarField, volVectorField]
        if "volScalarField" in content:
            field = volScalarField.read_field(mesh, field_name)
        elif "volVectorField" in content:
            field = volVectorField.read_field(mesh, field_name)
        else:
            raise ValueError(f"Unknown field type for {field_name}")

        try:
            return np.array(field.internalField())
        except AttributeError:
            return np.array(field)
    finally:
        os.chdir(original_dir)
        if temp_case.exists():
            shutil.rmtree(temp_case)


@requires_openfoam
def test_neoPimpleFoam_matches_pimpleFoam(tmp_path: Path) -> None:
    """Converged NeoN neoPimpleFoam internal p/U match OpenFOAM pimpleFoam."""
    repo_root = Path(__file__).parent.parent.parent
    source_case = repo_root / "test" / "setup_pimple"

    of_case = tmp_path / "openfoam"
    neo_case = tmp_path / "neofoam"
    _prepare_case(source_case, of_case)
    _prepare_case(source_case, neo_case)

    # --- reference: the real OpenFOAM pimpleFoam binary ---
    result = subprocess.run(
        ["pimpleFoam", "-case", str(of_case)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, f"pimpleFoam failed: {result.stderr}"

    # --- candidate: the NeoN neoPimpleFoam port, run as its own process. NeoN
    # (Kokkos) and OpenFOAM hold per-process global state that does not survive a
    # second in-process solver run, so isolate each run in a subprocess. ---
    run_neopimplefoam(neo_case)

    of_final = _final_time_dir(of_case)
    neo_final = _final_time_dir(neo_case)
    assert of_final.name == neo_final.name, (
        f"solvers wrote different final times: {of_final.name} vs {neo_final.name}"
    )

    tol = 5e-3
    failures = []
    for field_name in ("p", "U"):
        of_vals = _load_internal(of_case, of_final, field_name)
        neo_vals = _load_internal(neo_case, neo_final, field_name)
        assert of_vals.shape == neo_vals.shape, (
            f"{field_name}: shape {of_vals.shape} vs {neo_vals.shape}"
        )
        max_abs = float(np.max(np.abs(of_vals - neo_vals)))
        peak = float(np.max(np.abs(of_vals)))
        print(
            f"{field_name}: max abs diff = {max_abs:.3e} (peak |{field_name}| = {peak:.3e})"
        )
        if not np.allclose(of_vals, neo_vals, rtol=0.0, atol=tol):
            failures.append(f"{field_name}(max abs={max_abs:.3e})")

    if failures:
        pytest.fail(
            "neoPimpleFoam diverged from pimpleFoam beyond tol="
            f"{tol:.0e}: " + ", ".join(failures)
        )
