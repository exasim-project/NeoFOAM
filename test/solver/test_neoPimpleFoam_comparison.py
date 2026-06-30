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
(1e-2, ~0.2% of peak |p|) absorbs the singularity-amplified drift between the two
stacks while still failing loudly on any gross regression.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_FIELD_READER = Path(__file__).parent / "pyfoam_field_reader.py"

# Disable OpenFOAM's floating-point-exception trap before pybFoam is imported (in the
# field-reader subprocess). Constructing a Foam::Time enables trapfpe() process-wide,
# which then aborts on a benign denormal while loading/comparing fields.
os.environ.setdefault("FOAM_SIGFPE", "false")


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


def _load_internal_fields(
    case: Path, time_dir: Path, field_names: tuple[str, ...], out_dir: Path
) -> dict[str, np.ndarray]:
    """Read vol*Field internal fields via pybFoam, isolated in a subprocess.

    Constructing a ``Foam::Time`` pulls in OpenFOAM per-process global state that is
    corrupted by a second construction in the same interpreter — subsequent reads come
    back as nan (the same global-state hazard that forces each solver run into its own
    process). Each case is therefore read in its own subprocess (``pyfoam_field_reader``),
    which constructs ``Time`` exactly once and dumps every requested field to ``.npy``.
    """
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    env = {**os.environ, "FOAM_SIGFPE": "false"}
    result = subprocess.run(
        [
            sys.executable,
            str(_FIELD_READER),
            str(case),
            str(time_dir),
            str(out_dir),
            *field_names,
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"field read failed (rc={result.returncode}):\n{result.stderr[-3000:]}"
    )
    return {name: np.load(out_dir / f"{name}.npy") for name in field_names}


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

    fields = ("p", "U")
    of_vals = _load_internal_fields(of_case, of_final, fields, tmp_path / "of_read")
    neo_vals = _load_internal_fields(neo_case, neo_final, fields, tmp_path / "neo_read")

    tol = 1e-2
    failures = []
    for field_name in fields:
        of_field = of_vals[field_name]
        neo_field = neo_vals[field_name]
        assert of_field.shape == neo_field.shape, (
            f"{field_name}: shape {of_field.shape} vs {neo_field.shape}"
        )
        max_abs = float(np.max(np.abs(of_field - neo_field)))
        peak = float(np.max(np.abs(of_field)))
        print(
            f"{field_name}: max abs diff = {max_abs:.3e} (peak |{field_name}| = {peak:.3e})"
        )
        if not np.allclose(of_field, neo_field, rtol=0.0, atol=tol):
            failures.append(f"{field_name}(max abs={max_abs:.3e})")

    if failures:
        pytest.fail(
            "neoPimpleFoam diverged from pimpleFoam beyond tol="
            f"{tol:.0e}: " + ", ".join(failures)
        )
