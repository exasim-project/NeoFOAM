# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Cross-backend solver comparison: NeoN turbulence vs ``incompressibleFluid``.

Runs the framework ``incompressibleFluidNeoN`` solver — whose turbulence is the
pure-Python NeoN ModelSpec family (laminar / kEpsilon / SpalartAllmaras /
kOmegaSST) — and the trusted pybFoam ``incompressibleFluid`` solver on
byte-identical copies of the seeded ``cases/turbulentBox`` case, then compares
the final internal fields (velocity, pressure and the turbulence fields the
model owns) cell-by-cell.

The case is the committed no-wall-function box (seeded divergence-free random
``U`` + random ``k``/``epsilon``/``nuTilda``/``omega``, all-zeroGradient BCs,
one ``wall`` patch so wallDist is defined) with every linear solve driven to a
very tight tolerance — both backends then discretise the same equations and
converge to the same solution, so the comparison measures discretisation
parity of the full PIMPLE + turbulence step, not linear-solver residuals.

Process hygiene follows the established comparison tests: each solver runs in
its own subprocess (NeoN/Kokkos + OpenFOAM per-process global state), and each
case is read back in its own subprocess too (``pyfoam_field_reader``).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).parent
_CASE = _HERE / "cases" / "turbulentBox"
_FIELD_READER = _HERE.parent / "pyfoam_field_reader.py"

# Per-model fields to compare: U/p always, plus the transport unknowns and the
# eddy viscosity the closure owns (laminar owns no transport fields).
CASES = [
    ("laminar", ("U", "p")),
    ("kEpsilon", ("U", "p", "k", "epsilon", "nut")),
    ("SpalartAllmaras", ("U", "p", "nuTilda", "nut")),
    ("kOmegaSST", ("U", "p", "k", "omega", "nut")),
]

# Both backends solve the same matrices to ~1e-13/1e-14; over the 20 fixed
# steps the remaining linear-solver and rounding differences stay tiny.
RTOL = 1e-8
ATOL_SCALE = 1e-10  # atol = ATOL_SCALE * peak |reference field|


def _prepare_case(model: str, dest: Path) -> None:
    """Copy the committed case, select the model's turbulenceProperties, mesh it."""
    shutil.copytree(_CASE, dest, ignore=shutil.ignore_patterns("models"))
    shutil.copyfile(
        _CASE / "models" / model / "turbulenceProperties",
        dest / "constant" / "turbulenceProperties",
    )
    result = subprocess.run(
        ["blockMesh", "-case", str(dest)], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, f"blockMesh failed: {result.stderr}"


def _run_solver(code: str, case: Path, label: str) -> None:
    """Run a solver snippet in an isolated subprocess in ``case``."""
    env = {**os.environ, "FOAM_SIGFPE": "false"}
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(case),
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, (
        f"{label} failed (rc={result.returncode}):\n"
        f"{result.stdout[-2000:]}\n{result.stderr[-3000:]}"
    )


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
    case: Path, fields: tuple[str, ...], out_dir: Path
) -> dict[str, np.ndarray]:
    """Read the final time's internal fields via pybFoam, isolated in a subprocess."""
    out_dir.mkdir(parents=True)
    env = {**os.environ, "FOAM_SIGFPE": "false"}
    result = subprocess.run(
        [
            sys.executable,
            str(_FIELD_READER),
            str(case),
            str(_final_time_dir(case)),
            str(out_dir),
            *fields,
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"field read failed for {case} (rc={result.returncode}):\n"
        f"{result.stdout}\n{result.stderr[-3000:]}"
    )
    return {name: np.load(out_dir / f"{name}.npy") for name in fields}


@pytest.mark.parametrize(("model", "fields"), CASES, ids=[c[0] for c in CASES])
def test_neon_turbulence_solver_matches_incompressibleFluid(
    model: str, fields: tuple[str, ...], tmp_path: Path
) -> None:
    """incompressibleFluidNeoN (pure-Python NeoN turbulence) tracks incompressibleFluid."""
    neon_case = tmp_path / "neon"
    reference_case = tmp_path / "reference"
    _prepare_case(model, neon_case)
    _prepare_case(model, reference_case)

    _run_solver(
        "from neofoam.solver.incompressibleFluidNeoN import run;"
        " run(['incompressibleFluidNeoN'])",
        neon_case,
        f"incompressibleFluidNeoN ({model})",
    )
    _run_solver(
        "from neofoam.solver.incompressibleFluid import run;"
        " run(['incompressibleFluid'])",
        reference_case,
        f"incompressibleFluid ({model})",
    )

    neon_final = _final_time_dir(neon_case)
    reference_final = _final_time_dir(reference_case)
    assert neon_final.name == reference_final.name, (
        f"solvers wrote different final times: "
        f"{neon_final.name} vs {reference_final.name}"
    )

    neon_vals = _load_internal_fields(neon_case, fields, tmp_path / "neon_read")
    reference_vals = _load_internal_fields(
        reference_case, fields, tmp_path / "reference_read"
    )

    failures = []
    for name in fields:
        reference = reference_vals[name]
        result = neon_vals[name]
        assert reference.shape == result.shape, (
            f"{name}: shape {reference.shape} vs {result.shape}"
        )
        peak = float(np.max(np.abs(reference))) or 1.0
        max_abs = float(np.max(np.abs(result - reference)))
        print(f"[{model}] {name}: max abs diff = {max_abs:.3e} (peak = {peak:.3e})")
        if not np.allclose(result, reference, rtol=RTOL, atol=ATOL_SCALE * peak):
            failures.append(f"{name}(max abs={max_abs:.3e}, peak={peak:.3e})")

    assert not failures, (
        f"{model}: NeoN solver diverged from incompressibleFluid beyond "
        f"rtol={RTOL:.0e}: " + ", ".join(failures)
    )
