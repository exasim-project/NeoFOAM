# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Steady-state (SIMPLE) turbulence + wall-function comparison on pitzDaily.

Runs the framework ``incompressibleFluidNeoN`` solver — whose turbulence is the
pure-Python NeoN ModelSpec family with wall functions — and the trusted pybFoam
``incompressibleFluid`` solver (native OpenFOAM turbulence) on byte-identical
copies of ``cases/pitzDailySteady``, then compares the final internal fields
cell-by-cell. Parameterised over three models (each overlaying its own
``turbulenceProperties`` + wall-function ``0/`` fields from
``cases/pitzDailySteady/models/<name>``):

* **kEpsilon** — epsilon / kqR / nutk wall functions,
* **kOmegaSST** — omega / kqR / nutk wall functions,
* **SpalartAllmaras** — nutUSpalding (nuTilda fixedValue 0 at walls).

The case is the pitzDaily geometry adapted to schemes both backends discretise
identically (``steadyState`` ddt, ``Gauss upwind`` divergence, uncorrected
laplacians/snGrad, literal solver keys at tight tolerance) with the standard
steady relaxation set — steady SIMPLE diverges without under-relaxation.

Two comparisons with different discriminating power:

1. **2 iterations (per-step parity)** — the sharp discretisation check. U/p are
   machine precision for every model (~1e-12); kEpsilon/SpalartAllmaras are
   machine precision throughout. Every wall-treatment bug found while porting
   (BINOMIAL-vs-STEPWISE blender, nutk viscous-sublayer floor, stale nut
   boundaries, missing cell pin, the kOmegaSST production-cap ordering, the
   nutUSpalding tolerance) failed this at >=1e-3.
2. **500 iterations (converged, ~2 min/solver)** — the fixed point itself.
   kEpsilon contracts to machine precision; the stiffer wall functions
   (SpalartAllmaras, and especially kOmegaSST with its ~1e5 omega pin) contract
   the *same* fixed point far more slowly, so their bounds are peak-relative.
   See ``MODELS`` and report/steady-turbulence-wall-functions.md.

A third test pins the reference itself: ``incompressibleFluid`` (SIMPLE) against
the native ``simpleFoam`` binary, which is bitwise on this case.

Process hygiene follows the established comparison tests: each solver runs in
its own subprocess (NeoN/Kokkos + OpenFOAM per-process global state), and each
case is read back in its own subprocess too (``pyfoam_field_reader``).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).parent
_CASE = _HERE / "cases" / "pitzDailySteady"
_FIELD_READER = _HERE.parent / "pyfoam_field_reader.py"

# Per-model comparison spec:
#     (fields, (rtol, atol_scale)_roundoff, (rtol, atol_scale)_converged)
# with atol = atol_scale * peak|reference field|. U/p always, plus the transport
# unknowns + eddy viscosity the closure owns. Each model overlays its own
# turbulenceProperties + wall-function 0/ fields from
# ``cases/pitzDailySteady/models/<name>``.
#
# Per-step (2-iteration) parity is the sharp discretization check and is
# machine-precision for every model's momentum/pressure fields (U/p ~1e-12);
# kEpsilon/SpalartAllmaras are machine-precision throughout, while kOmegaSST's
# near-wall omega/k carry ~1e-6 (see below).
#
# At convergence (500 iterations) the *stiffer* wall functions have not fully
# contracted within the CI iteration budget: SpalartAllmaras (iterative
# Spalding nut) and especially kOmegaSST (omega pinned to ~1e5, ~100x stiffer
# than epsilon's ~1e3) inject per-step round-off that the SIMPLE loop contracts
# only slowly — the same fixed point (kOmegaSST k tightens ~10x from 30 to 500
# iterations), not a different one. Their converged bounds are therefore a
# peak-relative "agree to <=0.1% of peak" check rather than machine precision.
# See report/steady-turbulence-wall-functions.md ("convergence rate").
MODELS = {
    "kEpsilon": (("U", "p", "k", "epsilon", "nut"), (1e-8, 1e-10), (1e-8, 1e-10)),
    "SpalartAllmaras": (("U", "p", "nuTilda", "nut"), (1e-8, 1e-10), (3e-4, 3e-4)),
    "kOmegaSST": (("U", "p", "k", "omega", "nut"), (3e-5, 1e-10), (1e-3, 1e-3)),
}


def _prepare_case(model: str, dest: Path, end_time: int) -> None:
    """Copy the committed case, overlay the model, set the iteration count, mesh it."""
    shutil.copytree(_CASE, dest, ignore=shutil.ignore_patterns("models"))
    overlay = _CASE / "models" / model
    shutil.copyfile(
        overlay / "turbulenceProperties", dest / "constant" / "turbulenceProperties"
    )
    for field in (overlay / "0").iterdir():
        shutil.copyfile(field, dest / "0" / field.name)
    control_dict = dest / "system" / "controlDict"
    text = control_dict.read_text()
    text = re.sub(
        r"^endTime\s+\S+;", f"endTime         {end_time};", text, count=1, flags=re.M
    )
    text = re.sub(
        r"^writeInterval\s+\S+;",
        f"writeInterval   {end_time};",
        text,
        count=1,
        flags=re.M,
    )
    control_dict.write_text(text)
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


def _run_neon(case: Path) -> None:
    _run_solver(
        "from neofoam.solver.incompressibleFluidNeoN import run;"
        " run(['incompressibleFluidNeoN'])",
        case,
        "incompressibleFluidNeoN (steady)",
    )


def _run_reference(case: Path) -> None:
    _run_solver(
        "from neofoam.solver.incompressibleFluid import run;"
        " run(['incompressibleFluid'])",
        case,
        "incompressibleFluid (steady)",
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


def _assert_fields_match(
    result_case: Path,
    reference_case: Path,
    tmp_path: Path,
    label: str,
    fields: tuple[str, ...],
    rtol: float,
    atol_scale: float,
) -> None:
    result_final = _final_time_dir(result_case)
    reference_final = _final_time_dir(reference_case)
    assert result_final.name == reference_final.name, (
        f"solvers wrote different final times: "
        f"{result_final.name} vs {reference_final.name}"
    )

    result_vals = _load_internal_fields(
        result_case, fields, tmp_path / f"{label}_result"
    )
    reference_vals = _load_internal_fields(
        reference_case, fields, tmp_path / f"{label}_reference"
    )

    failures = []
    for name in fields:
        reference = reference_vals[name]
        result = result_vals[name]
        assert reference.shape == result.shape, (
            f"{name}: shape {reference.shape} vs {result.shape}"
        )
        peak = float(np.max(np.abs(reference))) or 1.0
        max_abs = float(np.max(np.abs(result - reference)))
        print(f"[{label}] {name}: max abs diff = {max_abs:.3e} (peak = {peak:.3e})")
        if not np.allclose(result, reference, rtol=rtol, atol=atol_scale * peak):
            failures.append(f"{name}(max abs={max_abs:.3e}, peak={peak:.3e})")

    assert not failures, (
        f"{label}: fields diverged beyond rtol={rtol:.0e}: " + ", ".join(failures)
    )


@pytest.mark.parametrize("model", list(MODELS), ids=list(MODELS))
def test_neon_steady_two_iterations_roundoff(model: str, tmp_path: Path) -> None:
    """Per-iteration SIMPLE parity is machine precision (sharp, fast check)."""
    fields, (rtol, atol_scale), _ = MODELS[model]
    neon_case = tmp_path / "neon"
    reference_case = tmp_path / "reference"
    _prepare_case(model, neon_case, end_time=2)
    _prepare_case(model, reference_case, end_time=2)

    _run_neon(neon_case)
    _run_reference(reference_case)

    _assert_fields_match(
        neon_case,
        reference_case,
        tmp_path,
        f"{model}_steady2",
        fields,
        rtol,
        atol_scale,
    )


@pytest.mark.parametrize("model", list(MODELS), ids=list(MODELS))
def test_neon_steady_converged_matches_incompressibleFluid(
    model: str, tmp_path: Path
) -> None:
    """The converged SIMPLE fixed points agree (slow: ~2 min per solver)."""
    fields, _, (rtol, atol_scale) = MODELS[model]
    neon_case = tmp_path / "neon"
    reference_case = tmp_path / "reference"
    _prepare_case(model, neon_case, end_time=500)
    _prepare_case(model, reference_case, end_time=500)

    _run_neon(neon_case)
    _run_reference(reference_case)

    _assert_fields_match(
        neon_case,
        reference_case,
        tmp_path,
        f"{model}_converged",
        fields,
        rtol,
        atol_scale,
    )


@pytest.mark.skipif(
    shutil.which("simpleFoam") is None, reason="native simpleFoam not on PATH"
)
@pytest.mark.parametrize("model", list(MODELS), ids=list(MODELS))
def test_reference_steady_matches_native_simpleFoam(model: str, tmp_path: Path) -> None:
    """The reference itself: incompressibleFluid (SIMPLE) is bitwise-parity
    with the native simpleFoam binary on this case."""
    fields = MODELS[model][0]
    reference_case = tmp_path / "reference"
    native_case = tmp_path / "native"
    _prepare_case(model, reference_case, end_time=50)
    _prepare_case(model, native_case, end_time=50)

    _run_reference(reference_case)
    result = subprocess.run(
        ["simpleFoam", "-case", str(native_case)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, f"simpleFoam failed: {result.stderr[-3000:]}"

    _assert_fields_match(
        reference_case, native_case, tmp_path, f"{model}_native", fields, 1e-10, 1e-15
    )
