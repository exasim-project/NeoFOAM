# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Parity test: the turbulence ModelSpec on NeoN, validated against pybFoam.

Verifies that the pure-Python ModelSpec model built on the **NeoN** backend
produces the same eddy viscosity ``nut`` — the universal turbulence output (``0``
for ``laminar``, ``Cmu k^2/eps`` evolved by the transport PDEs for ``kEpsilon``) —
as the trusted **pybFoam** OpenFOAM model, both **after one ``correct`` step**.

The step advances the k / epsilon transport equations one ``deltaT`` over a small
uniform box with **no walls** (hence no wall functions, whose near-wall cell
overrides the pure-Python model does not replicate) and **random** initial ``k``
and ``epsilon`` fields, so the convection / diffusion / production / dissipation
terms are all genuinely exercised. Because both backends discretise the *same*
matrix and solve it to a tight linear tolerance, the two ``nut`` fields agree to
the linear-solver tolerance rather than merely qualitatively.

Every ``Foam::Time`` lives in its own subprocess (see :mod:`_parity_worker`):
mesh + field seeding, the pybFoam reference, and the NeoN subject each run in a
fresh process, because several ``Foam::Time`` objects in one process corrupt the
shared OpenFOAM registry and segfault. This module only orchestrates those
subprocesses and compares the ``.npy`` arrays they emit.

Adding a model is one entry in :data:`CASES`; the body is generic.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).parent
_WORKER = _HERE / "_parity_worker.py"
_BASE_CASE = _HERE / "parity_base"  # shared no-wall box case (mesh + fields + system)
_MODELS = _HERE / "parity_models"  # per-model constant/turbulenceProperties overlay

#: Add a NeoN model by dropping its ``turbulenceProperties`` under
#: ``parity_models/<name>/`` and listing it here.
CASES = ["laminar", "kEpsilon", "SpalartAllmaras", "kOmegaSST"]

#: Models exercised on the pybFoam **fallback** path (``select(fallback=True)``).
#: Every dual model plus the fallback-only ``realizableKE`` (which has no native
#: subject — only the fallback op).
FALLBACK_CASES = CASES + ["realizableKE"]


def _run_worker(role: str, case: Path) -> None:
    """Run one worker role in a fresh process (one ``Foam::Time`` per process)."""
    subprocess.run(
        [sys.executable, str(_WORKER), role, str(case)],
        check=True,
        cwd=str(case.parent),
    )


@pytest.mark.parametrize("name", CASES)
def test_neon_nut_matches_pybfoam(name: str, tmp_path: Path) -> None:
    """After one ``correct`` step, the NeoN model's ``nut`` equals pybFoam's, cell-by-cell."""
    case = tmp_path / "case"
    shutil.copytree(_BASE_CASE, case)
    shutil.copyfile(
        _MODELS / name / "turbulenceProperties",
        case / "constant" / "turbulenceProperties",
    )

    _run_worker("setup", case)  # mesh + identical random U / k / epsilon on disk
    _run_worker("reference", case)  # pybFoam fields → reference_<field>.npy
    _run_worker("subject", case)  # NeoN fields → subject_<field>.npy

    # Every field the closure owns is checked, not just the derived viscosity:
    # ``nut`` universally, plus the transport unknowns ``k`` / ``epsilon`` when the
    # model solves them (a closure must publish the same set on both backends).
    compared = []
    for field in ("nut", "k", "epsilon", "nuTilda", "omega"):
        ref_path = case / f"reference_{field}.npy"
        sub_path = case / f"subject_{field}.npy"
        assert ref_path.exists() == sub_path.exists(), (
            f"{name}: only one backend produced {field!r} "
            f"(reference={ref_path.exists()}, subject={sub_path.exists()})"
        )
        if not ref_path.exists():
            continue
        reference = np.load(ref_path)
        result = np.load(sub_path)

        peak = float(np.max(np.abs(reference))) or 1.0
        max_abs = float(np.max(np.abs(result - reference)))
        print(f"[{name}] {field}: max abs diff = {max_abs:.3e} (peak = {peak:.3e})")
        np.testing.assert_allclose(
            result,
            reference,
            rtol=1e-8,
            atol=1e-10 * peak,
            err_msg=f"{name}: {field} differs on NeoN vs pybFoam after one correct step",
        )
        compared.append(field)

    assert "nut" in compared, f"{name}: nut was not compared"


@pytest.mark.parametrize("name", FALLBACK_CASES)
def test_fallback_nut_matches_pybfoam(name: str, tmp_path: Path) -> None:
    """The ``fallback=True`` path advances ``nut`` via the model's fallback op.

    ``select_turbulence_model(fallback=True)`` returns a ``FallbackHandle`` whose
    scheduled op wraps the pybFoam model — so its ``nut`` equals the pybFoam
    reference. This proves the FallbackHandle + op-dispatch wiring (the path
    ``incompressibleFluid`` uses), including for the fallback-only ``realizableKE``,
    which has no native NeoN closure.
    """
    case = tmp_path / "case"
    shutil.copytree(_BASE_CASE, case)
    shutil.copyfile(
        _MODELS / name / "turbulenceProperties",
        case / "constant" / "turbulenceProperties",
    )

    _run_worker("setup", case)  # mesh + identical random U / k / epsilon on disk
    _run_worker("reference", case)  # pybFoam fields → reference_nut.npy
    _run_worker("subject_fb", case)  # fallback handle + op → subject_fb_nut.npy

    reference = np.load(case / "reference_nut.npy")
    result = np.load(case / "subject_fb_nut.npy")
    peak = float(np.max(np.abs(reference))) or 1.0
    np.testing.assert_allclose(
        result,
        reference,
        rtol=1e-8,
        atol=1e-10 * peak,
        err_msg=f"{name}: fallback nut differs from the pybFoam reference",
    )
