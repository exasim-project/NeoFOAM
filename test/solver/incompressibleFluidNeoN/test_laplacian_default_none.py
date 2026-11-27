# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""``laplacianSchemes { default none; }`` with only the declared keys, on the NeoN path.

NeoN's laplacian looks its scheme up by ``laplacian(<gamma.name>,<field.name>)``, with
no fall-through to ``default``. An unnamed surface interpolation is called
``interpolated_<source>`` (``interpolated_tmp`` for ``nut + nu``), so a run whose
diffusivity names drift from the declared ``laplacian(nuEff,U)`` / ``laplacian(DkEff,k)``
/ ... keys dies with ``Key not found``. A ``default`` entry hides that, and tutorials
carry one; wizard-written cases list only the declared keys. The case
(``cases/laplacianDefaultNone``) is ``cases/turbulentBox`` with such a
``laplacianSchemes``. Its discretisation is unchanged, and
``test_turbulence_vs_incompressibleFluid`` already covers the values, so this only
checks that each model runs to the end.

The structural check of the same names, without a runtime, is
``test/turbulence/test_diffusivity_names.py``. Each run is a subprocess because
NeoN/Kokkos and OpenFOAM keep per-process global state.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).parent
_BOX = _HERE / "cases" / "turbulentBox"
_DELTA = _HERE / "cases" / "laplacianDefaultNone"

MODELS = ["laminar", "kEpsilon", "kOmegaSST", "SpalartAllmaras"]

#: turbulentBox's endTime (20 steps of 5e-05).
FINAL_TIME = "0.001"


def _prepare_case(model: str, dest: Path) -> None:
    """Copy the box, lay the ``default none`` fvSchemes over it, select the model, mesh it."""
    shutil.copytree(_BOX, dest, ignore=shutil.ignore_patterns("models"))
    shutil.copytree(_DELTA, dest, dirs_exist_ok=True)
    shutil.copyfile(
        _BOX / "models" / model / "turbulenceProperties",
        dest / "constant" / "turbulenceProperties",
    )
    result = subprocess.run(
        ["blockMesh", "-case", str(dest)], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, f"blockMesh failed: {result.stderr}"


@pytest.mark.parametrize("model", MODELS)
def test_runs_with_only_declared_laplacian_keys(model: str, tmp_path: Path) -> None:
    case = tmp_path / model
    _prepare_case(model, case)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from neofoam.solver.incompressibleFluidNeoN import run;"
            " run(['incompressibleFluidNeoN'])",
        ],
        cwd=str(case),
        env={**os.environ, "FOAM_SIGFPE": "false"},
        capture_output=True,
        text=True,
        timeout=600,
    )

    assert result.returncode == 0, (
        f"laplacianDefaultNone ({model}) failed (rc={result.returncode}):\n"
        f"{result.stdout[-2000:]}\n{result.stderr[-3000:]}"
    )
    assert (case / FINAL_TIME / "U").is_file(), f"{model}: no final time {FINAL_TIME} written"
    assert "nan" not in (case / FINAL_TIME / "U").read_text(), f"{model}: U holds nan"
