# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Control-block selection during ``incompressibleFluidNeoN`` initialization.

The NeoN pressure-velocity build reads its corrector counts and pressure
reference from a control subdict of ``system/fvSolution``. A stock pimpleFoam
case ships a ``PIMPLE`` block; a pisoFoam case ships a ``PISO`` block instead.
Two init steps in ``pimpleAlgorithm.build`` must therefore pick the block that
is present:

* ``create_pimple_state`` — ``PIMPLE`` if present, else ``PISO``, else a clear
  ``ValueError`` (the old code hardcoded ``subDict("PIMPLE")`` and raised
  ``Key 'PIMPLE' not found in Dictionary`` on a pisoFoam case);
* ``create_pressure_reference`` — passes ``"PIMPLE"`` or ``"PISO"`` to
  ``set_ref_cell`` (the old code hardcoded ``"PIMPLE"`` and aborted a pisoFoam
  case with a FOAM fatal ``Entry 'PIMPLE' not found``, *after*
  ``create_pimple_state`` — so a pisoFoam case initializing proves BOTH
  branches).

The PIMPLE (no-regression) path is already covered end-to-end by
``test/solver/incompressibleFluidNeoN/test_cavity_run.py`` (a laminar PIMPLE
cavity whose solve necessarily runs both init steps), so it is not duplicated
here.

A sibling behavior is pinned here too: ``simpleAlgorithm.create_simple_state``
accepts a SIMPLEC case (``consistent yes``) — it used to refuse it with a
``NotImplementedError``, so the test asserts that refusal is gone. That the
consistent branch is numerically right (not merely accepted) is a separate,
heavier check: ``test_steady_vs_incompressibleFluid`` compares a SIMPLEC run
against the pybFoam backend field-by-field.

Each case is a laminar lid-driven cavity (no turbulence/wallDist confound), so
the init failure/success is attributable to the control-block selection alone.
Initialization runs in an isolated subprocess: NeoN/Kokkos + OpenFOAM keep
per-process global state that a second in-process ``Foam::Time`` would corrupt,
and a FOAM fatal error calls ``::exit()`` — a subprocess turns that into a
return code the assertions can inspect instead of tearing down pytest. The
whole StagedInitRunner executes its lazy steps eagerly, so ``initialize()``
alone (no time-stepping) reaches both init steps under test.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from neofoam.tooling.casebuild import block_mesh, from_template

_CASES = Path(__file__).parent / "cases"

# Drive initialization only (not a full solve): the StagedInitRunner executes
# all lazy init steps eagerly, so this reaches create_pimple_state /
# create_pressure_reference (or create_simple_state) and then stops.
_INIT_DRIVER = (
    "from neofoam.solver.incompressibleFluidNeoN import incompressibleFluidNeoN;"
    " from neofoam.solver.neon_runtime import ensure_neon_initialized;"
    " ensure_neon_initialized(['incompressibleFluidNeoN']);"
    " solver = incompressibleFluidNeoN.instantiate(argv=['incompressibleFluidNeoN']);"
    " solver.initialize();"
    " print('NEON_INIT_OK')"
)


def _run_init(case_dir: Path, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    """Stage the committed case (blockMesh) and run the init driver in a subprocess."""
    case = (from_template(case_dir) | block_mesh()).build_at(tmp_path / "case")
    env = {**os.environ, "FOAM_SIGFPE": "false"}
    return subprocess.run(
        [sys.executable, "-c", _INIT_DRIVER],
        cwd=str(case.path),
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )


def test_piso_case_initializes(tmp_path: Path) -> None:
    """A pisoFoam-style case (PISO block, no PIMPLE) initializes cleanly.

    Proves both fixed branches: create_pimple_state selects the PISO block
    (old code raised ``Key 'PIMPLE' not found``) and create_pressure_reference
    passes ``"PISO"`` to set_ref_cell (old code aborted with a FOAM fatal
    ``Entry 'PIMPLE' not found``).
    """
    result = _run_init(_CASES / "piso_cavity", tmp_path)

    combined = result.stdout + result.stderr
    assert result.returncode == 0, (
        "incompressibleFluidNeoN init failed on a PISO case "
        f"(rc={result.returncode}):\n{result.stdout[-2000:]}\n{result.stderr[-3000:]}"
    )
    assert "NEON_INIT_OK" in result.stdout, "init did not run to completion"
    # The exact failure signatures the fix removes — must not appear.
    assert "Key 'PIMPLE' not found" not in combined
    assert "Entry 'PIMPLE' not found" not in combined


def test_simplec_case_initializes(tmp_path: Path) -> None:
    """A SIMPLEC case (``consistent yes``) initializes instead of being refused.

    ``create_simple_state`` used to raise ``NotImplementedError``; it now reads
    ``consistent`` as a control flag, so the case must reach initialization.
    """
    result = _run_init(_CASES / "simplec_cavity", tmp_path)

    assert result.returncode == 0, (
        "incompressibleFluidNeoN init failed on a SIMPLEC case "
        f"(rc={result.returncode}):\n{result.stdout[-2000:]}\n{result.stderr[-3000:]}"
    )
    assert "NEON_INIT_OK" in result.stdout, "init did not run to completion"
    assert "NotImplementedError" not in result.stderr, result.stderr[-3000:]
