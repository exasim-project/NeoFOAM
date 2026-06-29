# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""End-to-end turbulent (SA-DDES) run test for the NeoN-based ``neoPimpleFoam``.

Drives the full ``examples/neoPimpleFoam/neoPimpleFoam.cpp`` turbulence path
through the Python bindings: ``TurbulenceModel::create`` selects the LES
``SpalartAllmarasDDES`` model from ``constant/turbulenceProperties``, and each
time step solves the momentum + pressure system *and* the SA ``nuTilda``
transport PDE, then updates ``nut``.

This asserts the binding correctly runs the model end-to-end and that turbulence
actually develops (``nut`` grows from its seed). It deliberately does NOT assert
field-by-field parity against OpenFOAM: the underlying NeoFOAM SA-DDES is already
verified equal to OpenFOAM **per step to 1e-10** in ``test/spalartAllmarasDDES.cpp``,
and instantaneous field parity on a chaotic high-Re case is not a meaningful
metric (two independent solver stacks drift even for laminar flow — see the
``pitzDaily`` note in the port memory).
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

_TURB_PROPS = """\
FoamFile { version 2.0; format ascii; class dictionary; object turbulenceProperties; }
simulationType  LES;
LES
{
    LESModel        SpalartAllmarasDDES;
    turbulence      on;
    printCoeffs     on;
    delta           cubeRootVol;
    cubeRootVolCoeffs { deltaCoeff 1; }
}
"""

N_STEPS = 20
DELTA_T = 5e-5


def _insert_into_solvers(fv_solution: Path, block: str) -> None:
    """Insert ``block`` just before the closing brace of the ``solvers`` dict."""
    s = fv_solution.read_text()
    i = s.index("solvers")
    j = s.index("{", i)
    depth = 0
    k = j
    while k < len(s):
        if s[k] == "{":
            depth += 1
        elif s[k] == "}":
            depth -= 1
            if depth == 0:
                break
        k += 1
    fv_solution.write_text(s[:k] + block + s[k:])


def _prepare_sa_ddes_case(source: Path, dest: Path) -> None:
    """Turn the laminar neoPimpleFoam/pitzDaily tutorial into an SA-DDES case."""
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(source, dest)

    (dest / "constant" / "turbulenceProperties").write_text(_TURB_PROPS)

    # SA-DDES uses the U-based Spalding wall function for nut.
    nut = dest / "0" / "nut"
    nut.write_text(
        nut.read_text().replace("nutkWallFunction", "nutUSpaldingWallFunction")
    )

    # Seed nuTilda non-zero so turbulence develops (nuTilda = 0 -> nut stays 0).
    nutilda = dest / "0" / "nuTilda"
    txt = nutilda.read_text().replace(
        "internalField   uniform 0;", "internalField   uniform 4e-05;"
    )
    txt = re.sub(
        r"(inlet\s*\{[^}]*?value\s+uniform\s+)0(\s*;)",
        r"\g<1>4e-05\g<2>",
        txt,
        flags=re.S,
    )
    nutilda.write_text(txt)

    # nuTilda solver + momentum predictor / single outer corrector.
    fv = dest / "system" / "fvSolution"
    _insert_into_solvers(
        fv,
        "    nuTilda { solver PBiCGStab; preconditioner DILU; tolerance 1e-8; relTol 0; }\n"
        "    nuTildaFinal { $nuTilda; }\n",
    )
    fv.write_text(
        fv.read_text().replace(
            "PIMPLE\n{",
            "PIMPLE\n{\n    momentumPredictor yes;\n    nOuterCorrectors 1;",
        )
    )

    # NeoFOAM laplacian/snGrad are uncorrected; add the wallDist method SA needs.
    schemes = dest / "system" / "fvSchemes"
    s = schemes.read_text().replace(
        "Gauss linear corrected", "Gauss linear uncorrected"
    )
    s = s.replace("default         corrected;", "default         uncorrected;")
    schemes.write_text(s + "\nwallDist { method meshWave; }\n")

    # Fixed time step so the run is deterministic.
    cd = dest / "system" / "controlDict"
    lines = []
    for line in cd.read_text().splitlines():
        st = line.strip()
        if st.startswith("application"):
            lines.append("application pimpleFoam;")
        elif st.startswith("adjustTimeStep"):
            lines.append("adjustTimeStep no;")
        elif st.startswith("deltaT"):
            lines.append(f"deltaT {DELTA_T};")
        elif st.startswith("endTime"):
            lines.append(f"endTime {N_STEPS * DELTA_T};")
        elif st.startswith("writeControl"):
            lines.append("writeControl timeStep;")
        elif st.startswith("writeInterval"):
            lines.append(f"writeInterval {N_STEPS};")
        else:
            lines.append(line)
    if not any(line.strip().startswith("adjustTimeStep") for line in lines):
        lines.append("adjustTimeStep no;")
    cd.write_text("\n".join(lines) + "\n")

    result = subprocess.run(
        ["blockMesh", "-case", str(dest)], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, f"blockMesh failed: {result.stderr}"


def _read_internal(time_dir: Path, field_name: str) -> np.ndarray:
    """Parse an OpenFOAM scalar field's internalField (uniform or nonuniform)."""
    txt = (time_dir / field_name).read_text()
    m = re.search(r"internalField\s+nonuniform[^(]*\(([^)]*)\)", txt, re.S)
    if m:
        return np.array([float(x) for x in m.group(1).split()])
    m = re.search(r"internalField\s+uniform\s+([-0-9.eE+]+)", txt)
    assert m, f"could not parse internalField of {field_name}"
    return np.array([float(m.group(1))])


@requires_openfoam
def test_neoPimpleFoam_SA_DDES_runs_and_develops_turbulence(tmp_path: Path) -> None:
    """The SA-DDES turbulence path runs end-to-end and nut grows from its seed."""
    repo_root = Path(__file__).parent.parent.parent
    source = repo_root / "tutorials" / "neoPimpleFoam" / "pitzDaily"
    case = tmp_path / "sa_ddes"
    _prepare_sa_ddes_case(source, case)

    # Run in an isolated subprocess: NeoN/Kokkos + OpenFOAM hold per-process global
    # state that does not survive a second in-process solver run. FOAM_SIGFPE is
    # disabled so the signal handler does not abort the wild early SA transient.
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
        f"neoPimpleFoam (SA-DDES) failed (rc={result.returncode}):\n{result.stderr[-3000:]}"
    )

    final = case / str(N_STEPS * DELTA_T)
    assert final.is_dir(), f"solver did not write the final time {final}"

    nut = _read_internal(final, "nut")
    nutilda = _read_internal(final, "nuTilda")

    # Turbulence developed: nut grew from its uniform-0 start to a finite field.
    assert np.all(np.isfinite(nut)), "nut contains non-finite values"
    assert np.all(np.isfinite(nutilda)), "nuTilda contains non-finite values"
    assert float(np.max(nut)) > 1e-7, (
        f"nut did not develop (max nut = {float(np.max(nut)):.3e}); "
        "the SA-DDES model produced no turbulent viscosity"
    )
    print(
        f"SA-DDES developed: max nut = {float(np.max(nut)):.3e}, "
        f"max nuTilda = {float(np.max(nutilda)):.3e} (case {case.name})"
    )
