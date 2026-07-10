# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Solution-level comparison: incompressibleVoFNeon vs incompressibleVoF.

Both VoF variants run the identical damBreak case (fixed dt, so the step
sequences match and trajectory divergence is excluded) and the written fields
at the end time are compared. The pybFoam variant is bitwise-verified against
native interFoam elsewhere (``test/solver/incompressibleVoF/``), so it is the
reference here; the NeoN variant is run-to-run deterministic.

Current measured agreement at t=0.01 (10 fixed steps, 2268 cells):
alpha relL2 ~8e-4, p_rgh relL2 ~2e-3, U relL2 ~8e-2. The U gap is NOT noise:
it is concentrated in a few near-interface air cells (bulk mean diff ~1e-3
m/s) and points at remaining interface-scheme fidelity work (ddtCorr /
compression-flux details). The tolerances below lock in the current agreement
with margin so regressions fail loudly — tighten them as the interface-cell
gap is closed; the end goal is linear-solver-tolerance-level agreement.
"""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from incompressibleFluid.comparison_helpers import setup_case

os.environ.setdefault("FOAM_SIGFPE", "false")

_END_TIME = 0.01
_TIME_NAME = "0.01"

# relL2 ceilings = current measured agreement x margin (see module docstring).
_TOLERANCES = {
    "alpha.water": 5e-3,  # measured ~8e-4
    "p_rgh": 2e-2,  # measured ~2e-3
    "U": 2.5e-1,  # measured ~8e-2 (near-interface air cells dominate)
}


def _parse_scalar(path: Path) -> np.ndarray:
    txt = path.read_text()
    m = re.search(r"internalField\s+nonuniform\s+List<scalar>\s*\n\s*(\d+)\s*\n\(", txt)
    assert m, f"no scalar internalField in {path}"
    s = txt.index("(", m.end() - 1) + 1
    e = txt.index(")", s)
    v = np.fromstring(txt[s:e].replace("\n", " "), sep=" ")
    assert v.size == int(m.group(1))
    return v


def _parse_vector(path: Path) -> np.ndarray:
    txt = path.read_text()
    m = re.search(r"internalField\s+nonuniform\s+List<vector>\s*\n\s*(\d+)\s*\n\(", txt)
    assert m, f"no vector internalField in {path}"
    n = int(m.group(1))
    triplets = re.findall(r"\(([^()]+)\)", txt[m.end() :])[:n]
    v = np.array([[float(x) for x in t.split()] for t in triplets])
    assert v.shape == (n, 3)
    return v


def _rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-300))


def _fix_time_step(case: Path) -> None:
    control = case / "system" / "controlDict"
    control.write_text(
        re.sub(r"adjustTimeStep\s+\S+;", "adjustTimeStep  no;", control.read_text())
    )


def _run_solver(case: Path, driver: str) -> None:
    env = {**os.environ, "FOAM_SIGFPE": "false"}
    r = subprocess.run(
        [sys.executable, "-c", driver],
        cwd=str(case),
        env=env,
        capture_output=True,
        text=True,
        timeout=400,
    )
    assert r.returncode == 0, f"solver failed:\n{r.stdout[-800:]}\n{r.stderr[-2000:]}"


@pytest.fixture(scope="module")
def solution_fields(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """(neon, pybfoam) internal fields at the end time, per compared field."""
    repo_root = Path(__file__).parent.parent.parent
    source_case = repo_root / "tutorials" / "damBreak"

    cases: dict[str, Path] = {}
    for tag in ("pybfoam", "neon"):
        case = tmp_path_factory.mktemp(f"damBreak_cmp_{tag}")
        shutil.rmtree(case)
        setup_case(source_case, case, _END_TIME, _END_TIME, run_setfields=True)
        _fix_time_step(case)
        cases[tag] = case

    _run_solver(
        cases["pybfoam"],
        "from neofoam.solver.incompressibleVoF import run\nrun(['incompressibleVoF'])",
    )
    _run_solver(
        cases["neon"],
        "from neofoam.solver.incompressibleVoFNeon import run\nrun(['incompressibleVoFNeon'])",
    )

    fields: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, parse in (
        ("alpha.water", _parse_scalar),
        ("p_rgh", _parse_scalar),
        ("U", _parse_vector),
    ):
        fields[name] = (
            parse(cases["neon"] / _TIME_NAME / name),
            parse(cases["pybfoam"] / _TIME_NAME / name),
        )
    return fields


@pytest.mark.parametrize("field_name", ["alpha.water", "p_rgh", "U"])
def test_field_agreement(
    solution_fields: dict[str, tuple[np.ndarray, np.ndarray]], field_name: str
) -> None:
    """Each written field agrees with the pybFoam reference within tolerance."""
    neon, pybfoam = solution_fields[field_name]
    rel = _rel_l2(neon, pybfoam)
    assert rel < _TOLERANCES[field_name], (
        f"{field_name}: relL2 {rel:.3e} exceeds {_TOLERANCES[field_name]:.1e} "
        "(regression vs the locked-in NeoN/pybFoam agreement)"
    )


def test_interface_position_agrees(
    solution_fields: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    """The alpha>0.5 interface indicator matches in all but a few cells."""
    neon, pybfoam = solution_fields["alpha.water"]
    mismatched = int(((neon > 0.5) != (pybfoam > 0.5)).sum())
    assert mismatched <= 5, (
        f"interface indicator differs in {mismatched}/{neon.size} cells"
    )
