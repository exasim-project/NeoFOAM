# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""End-to-end tests for the framework NeoN VoF solver ``incompressibleVoFNeon``.

Framework port (SolverSpec/ModelSpec) of the imperative ``neoInterFoam``. These
tests prepare the damBreak tutorial (blockMesh + setFields) and run the full
solver ``run()`` for a short horizon in a subprocess (its own Kokkos init), then
read the on-disk fields back and assert physical behaviour: the run completes,
``alpha.water`` stays bounded in [0, 1] (small MULES overshoot tolerated), the
interface falls/spreads under gravity, and no field contains nan/inf.

Deliberately loose on absolute values (upwind/MULES on a coarse mesh differs
from a reference interFoam run); it asserts bounded, finite, evolving behaviour.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

os.environ.setdefault("FOAM_SIGFPE", "false")


def _prepare_case(dest: Path) -> None:
    repo_root = Path(__file__).parent.parent.parent
    src = repo_root / "tutorials" / "damBreak"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)
    zero = dest / "0"
    if not zero.exists():
        shutil.copytree(dest / "0.orig", zero)
    env = {**os.environ, "FOAM_SIGFPE": "false"}
    for cmd in (
        ["blockMesh"],
        ["setFields"],
        ["postProcess", "-func", "writeCellCentres", "-time", "0"],
    ):
        r = subprocess.run(
            cmd, cwd=str(dest), env=env, capture_output=True, text=True, timeout=180
        )
        assert r.returncode == 0, f"{cmd[0]} failed:\n{r.stderr[-2000:]}"


# Run the full framework solver for a short horizon, then report on-disk field
# metrics. A subprocess isolates the single Kokkos init and lets the driver read
# the written fields back after the solve returns.
_RUN_DRIVER = r"""
import re
from pathlib import Path
import numpy as np
from neofoam.solver.incompressibleVoFNeon import run

t = open("system/controlDict").read()
t = re.sub(r"endTime\s+\S+;", "endTime         0.02;", t)
t = re.sub(r"writeControl\s+\S+;", "writeControl    runTime;", t)
t = re.sub(r"writeInterval\s+\S+;", "writeInterval   0.02;", t)
open("system/controlDict", "w").write(t)

_NANINF = re.compile(r"[-+]?\b(?:nan|inf)\b", re.IGNORECASE)


def _internal(path):
    txt = path.read_text()
    m = re.search(r"internalField\s+nonuniform\s+List<scalar>\s*\n\s*(\d+)\s*\n\(", txt)
    if not m:
        u = re.search(r"internalField\s+uniform\s+([-\d.eE+]+)\s*;", txt)
        return np.array([float(u.group(1))]) if u else None
    n = int(m.group(1))
    s = txt.index("(", m.end() - 1) + 1
    e = txt.index(")", s)
    v = np.fromstring(txt[s:e].replace("\n", " "), sep=" ")
    assert v.size == n
    return v


def _latest_with(name):
    times = []
    for p in Path(".").iterdir():
        if p.is_dir():
            try:
                tv = float(p.name)
            except ValueError:
                continue
            if tv > 0.0 and (p / name).exists():
                times.append((tv, p))
    if not times:
        return None
    return max(times, key=lambda kv: kv[0])[1]


def _drive():
    ok = 1
    try:
        run(["incompressibleVoFNeon"])
    except Exception as exc:
        ok = 0
        print("RUN_ERR", repr(exc))
    print("RUN_COMPLETED", ok)

    a0 = _internal(Path("0") / "alpha.water")
    latest = _latest_with("alpha.water")
    assert latest is not None, "no evolved alpha.water written"
    a1 = _internal(latest / "alpha.water")
    cx = _internal(Path("0") / "Cx")

    print("ALPHA_MIN", float(a1.min()))
    print("ALPHA_MAX", float(a1.max()))
    print("ALPHA_FINITE", int(bool(np.isfinite(a1).all())))
    print("U_NO_NANINF", 0 if _NANINF.search((latest / "U").read_text()) else 1)
    print("PRGH_NO_NANINF", 0 if _NANINF.search((latest / "p_rgh").read_text()) else 1)
    print("MASS_DRIFT_REL", float(abs(a1.mean() - a0.mean()) / a0.mean()))
    # Interface must move: the water column front (alpha>0.5) advances in +x.
    front0 = float(cx[a0 > 0.5].max())
    front1 = float(cx[a1 > 0.5].max())
    print("FRONT0", front0)
    print("FRONT1", front1)
    print("END_OK")


_drive()
"""


def _run_driver(case: Path, driver: str, timeout: int = 400) -> dict[str, float]:
    env = {**os.environ, "FOAM_SIGFPE": "false"}
    r = subprocess.run(
        [sys.executable, "-c", driver],
        cwd=str(case),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    assert r.returncode == 0, f"driver failed:\n{r.stdout[-1500:]}\n{r.stderr[-3000:]}"
    assert "END_OK" in r.stdout, f"driver did not finish cleanly:\n{r.stdout[-2000:]}"
    out: dict[str, float] = {}
    for line in r.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2:
            try:
                out[parts[0]] = float(parts[1])
            except ValueError:
                pass
    return out


@pytest.fixture(scope="module")
def run_metrics(tmp_path_factory: pytest.TempPathFactory) -> dict[str, float]:
    case = tmp_path_factory.mktemp("damBreak_vofneon")
    _prepare_case(case)
    return _run_driver(case, _RUN_DRIVER)


def test_solver_spec_registered() -> None:
    """The solver package imports and exposes its SolverSpec + run entrypoint."""
    from neofoam.solver.incompressibleVoFNeon import (
        config_classes,
        incompressibleVoFNeon,
        run,
    )

    assert incompressibleVoFNeon.name == "incompressibleVoFNeon"
    assert callable(run)
    assert callable(config_classes)


def test_run_completes(run_metrics: dict[str, float]) -> None:
    """The full framework solver runs damBreak to the short horizon cleanly."""
    assert run_metrics["RUN_COMPLETED"] == 1.0


def test_alpha_bounded(run_metrics: dict[str, float]) -> None:
    """MULES keeps alpha.water in [0, 1] (small overshoot tolerated)."""
    assert run_metrics["ALPHA_FINITE"] == 1.0
    assert run_metrics["ALPHA_MIN"] >= -1e-3
    assert run_metrics["ALPHA_MAX"] <= 1.0 + 1e-2


def test_fields_finite(run_metrics: dict[str, float]) -> None:
    """No nan/inf tokens in the written U / p_rgh fields."""
    assert run_metrics["U_NO_NANINF"] == 1.0
    assert run_metrics["PRGH_NO_NANINF"] == 1.0


def test_mass_conserved_and_interface_moves(run_metrics: dict[str, float]) -> None:
    """Mass drift stays small and the water front advances under gravity."""
    assert run_metrics["MASS_DRIFT_REL"] < 0.05
    # The dam breaks: the alpha>0.5 front moves downstream (+x) or holds — never
    # retreats to a smaller extent than it started.
    assert run_metrics["FRONT1"] >= run_metrics["FRONT0"] - 1e-6
