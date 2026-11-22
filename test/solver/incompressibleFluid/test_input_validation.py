# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Input validation contract test for incompressibleFluid.

Runs the solver on the ``val_pitzDaily`` case (intentionally incomplete
fvSolution / turbulenceProperties). If the solver crashes, the validator
should have predicted the same missing entries upfront. Ported from
``feat/python_solvers``; the only API adaptation is that validation is
now reached via ``runner.run_load().validate()`` rather than a direct
``.validate()`` on the staged-init runner.

The source-branch test is documented as a "currently FAILS" contract
test exposing gaps in the validator — porting it preserves that intent.
"""

import os
import shutil
import subprocess
from pathlib import Path
from typing import Generator

import pytest

from neofoam.solver.incompressibleFluid.create_fields import create_init

CASE_DIR = Path(__file__).parent / "val_pitzDaily"


@pytest.fixture
def case(tmp_path: Path) -> Generator[Path, None, None]:
    """Copy val_pitzDaily to tmp, restore 0/, run blockMesh, chdir."""
    dst = tmp_path / "val_pitzDaily"
    shutil.copytree(CASE_DIR, dst)
    shutil.copytree(dst / "0.orig", dst / "0")
    original = os.getcwd()
    os.chdir(dst)
    subprocess.run(["blockMesh"], capture_output=True, timeout=30)
    yield dst
    os.chdir(original)


def test_solver_crashes_validator_should_predict(case: Path) -> None:
    """Contract: if the solver crashes on val_pitzDaily, the validator
    should have predicted it via ``run_load().validate()``.

    The LOAD stage now collects the solver-core configs (controlDict,
    transportProperties), so those are validated. The remaining gap is the
    PIMPLE ``fvSolution`` slice (here the ``U`` solver is hidden behind a
    regex key): validating it requires loading the slice, which the
    OpenFOAM reader cannot yet parse (typed scheme values / ``dict`` solver
    entries). Until then ``validate()`` cannot predict a crash that stems
    only from that slice.
    """
    os.environ["FOAM_SIGFPE"] = ""

    load_result = create_init(case_dir=case).run_load()
    errors = load_result.validate()

    log_file = case / "solver_output.log"
    with log_file.open("w") as f:
        result = subprocess.run(
            ["neofoam", "solver", "incompressiblefluid"],
            cwd=case,
            stdout=f,
            stderr=subprocess.STDOUT,
            timeout=60,
            env={**os.environ, "FOAM_SIGFPE": ""},
        )
    solver_crashed = result.returncode != 0

    persistent_log = Path(__file__).parent / "solver_output.log"
    shutil.copy(log_file, persistent_log)

    print(f"\n=== Validator found {len(errors)} errors ===")
    for e in errors:
        print(f"  {e.field}: {e.message}")

    print(f"\n=== Solver crashed: {solver_crashed} ===")

    if solver_crashed:
        assert len(errors) > 0, (
            f"Solver crashed (see {persistent_log}).\n"
            f"But validator found NO errors — it should have predicted this crash."
        )


def test_load_result_exposes_solver_configs(case: Path) -> None:
    """LOAD surfaces the solver's config instances and declared classes.

    ``LoadResult.configs`` is no longer empty for incompressibleFluid: the
    solver-core configs and the PIMPLE fvSchemes/fvSolution slices are
    collected, and ``config_classes`` lists the declared schema set (the
    basis for scaffolding a case from configs alone).
    """
    load_result = create_init(case_dir=case).run_load()

    class_names = {c.__name__ for c in load_result.config_classes}
    assert "ControlDictConfig" in class_names
    assert "TransportPropertiesConfig" in class_names
    assert any("fvSchemes" in n for n in class_names)
    assert any("fvSolution" in n for n in class_names)

    assert load_result.configs, "expected at least the solver-core configs"
