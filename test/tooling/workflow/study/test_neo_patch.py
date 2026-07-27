# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""``neo_patch`` deviates the neofoam side of a comparison, and nothing else.

A study may declare ``neo_patch: {rel/dict/path: {key: value}}`` to inject a key
the neofoam solver needs but no upstream tutorial ships — the interIsoFoam
fallback injects ``advectionScheme isoAdvector`` into ``system/fvSolution``. Two
properties matter and are proven here without a sourced OpenFOAM: the key reaches
the neo side (via a real ``patch`` step applied to a real fvSolution fixture), and
the native side never sees it — protecting the pure drop-in for any study that
omits the key.

The side asymmetry is a property of ``_swap``, which is where the candidate
deviation is applied: ``_build`` stages every solver from the same recipe, and
``_swap`` then patches and swaps the candidate only. Asserting on the real
``fvSolution`` each side ends up with is stronger than asserting on the arguments
a stubbed ``stage`` was handed.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from neofoam.io import DictFile
from neofoam.tooling.casebuild import CaseDir
from verification.dropin import runner
from verification.dropin.cases import Case, load_study
from verification.dropin.runner import _neo_patch

_CASE = Path(__file__).parent / "cases" / "vof_fvsolution"

#: A minimal Allrun naming the case's native solver, so ``_swap`` finds a real swap
#: point (an Allrun it cannot swap raises ``NoSwapPoint``, which would neutralise the
#: dir and make the assertions below pass for the wrong reason).
_ALLRUN = """#!/bin/sh
cd "${0%/*}" || exit
. ${WM_PROJECT_DIR:?}/bin/tools/RunFunctions

runApplication blockMesh
runApplication interIsoFoam
"""

_DISCOVER_PY = """
from pathlib import Path
from verification.dropin.cases import Case, case_id

def discover():
    name = "interIsoFoam/weirOverflow"
    return [Case(
        id=case_id(name), name=name, path=Path("/tut") / name,
        native_solver="interIsoFoam", app="neofoam solver incompressiblevof",
        fields=("alpha.water", "U", "p_rgh", "p"), turbulence="laminar", tier="A",
    )]
"""

_CONFIG = "title: vof\ndiscover: discover.py\n"
_CONFIG_PATCH = _CONFIG + "neo_patch:\n  system/fvSolution:\n    advectionScheme: isoAdvector\n"


def _study(tmp_path: Path, config: str) -> Any:
    (tmp_path / "discover.py").write_text(_DISCOVER_PY)
    (tmp_path / "config.yaml").write_text(config)
    return load_study(tmp_path / "config.yaml")


def _build_and_swap(
    study: Any, case: Case, label: str, cases_root: Path, tmp_path: Path
) -> dict[str, Any]:
    """Stand in for ``_build`` (identical for every solver), then run the real ``_swap``.

    ``_build`` only stages the tutorial with casebuild, which needs OpenFOAM; the
    fixture case + a ``.built.json`` stamp is the same starting state without it.
    Returns the swap stamp.
    """
    run_dir = cases_root / case.id / label
    shutil.copytree(_CASE, run_dir)
    (run_dir / "Allrun").write_text(_ALLRUN)
    built = tmp_path / f"{label}.built.json"
    built.write_text(json.dumps({"solver": label}))
    stamp = tmp_path / f"{label}.swapped.json"
    runner._swap(study, case, label, cases_root, built, stamp)
    result: dict[str, Any] = json.loads(stamp.read_text())
    return result


def _advection_scheme(cases_root: Path, case: Case, label: str) -> str | None:
    fv_solution = DictFile(cases_root / case.id / label / "system" / "fvSolution")
    if not fv_solution.found("advectionScheme"):
        return None
    return fv_solution.get[str]("advectionScheme")


def test_neo_patch_absent_yields_no_extra_steps(tmp_path: Path) -> None:
    """A study without ``neo_patch`` stages a pure drop-in — no extra steps."""
    assert _neo_patch(_study(tmp_path, _CONFIG)) == ()


def test_neo_patch_step_injects_the_key_into_fvsolution(tmp_path: Path) -> None:
    """The declared step really sets ``advectionScheme`` in a real fvSolution."""
    steps = _neo_patch(_study(tmp_path, _CONFIG_PATCH))
    case = tmp_path / "case"
    shutil.copytree(_CASE, case)

    assert len(steps) == 1
    steps[0](CaseDir(case))

    assert DictFile(case / "system" / "fvSolution").get[str]("advectionScheme") == "isoAdvector"


def test_swap_applies_neo_patch_to_the_candidate_side_only(tmp_path: Path) -> None:
    """The patched key lands in the candidate's fvSolution and in no other."""
    study = _study(tmp_path, _CONFIG_PATCH)
    case: Case = study.cases[0]
    cases_root = tmp_path / "cases"

    for label in (case.neo_label, case.native_label):
        stamp = _build_and_swap(study, case, label, cases_root, tmp_path)
        # The swap must have succeeded: a refused Allrun is neutralised instead,
        # which would leave the patch applied but prove nothing about the split.
        assert not stamp.get("no_swap"), stamp

    assert _advection_scheme(cases_root, case, case.neo_label) == "isoAdvector"
    assert _advection_scheme(cases_root, case, case.native_label) is None


def test_swap_leaves_the_native_allrun_pristine_and_swaps_the_candidate(
    tmp_path: Path,
) -> None:
    """Native is the reference, so only the candidate's Allrun names a neofoam solver."""
    study = _study(tmp_path, _CONFIG_PATCH)
    case: Case = study.cases[0]
    cases_root = tmp_path / "cases"

    for label in (case.neo_label, case.native_label):
        _build_and_swap(study, case, label, cases_root, tmp_path)

    def allrun(label: str) -> str:
        return (cases_root / case.id / label / "Allrun").read_text()

    assert allrun(case.native_label) == _ALLRUN
    assert "neofoam solver incompressiblevof" in allrun(case.neo_label)
    assert "runApplication interIsoFoam" not in allrun(case.neo_label)
