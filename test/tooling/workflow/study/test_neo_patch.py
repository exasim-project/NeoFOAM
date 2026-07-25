# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""``neo_patch`` deviates the neofoam side of a comparison, and nothing else.

A study may declare ``neo_patch: {rel/dict/path: {key: value}}`` to inject a key
the neofoam solver needs but no upstream tutorial ships — the interIsoFoam
fallback injects ``advectionScheme isoAdvector`` into ``system/fvSolution``. Two
properties matter and are proven here without a sourced OpenFOAM: the key reaches
the neo side (via a real ``patch`` step applied to a real fvSolution fixture), and
the native side never sees it — protecting the pure drop-in for any study that
omits the key. ``stage``/``run_allrun`` are stubbed so the side asymmetry is
tested without running a solver.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from neofoam.io import DictFile
from neofoam.tooling.casebuild import CaseDir
from neofoam.tooling.workflow.study import runner
from neofoam.tooling.workflow.study.cases import Case, load_study
from neofoam.tooling.workflow.study.runner import _neo_patch

_CASE = Path(__file__).parent / "cases" / "vof_fvsolution"

_DISCOVER_PY = """
from pathlib import Path
from neofoam.tooling.workflow.study.cases import Case, case_id

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


def test_run_applies_neo_patch_to_neo_side_only(tmp_path: Path, monkeypatch: Any) -> None:
    """``_run`` passes the extra steps for the neo side and an empty tuple for native."""
    captured: dict[str, tuple[Any, ...]] = {}

    def fake_stage(*args: Any, extra: tuple[Any, ...] = (), **kwargs: Any) -> None:
        captured["neo" if kwargs.get("app") else "native"] = extra

    monkeypatch.setattr(runner, "stage", fake_stage)
    monkeypatch.setattr(runner, "run_allrun", lambda *a, **k: {"finished": True})

    study = _study(tmp_path, _CONFIG_PATCH)
    case: Case = study.cases[0]
    runner._run(study, case, case.neo_label, tmp_path / "work", tmp_path / "neo.json")
    runner._run(study, case, case.native_label, tmp_path / "work", tmp_path / "native.json")

    assert len(captured["neo"]) == 1
    assert captured["native"] == ()
