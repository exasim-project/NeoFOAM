# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""``swap_solver`` redirects a tutorial's ``Allrun`` to the neofoam command.

Pure text manipulation — no OpenFOAM — so the swap idioms are pinned against real
``Allrun`` fixtures on disk (``cases/allrun_*``), copied into ``tmp_path`` so the
checked-in fixtures are never mutated. The idioms mirror the multiphase tutorials
that the incompressibleVoF sweep must stage:

* ``$(getApplication)`` named twice (a commented serial variant + the real parallel
  run — ``interFoam/laminar/oscillatingBox``, ``interFoam/RAS/motorBike``): every
  identical occurrence is swapped, so both invocations run the candidate.
* ``application="<solver>"`` + ``${application}``: the assignment's *value* is swapped,
  not the bare token (which would nest quotes).
* a single ``$(getApplication)``: still swapped (the pre-existing behaviour).
* no recognizable solver invocation: refused with :class:`NoSwapPoint`.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from neofoam.tooling.casebuild import CaseDir
from verification.dropin.stage import NoSwapPoint, swap_solver

_APP = "neofoam solver incompressiblevof"
_CASES = Path(__file__).parent / "cases"


def _staged(name: str, tmp_path: Path) -> CaseDir:
    """Copy an ``Allrun`` fixture into ``tmp_path`` so the swap never touches source."""
    dest = tmp_path / name
    shutil.copytree(_CASES / name, dest)
    return CaseDir(dest)


def test_swap_replaces_every_getapplication_when_named_twice(tmp_path: Path) -> None:
    case = _staged("allrun_getapp_twice", tmp_path)

    swap_solver("interFoam", _APP)(case)

    text = (case.path / "Allrun").read_text()
    assert "$(getApplication)" not in text  # both occurrences gone
    assert text.count(f'"{_APP}"') == 2  # incl. the commented-out serial variant
    assert f'runParallel "{_APP}"' in text


def test_swap_rewrites_an_application_assignment(tmp_path: Path) -> None:
    case = _staged("allrun_application_assign", tmp_path)

    swap_solver("interFoam", _APP)(case)

    text = (case.path / "Allrun").read_text()
    # The value of `application=` is swapped cleanly, without nesting the quotes.
    assert f'application="{_APP}"' in text
    assert '""' not in text
    assert "interFoam" not in text
    # `${application}` is left untouched — it carries the swapped value at run time.
    assert "runApplication ${application}" in text


def test_swap_replaces_a_single_getapplication(tmp_path: Path) -> None:
    case = _staged("allrun_getapp_once", tmp_path)

    swap_solver("interFoam", _APP)(case)

    text = (case.path / "Allrun").read_text()
    assert f'runApplication "{_APP}"' in text
    assert "$(getApplication)" not in text


def test_swap_refuses_an_allrun_with_no_solver_invocation(tmp_path: Path) -> None:
    case = _staged("allrun_no_solver", tmp_path)

    with pytest.raises(NoSwapPoint):
        swap_solver("interFoam", _APP)(case)
