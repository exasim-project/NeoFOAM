# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Enumerate the incompressible tutorials the drop-in study runs.

Selection, not classification: which cases run is stated in ``config.yaml``
(``cases:``), and this module only resolves each name to its tutorial directory
and reads the minimal metadata the sweep needs — the native solver to swap and the
subdomain count for thread budgeting. Whether a case *should* run, and why one
fails, is decided by the run itself and analysed afterwards, not predicted here.

With no ``cases:`` list the study falls back to walking the tutorial tree and
taking every ``simpleFoam``/``pimpleFoam`` case — a plain enumeration, still no
tiering.
"""

from __future__ import annotations

from pathlib import Path

from verification.dropin.cases import Case, case_id, subdomains
from verification.dropin.foamdict import entry, read, tutorials_root

#: Solver families incompressibleFluid claims to replace — the enumeration filter
#: for the fall-back tree walk (not a per-case verdict).
NATIVE_SOLVERS = ("simpleFoam", "pimpleFoam")

#: Fields diffed against the native reference after a run.
CANDIDATE_FIELDS = ("U", "p", "k", "epsilon", "omega", "nut", "nuTilda")


def _case(name: str, root: Path) -> Case:
    """Resolve one tutorial name to a runnable :class:`Case` (no classification)."""
    case = root / name
    application = entry(read(case / "system" / "controlDict"), "application")
    return Case(
        id=case_id(name),
        name=name,
        path=case,
        native_solver=application,
        app="",  # candidate backends come from config.yaml `apps:`
        fields=CANDIDATE_FIELDS,
        subdomains=subdomains(case),
    )


def _walk(root: Path) -> list[str]:
    """Every ``simpleFoam``/``pimpleFoam`` tutorial name under *root*, sorted.

    The fall-back when ``config.yaml`` names no cases. Skips the multi-setup
    harnesses and the pisoFoam tree, which are not clean drop-in single cases.
    """
    names: list[str] = []
    for control in sorted(root.glob("*/**/system/controlDict")):
        case = control.parent.parent
        if any(part.startswith("setups") for part in case.parts):
            continue
        if "pisoFoam" in case.relative_to(root).parts:
            continue
        if entry(read(control), "application") not in NATIVE_SOLVERS:
            continue
        names.append(str(case.relative_to(root)))
    return names


def discover(selection: list[str] | None = None) -> list[Case]:
    """The cases to run: the config's ``cases:`` list, or the whole tree if empty.

    Empty when no OpenFOAM is sourced (``$FOAM_TUTORIALS`` unset).
    """
    root = tutorials_root("incompressible")
    if root is None:
        return []
    names = list(selection) if selection else _walk(root)
    return [_case(name, root) for name in names]
