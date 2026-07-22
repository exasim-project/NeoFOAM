# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Classify every incompressible tutorial for the incompressibleFluid drop-in.

Purely static — read the case's dictionaries, run nothing — so the full inventory
is known in milliseconds. This predicts which cases stand a chance; whether one
*actually* runs is decided by the engine, which records the observed outcome
beside this prediction.
"""

from __future__ import annotations

from pathlib import Path

from neofoam.tooling.verification.foamdict import (
    entry,
    read,
    turbulence,
    tutorials_root,
    uses_ami,
)
from neofoam.tooling.verification.study import Case, case_id, subdomains

#: Solver families incompressibleFluid claims to replace.
NATIVE_SOLVERS = ("simpleFoam", "pimpleFoam")

#: Turbulence models with a pure-Python implementation under neofoam/turbulence.
SUPPORTED_RAS = ("kEpsilon", "kOmegaSST", "SpalartAllmaras", "realizableKE")

#: Case files whose presence implies a capability incompressibleFluid lacks.
UNSUPPORTED_FEATURES = {
    "dynamicMeshDict": "dynamic mesh / mesh motion",
    "MRFProperties": "MRF zones",
    "fvOptions": "fvOptions sources",
}

CANDIDATE_FIELDS = ("U", "p", "k", "epsilon", "omega", "nut", "nuTilda")

NEOFOAM_APP = "neofoam solver incompressiblefluid"

TIER_TITLES = {
    "A": "Tier A — expected to run (supported turbulence, no unsupported features)",
    "B": "Tier B — turbulence model not implemented",
    "C": "Tier C — LES (no LES models in neofoam)",
    "D": "Tier D — solver feature not implemented",
}


def classify(case: Path, root: Path) -> Case | None:
    """Classify one case directory; None when it is not a solver case we cover."""
    control = read(case / "system" / "controlDict")
    if not control:
        return None
    application = entry(control, "application")
    if application not in NATIVE_SOLVERS:
        return None

    simulation, model = turbulence(case)
    parallel = any(case.glob("system/decomposeParDict*"))
    name = str(case.relative_to(root))

    features = [
        label
        for name_, label in UNSUPPORTED_FEATURES.items()
        if any(case.glob(f"constant/{name_}")) or any(case.glob(f"system/{name_}"))
    ]
    if uses_ami(case):
        features.append("cyclicAMI interfaces")

    common = dict(
        id=case_id(name),
        name=name,
        path=case,
        native_solver=application,
        app=NEOFOAM_APP,
        fields=CANDIDATE_FIELDS,
        turbulence=model,
        parallel=parallel,
        subdomains=subdomains(case),
        features=features,
    )

    # Order matters: report the most fundamental blocker first.
    if simulation == "LES":
        return Case(
            **common, tier="C", reason=f"LES ({model}) — no LES models in neofoam"
        )
    if features:
        return Case(**common, tier="D", reason="; ".join(features))
    if model not in SUPPORTED_RAS and model != "laminar":
        return Case(**common, tier="B", reason=f"RAS model {model} not implemented")
    return Case(**common, tier="A", reason="")


def discover() -> list[Case]:
    """All classified cases, sorted by name. Empty when no OpenFOAM is sourced."""
    root = tutorials_root("incompressible")
    if root is None:
        return []

    cases: list[Case] = []
    for control in sorted(root.glob("*/**/system/controlDict")):
        case = control.parent.parent
        # Multi-setup harnesses (LES/planeChannel, laminar/planarPoiseuille) drive
        # their own per-setup Allrun scripts; out of scope for the sweep.
        if any(part.startswith("setups") for part in case.parts):
            continue
        # pisoFoam is not a solver family incompressibleFluid replaces. Most pisoFoam
        # cases declare `application pisoFoam` and are already dropped by classify(),
        # but LES/motorBike declares `application simpleFoam` for its steady init step
        # and would otherwise leak in — exclude the whole pisoFoam tree by path.
        if "pisoFoam" in case.relative_to(root).parts:
            continue
        classified = classify(case, root)
        if classified is not None:
            cases.append(classified)
    return cases
