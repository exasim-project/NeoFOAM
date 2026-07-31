# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Classify every interFoam / interIsoFoam tutorial for the incompressibleVoF drop-in.

Purely static — read the case's dictionaries, run nothing — so the full inventory
is known in milliseconds. This predicts which cases stand a chance; whether one
*actually* runs is decided by the engine, which records the observed outcome
beside this prediction.

Same contract as ``incompressibleFluid/discover.py``, but the blockers differ in
kind. There, physics models were missing (Python turbulence models, no LES); here
turbulence is *not* a blocker at all — the neofoam VoF solver builds the C++
``multiphase.TwoPhaseTransportModel``, so a VoF case gets the full native
OpenFOAM RAS/LES catalogue. What blocks VoF is numerics control: guards in the
Python solver reject settings the tutorials use routinely.
"""

from __future__ import annotations

from pathlib import Path

from verification.dropin.cases import Case, case_id, subdomains
from verification.dropin.foamdict import (
    entries,
    entry,
    read,
    turbulence,
    tutorials_root,
    uses_ami,
)

#: Solver families incompressibleVoF claims to replace.
NATIVE_SOLVERS = ("interFoam", "interIsoFoam")

#: Subtrees of ``$FOAM_TUTORIALS/multiphase`` to sweep.
SUBTREES = ("interFoam", "interIsoFoam")

NEOFOAM_APP = "neofoam solver incompressiblevof"

#: Fields diffed after the run, in addition to the case's own alpha field.
BASE_FIELDS = ("U", "p_rgh", "p")

#: Tier -> heading. Nothing consumes this today: the report shows the observed
#: outcome, never the prediction beside it. Kept because it is the label set a
#: predicted-vs-observed column would need, and that is an open question — see
#: plans/study-decomposition-spec.md. Delete it with `tier` if the answer is no.
TIER_TITLES = {
    "A": "Tier A — predicted runnable",
    "C": "Tier C — PIMPLE controls unsupported",
    "D": "Tier D — solver feature not implemented",
}


def _alpha_field(case: Path) -> str:
    """The case's alpha field name — ``alpha.water`` for all but nozzleFlow2D."""
    for sub in ("0.orig", "0"):
        found = sorted((case / sub).glob("alpha.*")) if (case / sub).is_dir() else []
        for item in found:
            if item.is_file() and not item.name.endswith(".orig"):
                return item.name
    return "alpha.water"


def _int_entries(text: str, key: str) -> list[int]:
    out = []
    for value in entries(text, key):
        try:
            out.append(int(value))
        except ValueError:
            continue
    return out


def _feature_blockers(case: Path) -> list[str]:
    """Case files/settings implying a solver capability incompressibleVoF lacks."""
    blockers = []

    mesh_dict = read(case / "constant" / "dynamicMeshDict")
    if mesh_dict:
        # Mesh *motion* is supported (dynamicFvMesh::New + mesh.update()); only
        # refinement is not, and the solver refuses it up front. Mirror its test:
        # every OpenFOAM refining mesh type carries "Refine" in its name (see
        # neofoam.foam.initialization._REFINEMENT_MESH_MARKER).
        kind = entry(mesh_dict, "dynamicFvMesh")
        if kind and "Refine" in kind:
            blockers.append(f"adaptive mesh refinement ({kind})")

    # `constant/MRFProperties` is no longer a blocker: the `mrf` optional model
    # binds IOMRFZoneList and hooks it into UEqn.H/pEqn.H. Nor is an `fvOptions`
    # dictionary: the `fvOptions` optional model binds fv::options and hooks its
    # source/constrain/correct into the same two files.
    if (case / "constant" / "porosityProperties").is_file():
        blockers.append("porosity zones")
    if "localEuler" in entry(read(case / "system" / "fvSchemes"), "default"):
        blockers.append("LTS (localEuler ddt)")
    if uses_ami(case):
        blockers.append("cyclicAMI interfaces")
    return blockers


def classify(case: Path, root: Path) -> Case | None:
    """Classify one case directory; None when it is not a solver case we cover."""
    control = read(case / "system" / "controlDict")
    if not control:
        return None
    application = entry(control, "application")
    if application not in NATIVE_SOLVERS:
        return None

    _, model = turbulence(case)
    solution = read(case / "system" / "fvSolution")
    parallel = any(case.glob("system/decomposeParDict*"))
    name = str(case.relative_to(root))

    features = _feature_blockers(case)

    # A key may appear once per sub-dict; take the extremum that would trip the guard.
    correctors = _int_entries(solution, "nCorrectors")
    frozen = entry(solution, "frozenFlow") in ("yes", "true", "on", "1")

    common = dict(
        id=case_id(name),
        name=name,
        path=case,
        native_solver=application,
        app=NEOFOAM_APP,
        fields=(_alpha_field(case), *BASE_FIELDS),
        turbulence=model,
        parallel=parallel,
        subdomains=subdomains(case),
    )

    # Order matters: report the most fundamental blocker first.
    if features:
        return Case(**common, tier="D", reason="; ".join(features))
    if frozen:
        return Case(**common, tier="C", reason="frozenFlow (no frozen-flow path in neofoam)")
    if correctors and min(correctors) < 2:
        return Case(
            **common,
            tier="C",
            reason=f"nCorrectors {min(correctors)} < 2 (control_factory.py rejects it)",
        )
    return Case(**common, tier="A", reason="")


def discover() -> list[Case]:
    """All classified cases, sorted by name. Empty when no OpenFOAM is sourced."""
    root = tutorials_root("multiphase")
    if root is None:
        return []

    cases: list[Case] = []
    for subtree in SUBTREES:
        for control in sorted((root / subtree).glob("**/system/controlDict")):
            case = control.parent.parent
            classified = classify(case, root)
            if classified is not None:
                cases.append(classified)
    return sorted(cases, key=lambda c: c.name)
