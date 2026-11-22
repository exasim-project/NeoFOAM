# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Case construction and subprocess plumbing shared by the turbulence tests.

Exactly **one** complete case is committed: ``parity_base/`` — a wall-free box
carrying the solver settings (``controlDict`` / ``fvSchemes`` / ``fvSolution``), the
transport properties and the uniform ``0/`` templates every case here shares. The
walled cases commit only what *differs* from it — their ``blockMeshDict`` and the
``0/`` fields that carry wall functions — and are composed onto it with
``neofoam.tooling.casebuild``: :func:`overlay` lays the delta tree down and
``patch`` edits an inherited dictionary in place.

No ``constant/turbulenceProperties`` is committed at all: :func:`turbulence_properties`
writes one from :data:`COEFFS`, so every coefficient appears exactly once, in Python,
next to the tests that assert it — and the dictionary both backends run *is* the one
those assertions are written against.

Nothing here constructs a ``Foam::Time`` (only dictionary reads/writes): a process may
own exactly one, so meshing and every model build happen in :mod:`_parity_worker`,
driven by :func:`run_worker`.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from neofoam.tooling.casebuild import (
    CaseDir,
    Pipeline,
    Step,
    configs,
    from_template,
    patch,
)
from neofoam.turbulence.config import RASProperties, TurbulencePropertiesConfig

_HERE = Path(__file__).parent
_WORKER = _HERE / "_parity_worker.py"

#: The one complete case; every other case starts from it.
PARITY_BASE = _HERE / "parity_base"

#: Deltas over :data:`PARITY_BASE` — a mesh with walls plus the ``0/`` fields whose
#: boundaries are wall functions. Not runnable on their own.
WALLED_DELTA = _HERE / "walled_base"
WALL_FUNCTION_DELTA = _HERE / "wall_function_base"


# --------------------------------------------------------------------------- #
# Turbulence dictionaries                                                      #
# --------------------------------------------------------------------------- #

#: The ``<model>Coeffs`` block each ``*Coeffs`` dictionary declares: **every**
#: coefficient off its OpenFOAM default. Both backends read the dictionary written
#: from this table, so parity can only mean the NeoN closure honours the overrides,
#: and ``test_config`` asserts these same numbers come back out of the reader.
#:
#: ``SpalartAllmaras`` deliberately omits ``Cw1``: OpenFOAM derives it from
#: ``Cb1`` / ``kappa`` / ``Cb2`` / ``sigmaNut``.
COEFFS: dict[str, dict[str, float]] = {
    "kEpsilon": {
        "Cmu": 0.085,
        "C1": 1.42,
        "C2": 1.68,
        "C3": 0.05,
        "sigmak": 1.2,
        "sigmaEps": 1.11,
    },
    "SpalartAllmaras": {
        "sigmaNut": 0.7,
        "kappa": 0.42,
        "Cb1": 0.14,
        "Cb2": 0.6,
        "Cw2": 0.32,
        "Cw3": 2.1,
        "Cv1": 7.0,
        "Cs": 0.35,
    },
    "kOmegaSST": {
        "alphaK1": 0.8,
        "alphaK2": 1.1,
        "alphaOmega1": 0.55,
        "alphaOmega2": 0.9,
        "gamma1": 0.52,
        "gamma2": 0.46,
        "beta1": 0.08,
        "beta2": 0.09,
        "betaStar": 0.085,
        "a1": 0.32,
        "b1": 1.1,
        "c1": 9.0,
    },
}

#: An entry no closure declares, written into every ``*Coeffs`` block: OpenFOAM
#: ignores an unrecognised key rather than rejecting the dictionary, and so must the
#: coefficient resolution.
UNDECLARED_COEFFICIENT = {"notACoefficient": 42}


def turbulence_config(name: str) -> TurbulencePropertiesConfig:
    """The ``turbulenceProperties`` *name* stands for.

    A bare model name (``laminar``, ``kEpsilon``, ``realizableKE``, …) declares no
    coefficients block, so the closure keeps every OpenFOAM default; the
    ``<model>Coeffs`` spelling selects the same closure through a full override
    block taken from :data:`COEFFS`.
    """
    if name == "laminar":
        return TurbulencePropertiesConfig(simulationType="laminar")
    model = name.removesuffix("Coeffs")
    overrides = (
        {f"{model}Coeffs": {**COEFFS[model], **UNDECLARED_COEFFICIENT}} if name != model else {}
    )
    return TurbulencePropertiesConfig(
        simulationType="RAS",
        RAS=RASProperties(RASModel=model, turbulence=True, printCoeffs=False, **overrides),
    )


def turbulence_properties(name: str) -> Step:
    """Write the ``constant/turbulenceProperties`` *name* stands for."""
    return configs(turbulence_config(name))


# --------------------------------------------------------------------------- #
# Cases                                                                        #
# --------------------------------------------------------------------------- #


def overlay(src: Path) -> Step:
    """Lay a delta tree over the case, replacing the base files it names."""

    def step(case: CaseDir) -> None:
        shutil.copytree(src, case.path, dirs_exist_ok=True)

    return step


def parity_case(model: str) -> Pipeline:
    """The wall-free box the NeoN-vs-pybFoam comparison runs, carrying *model*.

    No walls means no wall functions (whose near-wall cell overrides the pure-Python
    model does not replicate); those are :func:`wall_function_case`'s subject.
    """
    return from_template(PARITY_BASE) | turbulence_properties(model)


def walled_case() -> Pipeline:
    """A ``kEpsilon`` box whose two z patches are walls, with no ``wallDist`` scheme.

    ``kqRWallFunction`` / ``epsilonWallFunction`` / ``nutk`` boundaries on a unit cube
    of 4 x 4 x 4 uniform cells, so every wall face's owner-cell centre sits exactly
    half a cell — 0.5/4 = 0.125 m — from its wall.

    The inherited ``wallDist`` block is **removed**: OpenFOAM's kEpsilon needs no
    ``wallDist { method … }`` entry — its wall functions read ``nearWallDist``, which is
    pure patch geometry — so the tutorials ship none, and the NeoN closure has to cope
    with its absence (see ``test_near_wall_dist``).
    """
    return (
        from_template(PARITY_BASE)
        | overlay(WALLED_DELTA)
        | patch("system/fvSchemes", remove=["wallDist"])
        | turbulence_properties("kEpsilon")
    )


def wall_function_case(model: str) -> Pipeline:
    """A cube whose **four** x/z faces are walls, carrying *model*.

    ``nu = 2e-3`` is chosen so the wall-face
    ``y+ = Cmu^0.25 y sqrt(k) / nu = 34.2 sqrt(k)`` straddles ``yPlusLam = 11.53`` over
    the seeded ``k`` range: ``k`` below 0.113 lands in the viscous branch of the
    STEPWISE ``nutkWallFunction``, ``k`` above it in the log branch — so both branches
    are taken within one run.
    """
    return (
        from_template(PARITY_BASE)
        | overlay(WALL_FUNCTION_DELTA)
        | patch("constant/transportProperties", nu=2.0e-3)
        | turbulence_properties(model)
    )


def run_worker(role: str, case: Path) -> None:
    """Run one worker role in a fresh process (one ``Foam::Time`` per process)."""
    subprocess.run(
        [sys.executable, str(_WORKER), role, str(case)],
        check=True,
        cwd=str(case.parent),
    )
