# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The ``cases/row4`` family: one full case plus per-variant build steps.

``cases/row4/common`` is the whole case — four unit cells in a row along x, the
first two in a ``modelZone`` cellZone, run steady (``simpleFoam``/``SIMPLE``,
``momentumPredictor no``). Unit cells make the cell volume exactly 1, so an
``fvMatrix`` source contribution equals the field value it came from and every
expectation the tests assert is derivable by hand.

A *variant* is a named tuple of :mod:`neofoam.tooling.casebuild` steps composed
onto that base with ``|`` (:func:`row4`):

* ``transient``      — the same case driven by ``pimpleFoam``/``PIMPLE`` with an
  ``Euler`` ddt, for the algorithm that owns its own copy of a momentum hook.
* ``fvOptions``      — ``system/fvOptions``: a ``vectorSemiImplicitSource`` on
  the zone (see ``test_fv_options.py``).
* ``fvOptionsLimit`` — ``constant/fvOptions`` (the *other* location
  ``fv::options`` searches) with a ``limitVelocity`` correction, plus the
  stronger ``0/U`` and the momentum solve that correction needs.
* ``mrf``            — ``constant/MRFProperties``: the zone as a rotating frame
  (see ``test_mrf.py``).

A variant that merely *edits* base dictionaries says so as ``patch`` steps
rather than committing a second copy of each whole file. The ``fv::options`` /
MRF dictionaries are not edits — they are whole dictionaries with no typed
representation (cell-zone selections, ``((0 5 0) 0)`` source tuples), so they
stay on disk under ``cases/row4/<variant>/``, copied by :func:`_overlay`; that
directory is also what pins *which* of the two locations ``fv::options``
searches the file sits in.

``cases/movingRow4`` is deliberately *not* part of this family: it is a
different mesh (one block, no cell zone, ``slip`` side walls) with its own
numerics, and shares only the two ``constant/`` property dictionaries.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from neofoam.tooling.casebuild import CaseDir, Pipeline, Step, from_template, patch

_ROW4 = Path(__file__).parent / "cases" / "row4"


def _overlay(name: str) -> Step:
    """Copy the committed dictionary under ``cases/row4/<name>`` onto the case."""

    def step(case: CaseDir) -> None:
        shutil.copytree(_ROW4 / name, case.path, dirs_exist_ok=True)

    return step


# ``pFinal``/``UFinal`` are the base ``solvers`` ``p``/``U`` entries at
# ``relTol 0`` — what a running pressure corrector looks up on the final outer
# iteration (see ``cases/row4/common/system/fvSolution``).
_P_FINAL = {
    "solver": "GAMG",
    "tolerance": 1e-07,
    "relTol": 0,
    "smoother": "DICGaussSeidel",
}
_U_FINAL = {
    "solver": "smoothSolver",
    "smoother": "symGaussSeidel",
    "tolerance": 1e-05,
    "relTol": 0,
}

_VARIANTS: dict[str, tuple[Step, ...]] = {
    "transient": (
        patch("system/controlDict", application="pimpleFoam", endTime=0.01, deltaT=0.01),
        patch(
            "system/fvSchemes",
            **{
                "ddtSchemes.default": "Euler",
                # the steady base bounds the convection term; a transient run does not
                "divSchemes.div(phi,U)": "Gauss linearUpwind grad(U)",
            },
        ),
        patch(
            "system/fvSolution",
            remove=["SIMPLE"],
            **{
                "solvers.pFinal": _P_FINAL,
                "solvers.UFinal": _U_FINAL,
                "PIMPLE": {
                    "nOuterCorrectors": 1,
                    "nCorrectors": 2,
                    "nNonOrthogonalCorrectors": 0,
                    "momentumPredictor": False,
                },
            },
        ),
    ),
    "fvOptions": (_overlay("fvOptions"),),
    "fvOptionsLimit": (
        _overlay("fvOptionsLimit"),
        # Well above the dictionary's ``max 1``, so every zone cell leaves the
        # momentum solve above the limit and the clamp has something to bite on.
        patch(
            "0/U",
            internalField="uniform (3 0 0)",
            **{"boundaryField.inlet.value": "uniform (3 0 0)"},
        ),
        # The clamp is applied by ``fvOptions.correct(U)``, which native calls right
        # after the momentum solve — so this case has to run that solve.
        patch("system/fvSolution", **{"SIMPLE.momentumPredictor": True}),
    ),
    "mrf": (_overlay("mrf"),),
}


def row4(*variants: str) -> Pipeline:
    """``cases/row4/common`` composed with the named *variants*, in order."""
    pipeline = from_template(_ROW4 / "common")
    for name in variants:
        for step in _VARIANTS[name]:
            pipeline = pipeline | step
    return pipeline
