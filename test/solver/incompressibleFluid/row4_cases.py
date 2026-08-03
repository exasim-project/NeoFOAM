# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The ``cases/row4`` family: one full case plus per-variant overlays.

``cases/row4/common`` is the whole case — four unit cells in a row along x, the
first two in a ``modelZone`` cellZone, run steady (``simpleFoam``/``SIMPLE``,
``momentumPredictor no``). Unit cells make the cell volume exactly 1, so an
``fvMatrix`` source contribution equals the field value it came from and every
expectation the tests assert is derivable by hand.

Every other directory under ``cases/row4`` holds **only the files that differ**
from that base, and a case is composed by copying the base and overlaying them
(the layout ``cases/alphaCourant/common`` uses in ``incompressibleVoF``):

* ``transient``      — ``system/{controlDict,fvSchemes,fvSolution}``: the same
  case driven by ``pimpleFoam``/``PIMPLE`` with an ``Euler`` ddt, for the
  algorithm that owns its own copy of a momentum hook.
* ``fvOptions``      — ``system/fvOptions``: a ``vectorSemiImplicitSource`` on
  the zone (see ``test_fv_options.py``).
* ``fvOptionsLimit`` — ``constant/fvOptions`` (the *other* location
  ``fv::options`` searches) with a ``limitVelocity`` correction, plus the
  ``0/U`` and ``system/fvSolution`` that correction needs.
* ``mrf``            — ``constant/MRFProperties``: the zone as a rotating frame
  (see ``test_mrf.py``).

``cases/movingRow4`` is deliberately *not* part of this family: it is a
different mesh (one block, no cell zone, ``slip`` side walls) with its own
numerics, and shares only the two ``constant/`` property dictionaries.
"""

from __future__ import annotations

import shutil
from pathlib import Path

_ROW4 = Path(__file__).parent / "cases" / "row4"


def stage_row4(dest: Path, *overlays: str) -> Path:
    """Copy ``cases/row4/common`` to *dest*, then overlay each named directory."""
    shutil.copytree(_ROW4 / "common", dest)
    for overlay in overlays:
        shutil.copytree(_ROW4 / overlay, dest, dirs_exist_ok=True)
    return dest
