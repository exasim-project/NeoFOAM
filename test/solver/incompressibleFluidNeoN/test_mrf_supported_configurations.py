# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Which MRF cases the NeoN rotating-frame model runs, and which it refuses.

The NeoN MRF spec supports a subset of the cases native MRF supports,
and every gap is a *quiet* one. ``pEqn.H`` wraps the ddt flux correction of
``phiHbyA`` in ``MRF.zeroFilter(...)`` and the NeoN pressure extension declares
no hook for that, so a transient (PIMPLE / PISO) run would carry the
absolute-frame correction into the rotating zone; NeoN has no
``constrainPressure`` to set a ``fixedFluxPressure`` refGrad, so such a patch
would behave as ``zeroGradient``; and the frame quantities are probed out of the
zone list once at construction, with no mesh-change hook to rebuild them, so on a
moving mesh they would go on describing the mesh at ``t=0``. Each gives
wrong-but-plausible fields with nothing raised, so ``@mrfNeoN.build`` rejects them
up front — which is what these tests pin.

The BUILD stage alone is the subject: it raises before any InitStep runs, so no
mesh, no ``Foam::Time`` and no solve are needed. Its guards read case files
relative to the working directory the way the solver does, so each case is
copied into ``tmp_path`` and entered there.

Each unsupported case is committed as a small delta laid over ``cases/mrfBox``
(the layout of ``_regex_case.py``): ``mrfBoxTransient`` changes the algorithm
block (and the matching ``controlDict``), ``mrfBoxRestart`` only ``startFrom``
plus the ``1/p`` a restart actually reads, ``mrfBoxDynamicMesh`` only the
``constant/dynamicMeshDict`` whose mere presence selects a moving mesh — so the
committed diff *is* the statement of what each rejection turns on. In particular
the restart case keeps ``mrfBox``'s clean ``0/p``: a guard that inspected ``0/p``
rather than the start-time field would pass it.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any, Optional

import pytest

from neofoam.solver.incompressibleFluidNeoN.models.mrf import mrfNeoN

_HERE = Path(__file__).parent
_CASES = _HERE / "cases"

#: The complete steady case every delta is laid over — supported as it stands.
_MRF_BOX = _CASES / "mrfBox"


def staged(delta: Optional[str], tmp_path: Path) -> Path:
    """``cases/mrfBox`` with ``cases/<delta>`` laid over it, in a scratch copy."""
    case = tmp_path / "case"
    shutil.copytree(_MRF_BOX, case)
    if delta is not None:
        shutil.copytree(_CASES / delta, case, dirs_exist_ok=True)
    return case


def build_mrf(case: Path) -> list[Any]:
    """The MRF model's BUILD stage for *case*, as the solver's init graph runs it."""
    return mrfNeoN.instantiate(case_dir=case).run_build()


def test_the_steady_case_builds_the_rotating_frame_handle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = staged(None, tmp_path)
    monkeypatch.chdir(case)

    steps = build_mrf(case)

    assert [step.name for step in steps] == ["models.mrf_neon"]


@pytest.mark.parametrize(
    "delta, message",
    [
        ("mrfBoxTransient", "only for the steady SIMPLE algorithm"),
        ("mrfBoxRestart", "1/p patch(es) yMin use fixedFluxPressure"),
        ("mrfBoxDynamicMesh", "does not support rotating zones on a moving mesh"),
    ],
)
def test_an_unsupported_case_is_rejected_before_anything_is_built(
    delta: str, message: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = staged(delta, tmp_path)
    monkeypatch.chdir(case)

    with pytest.raises(ValueError, match=re.escape(message)):
        build_mrf(case)
