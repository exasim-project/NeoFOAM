# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The mesh step refuses adaptive mesh refinement and lets mesh motion through.

``create_mesh`` builds a ``dynamicFvMesh`` for any case carrying
``constant/dynamicMeshDict``, but only *motion* is implemented: nothing
downstream reacts to a topology change, so an AMR case would silently solve on
the initial mesh and report plausible but wrong results. The guard is a
dictionary check, so these tests need no mesh and no ``Foam::Time`` — the
refusal is asserted with an *empty* context, which is itself the claim that
nothing is constructed before the case is refused.

**Cases.** ``cases/refiningMesh`` is the ``constant/dynamicMeshDict`` of
``interFoam/laminar/damBreakWithObstacle`` (``dynamicRefineFvMesh``);
``cases/movingMesh`` is a solid-body motion dictionary, the supported kind.
That the moving one really does build its motion solver end to end is proven by
``test/solver/incompressibleVoF/test_dynamic_mesh.py``; here it only has to
survive the guard.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from neofoam.foam.initialization import _refinement_mesh_type, create_mesh

_CASES = Path(__file__).parent / "cases"


def test_a_refining_dynamic_mesh_is_refused_before_anything_is_built(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(_CASES / "refiningMesh")

    with pytest.raises(NotImplementedError, match="dynamicRefineFvMesh"):
        create_mesh().initializer({})


def test_a_motion_only_dynamic_mesh_is_not_a_refinement_mesh() -> None:
    assert _refinement_mesh_type(_CASES / "movingMesh") is None


def test_a_refining_dynamic_mesh_reports_the_type_the_case_selected() -> None:
    assert _refinement_mesh_type(_CASES / "refiningMesh") == "dynamicRefineFvMesh"
