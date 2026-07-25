# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Coverage survey over the interFoam / interIsoFoam tutorial tree.

A **survey, not a gate**: it never fails on how a case classifies, only that the
classification is self-consistent — cases were found, every tier is one of the
four documented ones, every non-A case explains itself, and the alpha field was
resolved per case (nozzleFlow2D is ``alpha.fuel``, not ``alpha.water``). Needs a
sourced OpenFOAM for ``$FOAM_TUTORIALS``, so it skips when ``blockMesh`` is
absent; the strict pass/fail gate is the Snakemake sweep, not this test.
"""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.skipif(
    shutil.which("blockMesh") is None, reason="needs a sourced OpenFOAM"
)

_DISCOVER = (
    Path(__file__).parents[4]
    / "verification"
    / "foam_tutorials"
    / "incompressibleVoF"
    / "discover.py"
)


def _load_discover() -> Any:
    spec = importlib.util.spec_from_file_location("_vof_discover", _DISCOVER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_vof_discovery_classifies_the_tutorial_tree() -> None:
    cases = _load_discover().discover()

    assert cases, "no VoF tutorials discovered"
    assert {c.tier for c in cases} <= {"A", "B", "C", "D"}
    # Every non-tier-A case must explain itself, or the report is useless.
    assert all(c.reason for c in cases if c.tier != "A")
    # The alpha field is per-case and must lead the diffed fields.
    assert all(c.fields[0].startswith("alpha.") for c in cases)
