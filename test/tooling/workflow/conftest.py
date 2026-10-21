# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

from pathlib import Path

import pytest

from neofoam.tooling.workflow.patch_set import PatchSet

CASES = Path(__file__).parent / "cases"


@pytest.fixture
def patch_set() -> PatchSet:
    return PatchSet.load(CASES / "tube_bank_manifest.json")
