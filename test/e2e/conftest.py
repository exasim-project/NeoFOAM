# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

from pathlib import Path

import pytest

from neofoam.e2e.manifest import PatchManifest

CASES = Path(__file__).parent / "cases"


@pytest.fixture
def manifest() -> PatchManifest:
    return PatchManifest.load(CASES / "tube_bank_manifest.json")
