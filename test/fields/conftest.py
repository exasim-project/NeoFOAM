# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Shared fixtures for ``test/fields``.

The ``hotRoom`` case (a copy of the upstream
``tutorials/heatTransfer/buoyantBoussinesqPimpleFoam/hotRoom/0.orig``) is
the canonical on-disk fixture for the field-IO round-trip tests. Staging
it into ``tmp_path`` keeps the checked-in copy immutable (never mutate
checked-in files) while giving each test a writable case.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest


HOT_ROOM_FIXTURE = Path(__file__).parent / "cases" / "hotRoom" / "0.orig"


@pytest.fixture
def staged_hot_room(tmp_path: Path) -> Path:
    """Copy ``hotRoom/0.orig/*`` into ``tmp_path/case/0/`` and return the case dir."""
    case = tmp_path / "case"
    (case / "0").mkdir(parents=True)
    for f in HOT_ROOM_FIXTURE.iterdir():
        shutil.copy(f, case / "0" / f.name)
    return case
