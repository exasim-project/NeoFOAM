# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""A field port's access on NeoN: ``T`` read and written on a patch and on the cells.

Region ``right`` of ``cases/coupledBlocks``, uncoupled by ``cases/uncoupledRight``:
350 K inside, 300 K on ``outer`` (5 faces). NeoNFields writes the 5 faces of
``interface`` (``calculated``, as a written patch is) as fixed values and the 30
cells, then reads them back; the values pass through unchanged, so they compare
exactly.
"""

from pathlib import Path
from typing import Any

import pytest

from solver.heatConductionNeoN._run_case import mesh_copy, result


@pytest.fixture(scope="module")
def access(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    case: Path = mesh_copy(
        "coupledBlocks", tmp_path_factory.mktemp("fields") / "case", "uncoupledRight"
    )
    return result(case / "right", "fields")


def test_neon_field_access_reads_a_patch_and_the_cells(access: dict[str, Any]) -> None:
    assert access["read"] == {"outer": [300.0] * 5, "cells": [350.0] * 30}


def test_neon_field_access_writes_a_patch_and_the_cells(access: dict[str, Any]) -> None:
    assert access["written"] == {
        "interface": [301.0, 302.0, 303.0, 304.0, 305.0],
        "cells": [float(cell) for cell in range(30)],
    }
