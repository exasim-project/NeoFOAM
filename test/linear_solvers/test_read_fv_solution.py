# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Spec that the typed models read and validate every tutorial's ``system/fvSolution``."""

import re
import warnings
from pathlib import Path
from typing import Any, Optional

import pybFoam as pyf
import pytest

from neofoam.foam import LinearSolverConversionWarning, linear_solver_entry
from neofoam.io.strategies.openfoam_strategy import _to_python

_ROOT = Path(__file__).parents[2]
_TUTORIALS = sorted((_ROOT / "tutorials").glob("**/system/fvSolution"))
_ROLES = {"p": "pressure", "pFinal": "pressure", "U": "momentum", "UFinal": "momentum"}


def _block(solvers: dict[str, Any], field: str) -> Optional[dict[str, Any]]:
    """The block selecting ``field``: its literal key, else a quoted regex key."""
    if field in solvers:
        return solvers[field]
    quoted = (key for key in solvers if len(key) > 1 and key[0] == key[-1] == '"')
    return next((solvers[key] for key in quoted if re.fullmatch(key[1:-1], field)), None)


@pytest.mark.parametrize("backend", ["openfoam", "ginkgo"])
@pytest.mark.parametrize(
    "path", _TUTORIALS, ids=lambda path: str(path.parents[1].relative_to(_ROOT / "tutorials"))
)
def test_tutorial_blocks_validate_as_plugins_of_the_backend(path: Path, backend: str) -> None:
    solvers = _to_python(pyf.dictionary.read(str(path)))["solvers"]
    blocks = {field: _block(solvers, field) for field in _ROLES}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LinearSolverConversionWarning)
        validated = [
            linear_solver_entry(backend, _ROLES[field]).model_validate(block).root
            for field, block in blocks.items()
            if block is not None
        ]
    assert {plugin.backend for plugin in validated} <= {backend}
