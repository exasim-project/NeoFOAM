# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""One executed incompressibleVoF staged-init pipeline, shared by the tests.

``create_init(...).run()` needs a ``Foam::Time`` and a mesh, and a process may
own exactly one of each — so the pipeline runs once, in
``_create_fields_worker.py``, against ``cases/vofRow4`` and dumps everything the
tests assert on to JSON. Both ``test_create_fields.py`` and
``models/alpha_advection/test_shared.py`` read that single dump.

``cases/vofRow4`` is a unit cube cut into 4 cells along x (cell volume 0.25,
x-face area 1) carrying the interFoam damBreak physical properties
(``rho1 = 1000``, ``rho2 = 1``, ``g = (0 -9.81 0)``) — so every field the
pipeline builds has an exact, hand-derivable value. See the two test modules for
the derivations.

``cases/vofRow4Href`` is the same mesh/properties plus a non-default
``constant/hRef`` (0.3), for the one test that needs ``hRef`` actually read
from disk rather than defaulted; see
``test_gh_uses_a_non_default_hRef_as_the_reference_head`` in
``test_create_fields.py``.

``cases/vofRow4Moving`` is the same mesh/properties plus a
``constant/dynamicMeshDict`` (solid-body oscillation along gravity), for the
tests that need the mesh *selection* to go through ``dynamicFvMesh::New`` —
see ``test_dynamic_mesh.py``.

``cases/vofRow4Divergent`` is the same mesh/properties with an inlet that feeds
half of what the interior carries, so ``createPhi(U)`` is *not* divergence-free —
the start-up flux projection of ``initCorrectPhi.H`` has something to do; see
``test_flux_correction.py``.

``cases/vofRow4Porous`` is the same mesh/properties switched to isoAdvector and
given ``constant/porosityProperties`` + ``0/porosity``, for the tests that need
``Foam::isoAdvection`` to find a porosity field in the registry — see
``models/alpha_advection/models/test_iso_advector.py``.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from typing import Any

import pybFoam as pyf
import pytest

_HERE = Path(__file__).parent
_CASE = _HERE / "cases" / "vofRow4"
_HREF_CASE = _HERE / "cases" / "vofRow4Href"
_MOVING_CASE = _HERE / "cases" / "vofRow4Moving"
_POROUS_CASE = _HERE / "cases" / "vofRow4Porous"
_DIVERGENT_CASE = _HERE / "cases" / "vofRow4Divergent"
_WORKER = _HERE / "_create_fields_worker.py"


@dataclass
class BuiltCase:
    """The worker's dump of one executed pipeline, plus the case it ran in."""

    result: dict[str, Any]
    case: Path
    # Parsed written-field dictionaries, kept alive on purpose: pybFoam hands
    # back views into the parent dictionary, so reading off a temporary
    # (``dictionary.read(path).subDict(...)``) yields a dangling view that
    # silently reads as empty.
    _dicts: dict[str, Any] = dataclass_field(default_factory=dict, repr=False)

    def internal(self, field: str) -> Any:
        """Internal-field values of ``ctx.fields[field]``."""
        return self.result["internal"][field]

    def _written(self, field: str) -> Any:
        """The field OpenFOAM wrote after init, read back as a dictionary."""
        if field not in self._dicts:
            path = self.case / self.result["written_time"] / field
            self._dicts[field] = pyf.dictionary.read(str(path))
        return self._dicts[field]

    def written_dimensions(self, field: str) -> list[int]:
        """Dimension exponents of the written field (e.g. ``[1, -3, 0, ...]``)."""
        entry = str(self._written(field).getOrDefault[str]("dimensions", ""))
        return [int(token) for token in entry.strip("[] ").split()]

    def written_boundary(self, field: str, patch: str) -> Any:
        """The written ``boundaryField/<patch>`` sub-dictionary."""
        return self._written(field).subDict("boundaryField").subDict(patch)


def _run_pipeline(
    tmp_path_factory: pytest.TempPathFactory, source_case: Path, tmp_name: str
) -> BuiltCase:
    """Mesh *source_case* and run the staged init on the copy, once."""
    case = tmp_path_factory.mktemp(tmp_name) / "case"
    shutil.copytree(source_case, case)
    subprocess.run(
        ["blockMesh", "-case", str(case)],
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    subprocess.run(
        [sys.executable, str(_WORKER), str(case)],
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    return BuiltCase(json.loads((case / "create_fields.json").read_text()), case)


@pytest.fixture(scope="session")
def vof_row4(tmp_path_factory: pytest.TempPathFactory) -> BuiltCase:
    """Mesh ``cases/vofRow4`` and run the staged init on it, once per session."""
    return _run_pipeline(tmp_path_factory, _CASE, "vofRow4")


@pytest.fixture(scope="session")
def vof_row4_href(tmp_path_factory: pytest.TempPathFactory) -> BuiltCase:
    """Mesh ``cases/vofRow4Href`` (``constant/hRef`` = 0.3) and run the pipeline."""
    return _run_pipeline(tmp_path_factory, _HREF_CASE, "vofRow4Href")


@pytest.fixture(scope="session")
def vof_row4_moving(tmp_path_factory: pytest.TempPathFactory) -> BuiltCase:
    """Mesh ``cases/vofRow4Moving`` (has a ``constant/dynamicMeshDict``) and run it."""
    return _run_pipeline(tmp_path_factory, _MOVING_CASE, "vofRow4Moving")


@pytest.fixture(scope="session")
def vof_row4_porous(tmp_path_factory: pytest.TempPathFactory) -> BuiltCase:
    """Mesh ``cases/vofRow4Porous`` (isoAdvector + porosity) and run the pipeline."""
    return _run_pipeline(tmp_path_factory, _POROUS_CASE, "vofRow4Porous")


@pytest.fixture(scope="session")
def vof_row4_divergent(tmp_path_factory: pytest.TempPathFactory) -> BuiltCase:
    """Mesh ``cases/vofRow4Divergent`` (non-solenoidal ``0/U``) and run the pipeline."""
    return _run_pipeline(tmp_path_factory, _DIVERGENT_CASE, "vofRow4Divergent")
