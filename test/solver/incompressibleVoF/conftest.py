# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The vofRow4 case layers, and one executed incompressibleVoF staged-init pipeline.

**The cases.** ``cases/vofRow4/common`` is the one full case every VoF test
runs on: a unit cube cut into 4 cells along x (cell volume 0.25, x-face area 1)
carrying the interFoam damBreak physical properties (``rho1 = 1000``,
``rho2 = 1``, ``g = (0 -9.81 0)``) — so every field the pipeline builds has an
exact, hand-derivable value. Its siblings under ``cases/vofRow4/`` are
*overlays*: each holds only the files that differ from ``common`` (and, where a
variant is a variant of a variant, from the layer below it), and
:func:`stage_case` composes a runnable case by copying the layers over one
another in order. Adding a scenario is therefore one file, not a case tree, and
"what makes this case different" is the directory listing.

The layers, and what each one is for:

* ``href`` — a non-default ``constant/hRef`` (0.3), for the one test that needs
  ``hRef`` actually read from disk rather than defaulted
  (``test_create_fields.py``);
* ``moving`` — a ``constant/dynamicMeshDict`` (solid-body oscillation along
  gravity) so the mesh *selection* goes through ``dynamicFvMesh::New``
  (``test_dynamic_mesh.py``), with ``movingCorrectPhi`` (``correctPhi yes``) and
  ``isoAdvectorMoving`` (``advectionScheme isoAdvector``) stacked on top of it;
* ``divergent`` — an inlet that feeds half of what the interior carries, so
  ``createPhi(U)`` is *not* divergence-free and ``initCorrectPhi.H`` has
  something to do (``test_flux_correction.py``);
* ``porous`` — isoAdvector plus ``constant/porosityProperties`` + ``0/porosity``,
  so ``Foam::isoAdvection`` finds a porosity field in the registry
  (``models/alpha_advection/models/test_iso_advector.py``);
* ``duct`` — the row re-cut as an open duct with a ``source`` cellZone, shared by
  the ``source`` (``system/fvOptions``) and ``limited``
  (``constant/fvOptions``) layers (``test_fv_options.py``);
* ``rotor`` — the block named so blockMesh makes a cellZone, plus
  ``constant/MRFProperties`` (``test_mrf.py``);
* ``subCycle``, ``prevCorr``, ``crankNicolson`` (+ ``crankNicolsonSubCycle``) —
  single-entry MULES control variants
  (``models/alpha_advection/models/test_mules.py``).

``models/pressure_velocity/cases/{open,closed,closedRefPoint}`` are further
overlays of the same ``common`` base, kept next to the tests that own them.

**The pipeline.** ``create_init(...).run()`` needs a ``Foam::Time`` and a mesh,
and a process may own exactly one of each — so the pipeline runs once, in
``_create_fields_worker.py``, and dumps everything the tests assert on to JSON.
Both ``test_create_fields.py`` and ``models/alpha_advection/test_shared.py``
read that single dump.
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
_WORKER = _HERE / "_create_fields_worker.py"

#: The vofRow4 layer directory: ``common`` plus one sub-directory per overlay.
VOF_ROW4 = _HERE / "cases" / "vofRow4"


def stage_case(dest: Path, *layers: Path) -> Path:
    """Compose a runnable case at *dest* by copying *layers* over one another.

    The first layer is a full case directory, each following one an overlay that
    replaces or adds the files it carries — so a variant is checked in as its
    diff. Use it instead of ``shutil.copytree`` of a single case (TEST_STYLE rule
    4: never mutate the checked-in files)::

        case = stage_case(tmp_path / "case", VOF_ROW4 / "common", VOF_ROW4 / "moving")

    A layer never *removes* a file, so a variant that must drop one gets its own
    base layer instead (see ``duct`` for ``source``/``limited``).
    """
    for layer in layers:
        shutil.copytree(layer, dest, dirs_exist_ok=True)
    return dest


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
    tmp_path_factory: pytest.TempPathFactory, variant: str | None = None
) -> BuiltCase:
    """Compose ``vofRow4/common`` (+ *variant*), mesh it and run the staged init."""
    tmp_name = f"vofRow4{variant.capitalize()}" if variant else "vofRow4"
    layers = [VOF_ROW4 / "common"] + ([VOF_ROW4 / variant] if variant else [])
    case = stage_case(tmp_path_factory.mktemp(tmp_name) / "case", *layers)
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
    """Mesh the plain ``vofRow4/common`` case and run the staged init, once."""
    return _run_pipeline(tmp_path_factory)


@pytest.fixture(scope="session")
def vof_row4_href(tmp_path_factory: pytest.TempPathFactory) -> BuiltCase:
    """The ``href`` overlay (``constant/hRef`` = 0.3), through the pipeline."""
    return _run_pipeline(tmp_path_factory, "href")


@pytest.fixture(scope="session")
def vof_row4_moving(tmp_path_factory: pytest.TempPathFactory) -> BuiltCase:
    """The ``moving`` overlay (has a ``constant/dynamicMeshDict``), through the pipeline."""
    return _run_pipeline(tmp_path_factory, "moving")


@pytest.fixture(scope="session")
def vof_row4_porous(tmp_path_factory: pytest.TempPathFactory) -> BuiltCase:
    """The ``porous`` overlay (isoAdvector + porosity), through the pipeline."""
    return _run_pipeline(tmp_path_factory, "porous")


@pytest.fixture(scope="session")
def vof_row4_divergent(tmp_path_factory: pytest.TempPathFactory) -> BuiltCase:
    """The ``divergent`` overlay (non-solenoidal ``0/U``), through the pipeline."""
    return _run_pipeline(tmp_path_factory, "divergent")
