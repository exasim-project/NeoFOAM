# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The vofRow4 case, its variants as casebuild steps, and one executed pipeline.

**The case.** ``cases/vofRow4/common`` is the one full case every VoF test runs
on: a unit cube cut into 4 cells along x (cell volume 0.25, x-face area 1)
carrying the interFoam damBreak physical properties (``rho1 = 1000``,
``rho2 = 1``, ``g = (0 -9.81 0)``) — so every field the pipeline builds has an
exact, hand-derivable value. Every variant is composed onto it with
:func:`build_case`, which is ``neofoam.tooling.casebuild``'s
``from_template(common) | step | step …`` with the destination filled in.

A variant that differs only in dictionary *entries* is a
:data:`~neofoam.tooling.casebuild.Step` — :func:`alpha_controls`,
:func:`iso_advector`, :func:`moving`, or a plain ``patch(...)`` — written where
the test that needs it can be read. A variant that brings whole *files*
(a mesh, a ``0/`` field, a ``constant/`` dictionary) is still a checked-in
overlay directory under ``cases/vofRow4/``, applied with :func:`overlay`:

* ``href`` — a non-default ``constant/hRef`` (0.3), for the one test that needs
  ``hRef`` actually read from disk rather than defaulted
  (``test_create_fields.py``);
* ``moving`` — a ``constant/dynamicMeshDict`` (solid-body oscillation along
  gravity) so the mesh *selection* goes through ``dynamicFvMesh::New``
  (``test_dynamic_mesh.py``); :func:`moving` applies it together with the
  ``correctPhi`` switch the moving tests fork on;
* ``divergent`` — an inlet that feeds half of what the interior carries, so
  ``createPhi(U)`` is *not* divergence-free and ``initCorrectPhi.H`` has
  something to do (``test_flux_correction.py``);
* ``porous`` — ``constant/porosityProperties`` + ``0/porosity``, so
  ``Foam::isoAdvection`` finds a porosity field in the registry
  (``models/alpha_advection/models/test_iso_advector.py``); combined with
  :func:`iso_advector`, the only scheme that reads it;
* ``duct`` — the row re-cut as an open duct with a ``source`` cellZone, shared
  by the ``source`` (``system/fvOptions``) and ``limited``
  (``constant/fvOptions``) layers (``test_fv_options.py``);
* ``rotor`` — the block named so blockMesh makes a cellZone, plus
  ``constant/MRFProperties`` (``test_mrf.py``).

``models/pressure_velocity/cases/{open,closed,closedRefPoint}`` are further
overlays of the same base, kept next to the tests that own them.

``cases/pimple`` is a different, deliberately bare case: a ``system/fvSolution``
carrying nothing but a ``PIMPLE`` dict, which the control-factory and
turbulence-schedule tests read by chdir'ing into a variant of it
(:func:`pimple_case`).

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
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from typing import Any

import pybFoam as pyf
import pytest

from neofoam.io import DictFile
from neofoam.tooling.casebuild import CaseDir, Step, from_template, patch
from neofoam.tooling.casebuild.pipeline import pipe

_HERE = Path(__file__).parent
_WORKER = _HERE / "_create_fields_worker.py"

#: The vofRow4 layer directory: ``common`` plus one sub-directory per overlay.
VOF_ROW4 = _HERE / "cases" / "vofRow4"

#: The bare PIMPLE-dict case the control-factory tests fork (see :func:`pimple_case`).
PIMPLE_CASE = _HERE / "cases" / "pimple"

#: The address of fvSolution's alpha solver dict. ``patch`` addresses sub-dicts
#: by splitting a dotted key, which cannot name the regex key ``"alpha.water.*"``
#: — so the steps below spell the address as a tuple instead.
_ALPHA_DICT = ("solvers", "alpha.water.*")


def overlay(*layers: Path) -> Step:
    """Copy checked-in overlay directories over the case, in order.

    For the variants that bring whole *files*; a variant that only changes
    dictionary entries is a :func:`~neofoam.tooling.casebuild.patch` (or one of
    the steps below) and has no directory at all.
    """

    def step(case: CaseDir) -> None:
        for layer in layers:
            shutil.copytree(layer, case.path, dirs_exist_ok=True)

    return step


def alpha_controls(**entries: object) -> Step:
    """Set entries of ``system/fvSolution``'s ``solvers."alpha.water.*"`` dict."""

    def step(case: CaseDir) -> None:
        dict_file = DictFile(case.path / "system" / "fvSolution")
        for key, value in entries.items():
            dict_file.set((*_ALPHA_DICT, key), value)
        dict_file.write()

    return step


def iso_advector(*, n_alpha_sub_cycles: int = 1) -> Step:
    """Switch the case to geometric VoF (interIsoFoam), with isoAdvector's controls.

    ``Foam::isoAdvection`` reads them at construction via
    ``mesh.solverDict("alpha.water")``; ``cAlpha`` is unused by it but
    ``interfaceProperties`` reads it at mixture construction, so the base case's
    entry is left in place.
    """
    controls = alpha_controls(
        isoFaceTol=1e-6,
        surfCellTol=1e-6,
        nAlphaBounds=3,
        snapTol=1e-12,
        clip=True,
        reconstructionScheme="isoAlpha",
        nAlphaSubCycles=n_alpha_sub_cycles,
    )
    select = patch("system/fvSolution", advectionScheme="isoAdvector")

    def step(case: CaseDir) -> None:
        select(case)
        controls(case)

    return step


def moving(*, correct_phi: bool = False) -> Step:
    """The oscillating-mesh overlay plus ``createDyMControls.H``'s correctPhi switch.

    ``correctPhi no`` is the default here because the start-up/post-move flux
    projection is a separate concern (9 of the 17 moving tutorials switch it off
    too); the ``correct_phi=True`` fork is the twin that asks for it.
    """
    layer = overlay(VOF_ROW4 / "moving")
    switch = patch("system/fvSolution", **{"PIMPLE.correctPhi": correct_phi})

    def step(case: CaseDir) -> None:
        layer(case)
        switch(case)

    return step


#: Drop the closed box's pressure reference — for the duct variants, which have
#: a fixed-pressure outlet and so need none.
no_pressure_reference = patch("system/fvSolution", remove=["PIMPLE.pRefCell", "PIMPLE.pRefValue"])


def build_case(dest: Path, *steps: Step) -> Path:
    """Materialize ``cases/vofRow4/common`` plus *steps* at *dest*.

    Use it instead of ``shutil.copytree`` of the checked-in case (TEST_STYLE
    rule 4: never mutate the checked-in files)::

        case = build_case(tmp_path / "case", moving(), alpha_controls(cAlpha=0))
    """
    return pipe(from_template(VOF_ROW4 / "common"), *steps).build_at(dest).path


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

    def written_boundary(self, field: str, patch_name: str) -> Any:
        """The written ``boundaryField/<patch>`` sub-dictionary."""
        return self._written(field).subDict("boundaryField").subDict(patch_name)


def _run_pipeline(tmp_path_factory: pytest.TempPathFactory, name: str, *steps: Step) -> BuiltCase:
    """Compose ``vofRow4/common`` (+ *steps*), mesh it and run the staged init."""
    case = build_case(tmp_path_factory.mktemp(name) / "case", *steps)
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
    return _run_pipeline(tmp_path_factory, "vofRow4")


@pytest.fixture(scope="session")
def vof_row4_href(tmp_path_factory: pytest.TempPathFactory) -> BuiltCase:
    """The ``href`` overlay (``constant/hRef`` = 0.3), through the pipeline."""
    return _run_pipeline(tmp_path_factory, "vofRow4Href", overlay(VOF_ROW4 / "href"))


@pytest.fixture(scope="session")
def vof_row4_moving(tmp_path_factory: pytest.TempPathFactory) -> BuiltCase:
    """The moving case (has a ``constant/dynamicMeshDict``), through the pipeline."""
    return _run_pipeline(tmp_path_factory, "vofRow4Moving", moving())


@pytest.fixture(scope="session")
def vof_row4_porous(tmp_path_factory: pytest.TempPathFactory) -> BuiltCase:
    """The ``porous`` overlay under isoAdvector, through the pipeline."""
    return _run_pipeline(
        tmp_path_factory, "vofRow4Porous", overlay(VOF_ROW4 / "porous"), iso_advector()
    )


@pytest.fixture(scope="session")
def vof_row4_divergent(tmp_path_factory: pytest.TempPathFactory) -> BuiltCase:
    """The ``divergent`` overlay (non-solenoidal ``0/U``), through the pipeline."""
    return _run_pipeline(tmp_path_factory, "vofRow4Divergent", overlay(VOF_ROW4 / "divergent"))


#: The PIMPLE dicts the control-factory / turbulence-schedule tests read, each a
#: diff of ``cases/pimple`` (nOuterCorrectors 3, nCorrectors 1,
#: nNonOrthogonalCorrectors 0, momentumPredictor no, turbOnFinalIterOnly absent
#: — interFoam/RAS/damBreakPorousBaffle's shape) as (entries to set, keys to
#: remove).
PIMPLE_VARIANTS: dict[str, tuple[dict[str, object], list[str]]] = {
    # The base itself: every interFoam tutorial leaves turbOnFinalIterOnly out,
    # so pimpleControl::read()'s ``true`` default applies.
    "outer3": ({}, []),
    # The same case with the one key that puts the turbulence correction back
    # into every outer corrector.
    "turb_every_outer": ({"turbOnFinalIterOnly": False}, []),
    # No nCorrectors: the factory must default to 2 (the PISO pressure
    # correction needs at least two corrector iterations).
    "defaults": ({"nOuterCorrectors": 1}, ["nCorrectors"]),
    # All PIMPLE control keys set explicitly (damBreak-style values).
    "full": (
        {
            "nOuterCorrectors": 2,
            "nCorrectors": 3,
            "nNonOrthogonalCorrectors": 1,
            "turbCorr": False,
        },
        [],
    ),
    # Deliberate misconfiguration: the negative corrector counts are the
    # frozenFlow idiom, but frozenFlow is NOT set (defaults to no). The factory
    # must still build a PimpleControl and let its ge=1/ge=0 bounds reject the
    # -1s, so a genuine typo is surfaced rather than silently swallowed.
    "negative_correctors": (
        {"nCorrectors": -1, "nNonOrthogonalCorrectors": -1},
        ["nOuterCorrectors"],
    ),
}


@pytest.fixture(scope="session")
def pimple_case(tmp_path_factory: pytest.TempPathFactory) -> Callable[[str], Path]:
    """Build a named :data:`PIMPLE_VARIANTS` fork of ``cases/pimple``; return its path."""

    def build(name: str) -> Path:
        entries, removals = PIMPLE_VARIANTS[name]
        return (
            (
                from_template(PIMPLE_CASE)
                | patch(
                    "system/fvSolution",
                    {f"PIMPLE.{key}": value for key, value in entries.items()},
                    remove=[f"PIMPLE.{key}" for key in removals],
                )
            )
            .build_at(tmp_path_factory.mktemp(f"pimple_{name}") / "case")
            .path
        )

    return build
