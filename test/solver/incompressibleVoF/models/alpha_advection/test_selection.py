# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Tests for alpha-advection scheme selection / factory (``selection.py``).

``model_name`` is duck-typed (no file IO), so it is exercised directly with
``SimpleNamespace``. ``select_advection_scheme`` and ``select_from_case`` are
exercised against the two real, registered schemes (``MULES``/``isoAdvector``)
using real ``system/fvSolution`` files:

- ``cases/damBreak_mules`` is ``$FOAM_TUTORIALS/multiphase/interFoam/laminar/
  damBreak/damBreak/system/fvSolution`` (no ``advectionScheme`` key — the
  MULES default path).
- ``cases/damBreak_isoAdvector`` is this repo's own
  ``tutorials/damBreak_isoAdvector/system/fvSolution`` (the same tutorial
  ``test_damBreak_isoAdvector_comparison.py`` runs), which carries an explicit
  ``advectionScheme isoAdvector;`` plus isoAdvector controls
  (``reconstructionScheme`` etc.) in its ``solvers."alpha.water.*"`` sub-dict.

The "no explicit key" and "unknown scheme" cases are derived from these two real
fvSolution files at test time via pybFoam's own
``dictionary.read``/``set``/``remove``/``write`` (TEST_STYLE rule 3: vary
content through the format's reader/writer, never text-patch a copy) into
``tmp_path`` (TEST_STYLE rule 4: never mutate the checked-in case).
"""

import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from pybFoam import dictionary

from neofoam.solver.incompressibleVoF.models.alpha_advection.selection import (
    model_name,
    select_advection_scheme,
    select_from_case,
)

_CASES = Path(__file__).parent / "cases"


def test_model_name_defaults_to_mules_when_attribute_absent() -> None:
    """No ``advectionScheme`` attribute on the config -> the MULES default."""
    assert model_name(SimpleNamespace()) == "MULES"


def test_model_name_returns_the_configured_advection_scheme() -> None:
    assert model_name(SimpleNamespace(advectionScheme="isoAdvector")) == "isoAdvector"


@pytest.mark.parametrize("name", ["MULES", "isoAdvector"])
def test_select_advection_scheme_returns_the_matching_registered_spec(
    name: str,
) -> None:
    spec = select_advection_scheme(name)
    assert spec.name == name


def test_select_advection_scheme_raises_for_an_unknown_name() -> None:
    """An unregistered name raises instead of silently falling back to MULES."""
    with pytest.raises(ValueError, match="Unknown advectionScheme 'bogusScheme'"):
        select_advection_scheme("bogusScheme")


def test_select_from_case_defaults_to_mules_from_the_real_damBreak_case(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The interFoam damBreak fvSolution has no ``advectionScheme`` key."""
    monkeypatch.chdir(_CASES / "damBreak_mules")
    spec = select_from_case()
    assert spec.name == "MULES"


def test_select_from_case_selects_isoadvector_from_the_real_case(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The damBreak_isoAdvector fvSolution has ``advectionScheme isoAdvector;``."""
    monkeypatch.chdir(_CASES / "damBreak_isoAdvector")
    spec = select_from_case()
    assert spec.name == "isoAdvector"


def test_select_from_case_defaults_to_mules_when_fvSolution_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No ``system/fvSolution`` at all (unreadable) also falls back to MULES."""
    monkeypatch.chdir(tmp_path)
    spec = select_from_case()
    assert spec.name == "MULES"


def test_select_from_case_raises_for_an_unknown_scheme_in_the_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bogus ``advectionScheme``, injected into a copy of the real MULES
    fvSolution via pybFoam's own reader/writer, raises."""
    shutil.copytree(_CASES / "damBreak_mules", tmp_path, dirs_exist_ok=True)
    fv_solution_path = tmp_path / "system" / "fvSolution"
    fv_solution = dictionary.read(str(fv_solution_path))
    fv_solution.set("advectionScheme", "bogusScheme")
    fv_solution.write(str(fv_solution_path))

    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="Unknown advectionScheme 'bogusScheme'"):
        select_from_case()


def test_select_from_case_selects_isoadvector_from_reconstructionScheme_without_the_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Upstream interIsoFoam tutorials never declare ``advectionScheme`` — with
    the key stripped from a copy of the real isoAdvector fvSolution, the
    ``reconstructionScheme``/``isoFaceTol``/``surfCellTol``/``nAlphaBounds``
    controls in ``solvers."alpha.water.*"`` alone must still select isoAdvector."""
    shutil.copytree(_CASES / "damBreak_isoAdvector", tmp_path, dirs_exist_ok=True)
    fv_solution_path = tmp_path / "system" / "fvSolution"
    fv_solution = dictionary.read(str(fv_solution_path))
    fv_solution.remove("advectionScheme")
    fv_solution.write(str(fv_solution_path))

    monkeypatch.chdir(tmp_path)
    spec = select_from_case()
    assert spec.name == "isoAdvector"


def test_select_from_case_explicit_advectionScheme_overrides_the_alpha_controls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit ``advectionScheme MULES;`` wins even though the ``alpha.water``
    solver sub-dict carries isoAdvector controls."""
    shutil.copytree(_CASES / "damBreak_isoAdvector", tmp_path, dirs_exist_ok=True)
    fv_solution_path = tmp_path / "system" / "fvSolution"
    fv_solution = dictionary.read(str(fv_solution_path))
    fv_solution.set("advectionScheme", "MULES")
    fv_solution.write(str(fv_solution_path))

    monkeypatch.chdir(tmp_path)
    spec = select_from_case()
    assert spec.name == "MULES"
