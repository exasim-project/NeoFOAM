# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Tests for the incompressibleFluid PIMPLE control factory.

The factory reads the ``PIMPLE`` (or, for pisoFoam cases, ``PISO``) subdict of
``system/fvSolution`` from the cwd; each test chdirs into a real OpenFOAM case
directory under ``cases/`` (per the convention that dict-reading tests load
real case files).

The block is read through the typed config
(:mod:`neofoam.foam.algorithm_configs`); ``load_pimple_config`` is the seam that
picks which of the two spellings the case ships and pins the one PISO rule the
schema cannot express — a ``PISO`` block is a single-outer-loop PIMPLE, so
``nOuterCorrectors`` is 1 whatever the block says.
"""

import shutil
from pathlib import Path

import pytest

pytest.importorskip("pybFoam")

from pybFoam import dictionary  # noqa: E402

from neofoam.algorithms.solution_loop.control import PimpleControl  # noqa: E402
from neofoam.foam.algorithm_configs import PisoAlgorithmConfig  # noqa: E402
from neofoam.solver.incompressibleFluid.models.pressure_velocity.control_factory import (  # noqa: E402
    create_pimple_control,
    load_pimple_config,
)

_CASES = Path(__file__).parent / "cases"


def test_piso_block_maps_onto_pimple_control(monkeypatch: pytest.MonkeyPatch) -> None:
    """A pisoFoam PISO block builds a PimpleControl (nOuterCorrectors fixed to 1)."""
    monkeypatch.chdir(_CASES / "piso_cavity")
    control = create_pimple_control({})
    assert isinstance(control, PimpleControl)
    assert control.nOuterCorrectors == 1
    assert control.nCorrectors == 2
    assert control.nNonOrthogonalCorrectors == 0
    assert control.momentumPredictor() is True


def test_a_piso_case_is_read_through_the_piso_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """The block the case ships picks the config class bound to it."""
    monkeypatch.chdir(_CASES / "piso_cavity")
    config = load_pimple_config()
    assert isinstance(config, PisoAlgorithmConfig)
    assert config.nCorrectors == 2


def test_n_outer_correctors_stays_one_even_if_the_piso_block_asks_for_more(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PISO has no outer loop: the key is pinned, not read.

    Derived from the real piso_cavity fvSolution via pybFoam's own
    ``read``/``set``/``write`` (TEST_STYLE rule 3), since no pisoFoam tutorial
    writes ``nOuterCorrectors`` into a PISO block.
    """
    shutil.copytree(_CASES / "piso_cavity", tmp_path, dirs_exist_ok=True)
    fv_solution_path = tmp_path / "system" / "fvSolution"
    fv_solution = dictionary.read(str(fv_solution_path))
    fv_solution.subDict("PISO").set("nOuterCorrectors", 7)
    fv_solution.write(str(fv_solution_path))

    monkeypatch.chdir(tmp_path)
    assert load_pimple_config().nOuterCorrectors == 1
    assert create_pimple_control({}).nOuterCorrectors == 1


def test_pimple_block_still_read(monkeypatch: pytest.MonkeyPatch) -> None:
    """The existing PIMPLE path keeps reading all keys."""
    monkeypatch.chdir(_CASES / "pimple_full")
    control = create_pimple_control({})
    assert control.nOuterCorrectors == 2
    assert control.nCorrectors == 3
    assert control.nNonOrthogonalCorrectors == 1
    assert control.momentumPredictor() is False
    assert control.turbCorr() is False
    # pimpleControl::read() defaults for the two keys the dict leaves out
    assert control.turbOnFinalIterOnly is True
    assert control.finalOnLastPimpleIterOnly is False


def test_turbulence_correction_switches_are_read(monkeypatch: pytest.MonkeyPatch) -> None:
    """`turbOnFinalIterOnly no` opens the correction gate in every outer iteration."""
    monkeypatch.chdir(_CASES / "pimple_turb_every_iteration")
    control = create_pimple_control({})
    assert control.turbOnFinalIterOnly is False
    assert control.finalOnLastPimpleIterOnly is True

    corrections = []
    while control.loop():
        corrections.append(control.turbCorr())
    assert corrections == [True, True, True]
