# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Coupling plugins add their strategies to the families through the entry-point group.

Each plugin is a fake entry point of ``neofoam.coupling`` (:mod:`plugins`) naming
a module that registers its strategies as it is imported; :func:`load_plugins`
imports them, once, next to the built-in strategies. A strategy is selected by
its ``type``, its options validated; an unknown type or option raises listing
what is registered. A type registered twice, or a plugin that fails, raises
naming the plugin.
"""

import threading
from pathlib import Path

import pytest
from framework.coupling.plugins import installed

from neofoam.core.plugin_system import PluginSystem
from neofoam.framework.coupling import (
    CouplingAcceleration,
    CouplingConvergence,
    CouplingMapping,
    load_plugins,
    strategy,
)
from neofoam.framework.coupling.kernels import ENTRY_POINT_GROUP

PRECICE = """
import time
from typing import Literal

from pydantic import Field

from neofoam.framework.coupling import ConstantRelaxation, CouplingAcceleration, CouplingMapping

time.sleep({delay})


@CouplingMapping.register
class RadialBasis(CouplingMapping):
    type: Literal["fake-rbf"] = "fake-rbf"
    radius: float = Field(gt=0)
    polynomial: bool = True


@CouplingAcceleration.register
class QuasiNewton(ConstantRelaxation):
    type: Literal["fake-iqn"] = "fake-iqn"
    columns: int = 10
"""


def _precice(delay: float = 0.0) -> str:
    return PRECICE.format(delay=delay)


def test_plugin_strategies_join_the_families_once_loaded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with installed(monkeypatch, tmp_path, precice=_precice()) as asked:
        with pytest.raises(ValueError, match="unknown mapping type 'fake-rbf'"):
            strategy(CouplingMapping, {"type": "fake-rbf", "radius": 0.5})

        load_plugins()
        load_plugins()

        rbf = strategy(CouplingMapping, {"type": "fake-rbf", "radius": 0.5, "polynomial": False})
        assert (type(rbf).__name__, rbf.radius, rbf.polynomial) == ("RadialBasis", 0.5, False)
        iqn = strategy(CouplingAcceleration, {"type": "fake-iqn", "relaxation": 0.3, "columns": 4})
        assert (type(iqn).__name__, iqn.relaxation, iqn.columns) == ("QuasiNewton", 0.3, 4)
        assert asked == [ENTRY_POINT_GROUP]

    with pytest.raises(ValueError, match="unknown mapping type 'fake-rbf'"):
        strategy(CouplingMapping, "fake-rbf")


def test_an_unknown_type_raises_listing_the_registered_types(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with installed(monkeypatch, tmp_path, precice=_precice()):
        load_plugins()
        with pytest.raises(
            ValueError,
            match=r"unknown acceleration type 'aitkn'; registered acceleration types: "
            r"\['aitken', 'constant', 'fake-iqn'\]",
        ):
            strategy(CouplingAcceleration, {"type": "aitkn", "relaxation": 0.5})
        with pytest.raises(
            ValueError,
            match=r"convergence \{'data': 'T'\} has no type; registered convergence types: "
            r"\['relative'\]",
        ):
            strategy(CouplingConvergence, {"data": "T"})


@pytest.mark.parametrize(
    ("options", "error"),
    [
        (
            {"type": "fake-rbf", "radius": 0.5, "radios": 1.0},
            r"mapping 'fake-rbf': unknown option 'radios'; its options: \['radius', 'polynomial'\]",
        ),
        (
            {"type": "fake-rbf"},
            r"mapping 'fake-rbf': missing option 'radius'; its options: \['radius', 'polynomial'\]",
        ),
        (
            {"type": "fake-rbf", "radius": -1.0},
            r"mapping 'fake-rbf': radius: Input should be greater than 0; its options",
        ),
    ],
    ids=["unknown", "missing", "invalid"],
)
def test_a_wrong_option_raises_naming_the_options_of_the_strategy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, options: dict[str, object], error: str
) -> None:
    with installed(monkeypatch, tmp_path, precice=_precice()):
        load_plugins()
        with pytest.raises(ValueError, match=error):
            strategy(CouplingMapping, options)


def test_concurrent_first_loads_from_several_region_threads_see_every_plugin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    results: dict[int, object] = {}
    errors: list[BaseException] = []
    start = threading.Barrier(8)

    def region(index: int) -> None:
        try:
            start.wait()
            load_plugins()
            results[index] = strategy(CouplingMapping, {"type": "fake-rbf", "radius": 1.0})
        except BaseException as error:  # reported in the test thread
            errors.append(error)

    # the plugin takes a while to import: every other thread arrives while it loads
    with installed(monkeypatch, tmp_path, precice=_precice(delay=0.2)) as asked:
        threads = [threading.Thread(target=region, args=(index,)) for index in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert errors == []
        assert sorted(results) == list(range(8))
        assert {type(mapping).__name__ for mapping in results.values()} == {"RadialBasis"}
        assert asked == [ENTRY_POINT_GROUP]


def test_a_plugin_that_registers_a_built_in_type_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    clash = """
from typing import Literal
from neofoam.framework.coupling import CouplingMapping

@CouplingMapping.register
class Other(CouplingMapping):
    type: Literal["identity"] = "identity"
"""
    with installed(monkeypatch, tmp_path, precice=clash):
        with pytest.raises(
            ImportError,
            match=r"coupling plugin 'precice' \(fake_coupling_plugin_precice\) of entry-point "
            r"group 'neofoam.coupling' failed to load: plugin family 'CouplingMapping': "
            r"type='identity' is already registered by "
            r"neofoam.framework.coupling.kernels.IdentityMapping; "
            r"fake_coupling_plugin_precice.Other cannot register the same type",
        ):
            load_plugins()
        registry = PluginSystem.get_registered("CouplingMapping")
        assert registry is not None
        assert "Other" not in registry.get_plugin_names()


def test_a_plugin_that_fails_to_import_raises_with_its_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with installed(monkeypatch, tmp_path, broken="neofoam_no_such_plugin", precice=_precice()):
        for _ in range(2):  # not marked as loaded: every call raises
            with pytest.raises(
                ImportError,
                match=r"coupling plugin 'broken' \(neofoam_no_such_plugin\) of "
                "entry-point group 'neofoam.coupling' failed to load: No module named",
            ):
                load_plugins()
