# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Coupling plugins add their kernels to the registries through the entry-point group.

Each plugin is a fake entry point of ``neofoam.coupling`` (:mod:`plugins`); the
registries load them on their first read, once, next to the built-in kernels.
A name registered twice, or a plugin that fails, raises naming the plugin.
"""

from typing import Any

import pytest
from framework.coupling.plugins import installed

from neofoam.framework.coupling import (
    ACCELERATIONS,
    CONVERGENCES,
    ENTRY_POINT_GROUP,
    MAPPINGS,
    ConstantRelaxation,
    Mapping,
    mapping,
)


class _RadialBasis(Mapping):
    kind = "fake-rbf"


class _QuasiNewton(ConstantRelaxation):
    pass


def _precice(mappings: dict[str, Any], accelerations: dict[str, Any], _: dict[str, Any]) -> None:
    mappings[_RadialBasis.kind] = _RadialBasis
    accelerations["fake-iqn"] = _QuasiNewton


def test_plugin_kernels_join_the_registries_on_their_first_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with installed(monkeypatch, precice=_precice) as asked:
        assert asked == []

        assert isinstance(mapping("fake-rbf"), _RadialBasis)
        assert isinstance(ACCELERATIONS["fake-iqn"](0.5), _QuasiNewton)
        assert sorted(MAPPINGS) == ["fake-rbf", "identity", "nearest-neighbor"]
        assert sorted(ACCELERATIONS) == ["aitken", "constant", "fake-iqn"]
        assert sorted(CONVERGENCES) == ["relative"]
        assert asked == [ENTRY_POINT_GROUP]

    assert "fake-rbf" not in MAPPINGS


def test_a_plugin_that_registers_a_built_in_name_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def register(mappings: dict[str, Any], *_: dict[str, Any]) -> None:
        mappings["identity"] = _RadialBasis

    with installed(monkeypatch, precice=register):
        with pytest.raises(
            ValueError,
            match=r"plugin 'precice' \(fake_coupling_plugin_precice:register\) and NeoFOAM "
            r"\(built-in\) both register the mapping 'identity'",
        ):
            mapping("identity")


def test_two_plugins_that_register_one_name_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    def register(_: dict[str, Any], accelerations: dict[str, Any], __: dict[str, Any]) -> None:
        accelerations["iqn-ils"] = _QuasiNewton

    with installed(monkeypatch, other=register, precice=_precice, second=register):
        with pytest.raises(
            ValueError,
            match=r"plugin 'second' \(fake_coupling_plugin_second:register\) and plugin "
            r"'other' \(fake_coupling_plugin_other:register\) both register the acceleration "
            r"'iqn-ils'",
        ):
            "iqn-ils" in ACCELERATIONS  # noqa: B015
        # nothing of the plugins was added
        assert "fake-rbf" not in dict.keys(MAPPINGS)


def test_a_plugin_that_fails_to_import_raises_with_its_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with installed(monkeypatch, broken="neofoam_no_such_plugin:register", precice=_precice):
        for _ in range(2):  # not marked as loaded: every read raises
            with pytest.raises(
                ImportError,
                match=r"coupling plugin 'broken' \(neofoam_no_such_plugin:register\) of "
                "entry-point group 'neofoam.coupling' failed to load: No module named",
            ):
                mapping("identity")


def test_a_plugin_whose_register_fails_raises_with_its_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def register(*_: dict[str, Any]) -> None:
        raise RuntimeError("preCICE not found")

    with installed(monkeypatch, precice=register):
        with pytest.raises(ImportError, match="plugin 'precice' .* failed to register: preCICE"):
            sorted(CONVERGENCES)
