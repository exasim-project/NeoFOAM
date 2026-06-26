# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Spec for create_fields helpers — the optional-model ctx.models projection."""

import pytest

# create_fields imports pybFoam (argList/Time/fvMesh) at module top.
pytest.importorskip("pybFoam")

from neofoam.solver.incompressibleFluid.create_fields import _optional_models_by_name


class _FakeRuntime:
    """Stands in for a ModelRuntime: the projection only reads ``.name``."""

    def __init__(self, name: str) -> None:
        self.name = name


def test_optional_models_are_keyed_by_their_model_name() -> None:
    # The projection keys each detected runtime under its model name, which is
    # exactly what the interface owner-gate (owner.name in ctx.models) reads.
    cap = _FakeRuntime("maxDeltaT")
    cfl = _FakeRuntime("courant")
    assert _optional_models_by_name([cap, cfl]) == {"maxDeltaT": cap, "courant": cfl}


def test_no_optional_models_yields_an_empty_mapping() -> None:
    assert _optional_models_by_name([]) == {}
