# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

import pytest

from neofoam.mcp.registry import list_solver_names, resolve_solver


def test_list_solver_names_contains_incompressible_fluid() -> None:
    assert "incompressibleFluid" in list_solver_names()


def test_resolve_unknown_solver_raises_clear_error() -> None:
    with pytest.raises(ValueError) as exc:
        resolve_solver("nope")
    msg = str(exc.value)
    assert "nope" in msg and "incompressibleFluid" in msg


def test_resolve_known_solver_returns_spec() -> None:
    spec = resolve_solver("incompressibleFluid")
    assert spec.name  # SolverSpec has a name
