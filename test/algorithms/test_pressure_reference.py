# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Spec for :class:`neofoam.algorithms.PressureReference`.

The payload every pressure-velocity algorithm publishes on ``ctx.models`` for
its corrector. It replaced a bare ``dict`` with string keys, so what is worth
pinning is exactly the contract the corrector relies on: three named attributes,
and immutability — the reference is computed once at initialisation and read on
every corrector pass of every time step, so an operation must not be able to
edit it in place.

Pure Python (no case, no OpenFOAM): the *values* come from ``Foam::setRefCell``
and are pinned per solver where that call happens — e.g.
``test/solver/incompressibleVoF/models/pressure_velocity/test_pressure_reference.py``.
"""

import dataclasses

import pytest

from neofoam.algorithms import PressureReference


def test_pressure_reference_carries_the_cell_the_value_and_the_flag() -> None:
    reference = PressureReference(cell=2, value=50.0, needs_ref=True)
    assert reference.cell == 2
    assert reference.value == 50.0
    assert reference.needs_ref is True


def test_pressure_reference_is_immutable() -> None:
    reference = PressureReference(cell=-1, value=0.0, needs_ref=False)
    with pytest.raises(dataclasses.FrozenInstanceError):
        reference.cell = 0  # type: ignore[misc]
