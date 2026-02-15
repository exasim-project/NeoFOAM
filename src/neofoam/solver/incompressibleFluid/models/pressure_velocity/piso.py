# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""PISO pressure-velocity algorithm model (not implemented)."""

from typing import Any

from neofoam.framework.context import FieldUpdates

from ..incompressibleFluidModel import Model
from .base import pressureVelocityAlgorithm

piso = Model("Piso").register_with(pressureVelocityAlgorithm)


def build() -> list[Any]:
	raise NotImplementedError("PISO pressure-velocity model is not implemented")


def inner_loop(_ctx: Any) -> bool:
	raise NotImplementedError("PISO pressure-velocity model is not implemented")


def momentum(**_kwargs: Any) -> FieldUpdates:
	raise NotImplementedError("PISO pressure-velocity model is not implemented")


def continuity(**_kwargs: Any) -> FieldUpdates:
	raise NotImplementedError("PISO pressure-velocity model is not implemented")
