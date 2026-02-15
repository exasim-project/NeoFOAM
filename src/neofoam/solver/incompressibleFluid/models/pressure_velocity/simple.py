# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

from typing import Any

from neofoam.framework.context import FieldUpdates

from .base import register_pressure_velocity_algorithm


@register_pressure_velocity_algorithm("SIMPLE")
class SimplePressureVelocityAlgorithm:
    def build(self) -> list[Any]:
        raise NotImplementedError("SIMPLE pressure-velocity model is not implemented")

    def inner_loop(self, _ctx: Any) -> bool:
        raise NotImplementedError("SIMPLE pressure-velocity model is not implemented")

    def momentum(self, **_kwargs: Any) -> FieldUpdates:
        raise NotImplementedError("SIMPLE pressure-velocity model is not implemented")

    def continuity(self, **_kwargs: Any) -> FieldUpdates:
        raise NotImplementedError("SIMPLE pressure-velocity model is not implemented")
