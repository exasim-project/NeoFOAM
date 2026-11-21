# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""A fake coupling plugin: a mapping and an acceleration that register as it is imported."""

from typing import Any, Literal

from pydantic import Field

from neofoam.framework.coupling import ConstantRelaxation, CouplingAcceleration, CouplingMapping

calls = {"fake-rbf": 0, "fake-iqn": 0}


@CouplingMapping.register
class RadialBasis(CouplingMapping):
    """The identity, counting its calls."""

    type: Literal["fake-rbf"] = "fake-rbf"
    radius: float = Field(gt=0)

    def bind(self, source: Any, target: Any) -> Any:
        def mapper(data: Any) -> Any:
            calls["fake-rbf"] += 1
            return data

        return mapper


@CouplingAcceleration.register
class QuasiNewton(ConstantRelaxation):
    """A constant relaxation, counting its calls."""

    type: Literal["fake-iqn"] = "fake-iqn"  # type: ignore[assignment]
    columns: int = Field(gt=0)

    def __call__(self, sent: Any, iteration: int) -> Any:
        calls["fake-iqn"] += 1
        return super().__call__(sent, iteration)
