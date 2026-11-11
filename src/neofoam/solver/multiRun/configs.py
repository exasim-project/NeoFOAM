# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Case config of multiRun: how its regions are coupled (``system/regionCoupling.yaml``)."""

from functools import partial
from typing import Annotated, Any, Literal, Optional

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    SerializeAsAny,
    WithJsonSchema,
    field_validator,
    model_validator,
)

from neofoam.framework.coupling import (
    CouplingAcceleration,
    CouplingConvergence,
    CouplingMapping,
    CouplingNode,
    Exchange,
)
from neofoam.framework.coupling.comm_channel import DEFAULT_TIMEOUT
from neofoam.framework.coupling.kernels import default_mapping
from neofoam.framework.coupling.region import check_couplings
from neofoam.framework.types import Placement
from neofoam.io import YAML, BaseConfig, IOStrategy


def _created(family: Any, variable: str, value: Any) -> Any:
    """The strategy of ``family`` that ``value`` names: ``{type: ..., <option>: ...}`` or a type."""
    if isinstance(value, str):
        value = {"type": value}
    if not isinstance(value, dict):
        return value
    return getattr(family.create(**{variable: value}), variable)


def _schema(what: str) -> WithJsonSchema:
    """An open object: a plugin adds its own options."""
    return WithJsonSchema({"type": "object", "description": f"a {what}: its type and options"})


MappingConfig = Annotated[
    SerializeAsAny[CouplingMapping],
    BeforeValidator(partial(_created, CouplingMapping, "mapping")),
    _schema("mapping"),
]
AccelerationConfig = Annotated[
    SerializeAsAny[CouplingAcceleration],
    BeforeValidator(partial(_created, CouplingAcceleration, "acceleration")),
    _schema("acceleration"),
]
ConvergenceConfig = Annotated[
    SerializeAsAny[CouplingConvergence],
    BeforeValidator(partial(_created, CouplingConvergence, "convergence")),
    _schema("convergence measure"),
]


class ExchangeConfig(BaseModel):
    """One exchange: the port of each of its two regions; ``name`` defaults to theirs."""

    model_config = ConfigDict(extra="forbid")

    name: Optional[str] = None
    ports: dict[str, str] = Field(min_length=2, max_length=2)
    mapping: MappingConfig = Field(default_factory=default_mapping)


class NodeExchangeConfig(BaseModel):
    """An exchange of a coupling node and the acceleration of its data; a plain name has none."""

    model_config = ConfigDict(extra="forbid")

    name: str
    acceleration: Optional[AccelerationConfig] = None


class PlacementConfig(BaseModel):
    """Where a node's op sits in a region's graph, as a :class:`Placement`."""

    model_config = ConfigDict(extra="forbid")

    operations: Optional[list[str]] = None


class CouplingNodeConfig(BaseModel):
    """One coupling node of ``couplings``."""

    model_config = ConfigDict(extra="forbid")

    name: str
    scheme: Literal["serial", "parallel"] = CouplingNode.scheme
    maxIterations: int = Field(default=CouplingNode.max_iterations, gt=0)
    exchanges: list[NodeExchangeConfig] = Field(min_length=1)
    convergence: list[ConvergenceConfig] = Field(min_length=1)
    wraps: dict[str, PlacementConfig] = Field(default_factory=dict)

    @field_validator("exchanges", mode="before")
    @classmethod
    def _plain_names(cls, value: Any) -> Any:
        if isinstance(value, list):
            return [{"name": entry} if isinstance(entry, str) else entry for entry in value]
        return value

    def coupling_node(self) -> CouplingNode:
        """The node, with an acceleration of its own per exchange (they keep state)."""
        return CouplingNode(
            self.name,
            tuple(exchange.name for exchange in self.exchanges),
            max_iterations=self.maxIterations,
            scheme=self.scheme,
            convergence=tuple(self.convergence),
            acceleration={
                exchange.name: exchange.acceleration.model_copy(deep=True)
                for exchange in self.exchanges
                if exchange.acceleration is not None
            },
            placement={
                region: Placement(None if wraps.operations is None else tuple(wraps.operations))
                for region, wraps in self.wraps.items()
            },
        )


@IOStrategy(YAML("system/regionCoupling.yaml"))
class RegionCouplingConfig(BaseConfig):
    """``system/regionCoupling.yaml``: the exchanges between regions and the coupling nodes.

    ``couplings`` lists the nodes outer to inner, each exchange in one; checked on load.
    ``commChannelTimeout`` is in seconds, ``0`` or ``null`` without end. Without the file
    no region is coupled.

    Example::

        exchanges:
          - {name: wall, ports: {fluid: wall, solid: right}, mapping: nearest-neighbor}
        couplings:
          - name: thermal
            scheme: serial
            exchanges:
              - {name: wall, acceleration: {type: aitken, relaxation: 0.5}}
            convergence:
              - {type: relative, data: T, limit: 1.0e-5}
            wraps:
              fluid: {operations: [solve_T]}
              solid: {operations: [solve_T]}
    """

    model_config = ConfigDict(extra="forbid")

    exchanges: list[ExchangeConfig] = Field(min_length=1)
    couplings: list[CouplingNodeConfig] = Field(min_length=1)
    commChannelTimeout: Optional[float] = Field(default=DEFAULT_TIMEOUT, ge=0, allow_inf_nan=False)

    def coupling_exchanges(self) -> list[Exchange]:
        """The exchanges; a default name another exchange has is numbered (``a-b-2``)."""
        exchanges: list[Exchange] = []
        for config in self.exchanges:
            taken = [exchange.name for exchange in exchanges]
            name = config.name or _free("-".join(config.ports), taken)
            first, second = config.ports.items()
            exchanges.append(Exchange(name, (first, second), config.mapping))
        return exchanges

    def nodes(self) -> list[CouplingNode]:
        """The coupling nodes, outer to inner."""
        return [node.coupling_node() for node in self.couplings]

    @model_validator(mode="after")
    def _check(self) -> "RegionCouplingConfig":
        exchanges = self.coupling_exchanges()
        regions = list(
            dict.fromkeys(region for exchange in exchanges for region, _patch in exchange.ends)
        )
        check_couplings(regions, exchanges, self.nodes())
        return self


def _free(name: str, taken: list[str]) -> str:
    """``name``, numbered if ``taken`` has it."""
    free, number = name, 1
    while free in taken:
        number += 1
        free = f"{name}-{number}"
    return free
