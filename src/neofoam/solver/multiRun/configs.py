# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Case config of multiRun: the region solvers and how their regions are coupled."""

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

from neofoam.algorithms.solution_loop.config import TimeControlConfig
from neofoam.framework.coupling import (
    CouplingAcceleration,
    CouplingConvergence,
    CouplingMapping,
    CouplingNode,
    Link,
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


@IOStrategy(YAML("system/controlDict.yaml"))
class ControlDictConfig(TimeControlConfig):
    """``system/controlDict.yaml``: the time control of every region, and its solvers.

    ``regionSolvers`` maps each region to its solver, in stepping order; a region
    reads its own files under ``<region>/``.

    Example::

        endTime: 10.0
        deltaT: 1.0
        regionSolvers:
          fluid: myFluidSolver
          solid: mySolidSolver
    """

    # The i18n prefix labels the wizard's "add a region" row (neofoam.ui.form_schema).
    regionSolvers: dict[str, str] = Field(min_length=1, json_schema_extra={"i18n": "nf.region"})


class ExchangeConfig(BaseModel):
    """One exchange: the coupled patch of each of its two regions; ``name`` defaults to theirs."""

    model_config = ConfigDict(extra="forbid")

    name: Optional[str] = None
    sides: dict[str, str] = Field(min_length=2, max_length=2)
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
    """One coupling node; ``scheme`` and ``maxIterations`` default as a ``CouplingNode``'s."""

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

    ``couplings`` lists the nodes outer to inner, each exchange in exactly one;
    the file is checked as it loads (:func:`~neofoam.framework.coupling.region.check_couplings`).
    ``transportTimeout`` bounds a wait for a message in seconds; ``0`` or ``null``
    waits without end. Without the file no region is coupled.

    Example::

        exchanges:
          - {name: wall, sides: {fluid: wall, solid: right}, mapping: nearest-neighbor}
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
    transportTimeout: Optional[float] = Field(default=DEFAULT_TIMEOUT, ge=0, allow_inf_nan=False)

    def links(self) -> list[Link]:
        """One link per exchange; a default name another exchange has is numbered (``a-b-2``)."""
        links: list[Link] = []
        for exchange in self.exchanges:
            taken = [link.name for link in links]
            name = exchange.name or _free("-".join(exchange.sides), taken)
            first, second = exchange.sides.items()
            links.append(Link(name, (first, second), exchange.mapping))
        return links

    def nodes(self) -> list[CouplingNode]:
        """The coupling nodes, outer to inner."""
        return [node.coupling_node() for node in self.couplings]

    @model_validator(mode="after")
    def _check(self) -> "RegionCouplingConfig":
        links = self.links()
        regions = list(dict.fromkeys(region for link in links for region, _patch in link.ends))
        check_couplings(regions, links, self.nodes())
        return self


def _free(name: str, taken: list[str]) -> str:
    """``name``, numbered if ``taken`` has it."""
    free, number = name, 1
    while free in taken:
        number += 1
        free = f"{name}-{number}"
    return free
