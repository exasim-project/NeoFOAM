# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""How a case couples its regions, ``system/coupling.yaml``: :class:`CouplingConfig`."""

from functools import partial
from typing import Annotated, Any, Literal, Optional, Sequence

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

from neofoam.framework.types import Placement
from neofoam.io import YAML, BaseConfig, IOStrategy

from .comm.channel import DEFAULT_TIMEOUT, CommChannel
from .exchanges import (
    CouplingLoop,
    Exchange,
    ParticipantExchanges,
    build_participant_exchanges,
    check_exchanges,
    check_regions,
)
from .field_port import FieldPort
from .strategies.acceleration import CouplingAcceleration
from .strategies.convergence import CouplingConvergence
from .strategies.mapping import CouplingMapping, default_mapping


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


class FieldPortConfig(BaseModel):
    """A field port of ``ports:``: the region's ``field`` on ``patch`` (none: its cells).

    ``direction`` ``out`` reads it (the default), ``in`` writes it, ``inout`` both.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    field: str
    patch: Optional[str] = None
    direction: Literal["out", "in", "inout"] = "out"

    def field_port(self) -> FieldPort:
        return FieldPort(self.name, self.field, self.patch, self.direction)


class LoopExchangeConfig(BaseModel):
    """An exchange of a coupling loop and the acceleration of its data; a plain name has none."""

    model_config = ConfigDict(extra="forbid")

    name: str
    acceleration: Optional[AccelerationConfig] = None


class PlacementConfig(BaseModel):
    """Where a loop's op sits in a region's graph, as a :class:`Placement`."""

    model_config = ConfigDict(extra="forbid")

    operations: Optional[list[str]] = None


class CouplingLoopConfig(BaseModel):
    """One coupling loop of ``couplings``."""

    model_config = ConfigDict(extra="forbid")

    name: str
    scheme: Literal["serial", "parallel"] = CouplingLoop.scheme
    maxIterations: int = Field(default=CouplingLoop.max_iterations, gt=0)
    exchanges: list[LoopExchangeConfig] = Field(min_length=1)
    convergence: list[ConvergenceConfig] = Field(min_length=1)
    wraps: dict[str, PlacementConfig] = Field(default_factory=dict)

    @field_validator("exchanges", mode="before")
    @classmethod
    def _plain_names(cls, value: Any) -> Any:
        if isinstance(value, list):
            return [{"name": entry} if isinstance(entry, str) else entry for entry in value]
        return value

    def coupling_loop(self) -> CouplingLoop:
        """The loop, with an acceleration of its own per exchange (they keep state)."""
        return CouplingLoop(
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


@IOStrategy(YAML("system/coupling.yaml"))
class CouplingConfig(BaseConfig):
    """``system/coupling.yaml``: field ports, the exchanges and the coupling loops.

    ``ports`` lists each region's field ports, which an exchange names like the
    ports of its coupled patches. ``couplings`` lists the loops outer to inner,
    each exchange in one; checked on load.
    ``commChannelTimeout`` is in seconds, ``0`` or ``null`` without end. Without the file
    no region is coupled.

    Example::

        ports:
          fluid:
            - {name: T_out, field: T, patch: outlet}   # direction: out
        exchanges:
          - {name: wall, ports: {fluid: wall, solid: right}, mapping: nearest-neighbor}
          - {name: mean_T, ports: {fluid: T_out, controller: T_meas}, mapping: area-mean}
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

    ports: dict[str, list[FieldPortConfig]] = Field(default_factory=dict)
    exchanges: list[ExchangeConfig] = Field(min_length=1)
    couplings: list[CouplingLoopConfig] = Field(min_length=1)
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

    def field_ports(self) -> dict[str, list[FieldPort]]:
        """Each region's field ports."""
        return {
            region: [port.field_port() for port in ports] for region, ports in self.ports.items()
        }

    def loops(self) -> list[CouplingLoop]:
        """The coupling loops, outer to inner."""
        return [loop.coupling_loop() for loop in self.couplings]

    def participant_exchanges(
        self, regions: Sequence[str], comm_channel: Optional[CommChannel]
    ) -> dict[str, ParticipantExchanges]:
        """Each region's coupling, as :func:`participant_exchanges` builds it from the file.

        The file was checked as it loaded; only that ``regions`` (the case's, in stepping
        order) hold every region an exchange names is left to check.
        """
        exchanges = self.coupling_exchanges()
        check_regions(regions, exchanges)
        return build_participant_exchanges(
            regions, exchanges, comm_channel, self.loops(), self.field_ports()
        )

    @model_validator(mode="after")
    def _check(self) -> "CouplingConfig":
        exchanges = self.coupling_exchanges()
        regions = list(
            dict.fromkeys(region for exchange in exchanges for region, _patch in exchange.ends)
        )
        check_exchanges(regions, exchanges, self.loops(), self.field_ports())
        return self


def _free(name: str, taken: list[str]) -> str:
    """``name``, numbered if ``taken`` has it."""
    free, number = name, 1
    while free in taken:
        number += 1
        free = f"{name}-{number}"
    return free


def comm_channel_timeout() -> Optional[float]:
    """Seconds a region waits for a message, the case's ``commChannelTimeout``; ``None``: no end.

    The case is the working directory's; without ``system/coupling.yaml`` the
    comm channel's default.

    Example::

        InProcessCommChannel(timeout=comm_channel_timeout())
    """
    if not CouplingConfig.get_default_path().exists():
        return DEFAULT_TIMEOUT
    return CouplingConfig.load().commChannelTimeout or None
