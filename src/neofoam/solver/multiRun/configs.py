# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Case config of the multiRun solver: the shared time loop, the region solvers, their coupling."""

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
    load_plugins,
    strategy,
)
from neofoam.framework.coupling.comm_channel import DEFAULT_TIMEOUT
from neofoam.framework.coupling.kernels import default_mapping
from neofoam.io import YAML, BaseConfig, IOStrategy


def _with_plugins(family: type[BaseModel], options: Any) -> Any:
    """The strategy of ``family`` that ``options`` describe, the coupling plugins loaded first.

    :func:`~neofoam.framework.coupling.load_plugins` imports them once (locked), so
    reading ``system/regionCoupling.yaml`` anywhere (a tool, the UI, ``multiRun``)
    selects a plugin's strategy by its type.
    """
    load_plugins()
    return strategy(family, options)


def _selected(family: type[BaseModel], what: str) -> Any:
    """A field holding a strategy of ``family``, selected by its ``type`` as it is validated.

    The coupling plugins are loaded before the first strategy is selected
    (:func:`_with_plugins`), so a plugin's strategy is selected too; the JSON
    Schema is an open object, as plugins add their own options.
    """
    return Annotated[
        SerializeAsAny[family],  # type: ignore[valid-type]
        BeforeValidator(partial(_with_plugins, family)),
        WithJsonSchema(
            {
                "type": "object",
                "description": f"a {what}: its type and options",
            }
        ),
    ]


#: A convergence measure: ``{type: relative, data: T, limit: 1.0e-5}``
#: (:class:`~neofoam.framework.coupling.CouplingConvergence`; ``relative``: the data of
#: field ``data`` changes by less than ``limit`` relative to it).
ConvergenceConfig = _selected(CouplingConvergence, "convergence measure")
#: How the leader relaxes the data carried into a pass: ``{type: aitken, relaxation: 0.5}``
#: (:class:`~neofoam.framework.coupling.CouplingAcceleration`; ``constant`` factor
#: ``relaxation``, or ``aitken`` starting from it).
AccelerationConfig = _selected(CouplingAcceleration, "acceleration")
#: How the point data of one side is mapped onto the other's points:
#: ``nearest-neighbor`` (both patches need coordinates) or ``identity`` (both
#: patches share their points); a type alone, or ``{type: ..., <option>: ...}``
#: (:class:`~neofoam.framework.coupling.CouplingMapping`).
MappingConfig = _selected(CouplingMapping, "mapping")


@IOStrategy(YAML("system/controlDict.yaml"))
class ControlDictConfig(TimeControlConfig):
    """``system/controlDict.yaml`` — the time control every region reads, and its solvers.

    ``regionSolvers`` maps each region to the solver that runs it, in stepping
    order (a coupling group's last region leads it); each region keeps its own
    files under ``constant/<region>/``, ``0/<region>/`` and ``system/<region>/``.
    Every region runs its own time loop on this file's times.

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
    """One exchange: the coupled patch of each of its two regions, and how data is mapped.

    ``sides`` maps each region to its patch; ``name`` defaults to the regions
    joined by ``-``. ``mapping`` maps the point data of one side onto the other's
    points (:data:`MappingConfig`), by default ``nearest-neighbor``
    (:func:`~neofoam.framework.coupling.kernels.default_mapping`, as a ``Link``'s).
    """

    model_config = ConfigDict(extra="forbid")

    name: Optional[str] = None
    sides: dict[str, str] = Field(min_length=2, max_length=2)
    mapping: MappingConfig = Field(default_factory=default_mapping)


class NodeExchangeConfig(BaseModel):
    """An exchange of a coupling node, by its ``name``, and the ``acceleration`` of its data.

    In the yaml a plain name stands for an exchange without acceleration.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    acceleration: Optional[AccelerationConfig] = None


class PlacementConfig(BaseModel):
    """Where a coupling node's op sits in a region's graph (a ``Placement``).

    Without ``operations`` it wraps the whole time step (the body of the top
    scope, or of the node before); with them those operations and, if
    ``downstream``, every operation downstream of them.
    """

    model_config = ConfigDict(extra="forbid")

    operations: Optional[list[str]] = None
    downstream: bool = True


class CouplingNodeConfig(BaseModel):
    """One coupling node: a named coupling loop over some ``exchanges``.

    Its leader repeats the loop until every ``convergence`` measure holds, or
    ``maxIterations`` passes; ``wraps`` places its op per region (a region it
    does not name: the whole time step). ``scheme`` is the node's (by default
    the file's).
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    scheme: Optional[Literal["serial", "parallel"]] = None
    maxIterations: int = Field(default=100, gt=0)
    exchanges: list[NodeExchangeConfig] = Field(min_length=1)
    convergence: list[ConvergenceConfig] = Field(min_length=1)
    wraps: dict[str, PlacementConfig] = Field(default_factory=dict)

    @field_validator("exchanges", mode="before")
    @classmethod
    def _plain_names(cls, value: Any) -> Any:
        if isinstance(value, list):
            return [{"name": entry} if isinstance(entry, str) else entry for entry in value]
        return value


#: The name of the one coupling node of a file without ``couplings``.
DEFAULT_NODE = "coupling"


@IOStrategy(YAML("system/regionCoupling.yaml"))
class RegionCouplingConfig(BaseConfig):
    """``system/regionCoupling.yaml`` — the exchanges between regions and how they are coupled.

    Each entry of ``exchanges`` joins the coupled patches of two regions. Exchanges
    that share a region form one group; every region of a group repeats its
    solution in its coupling loop until every ``convergence`` measure holds, or
    ``maxIterations`` passes, as the group's leader decides. ``serial`` lets a
    region receive what a region before it sent in the same pass, ``parallel``
    (the default) what it sent in the previous one. Without the file the
    regions are not coupled.

    A serial node waits in each pass for the whole pass of the regions that step
    before it, so it holds the time step only around named operations
    (``wraps.<region>.operations``) and encloses no node that couples two of its
    regions as well; either raises ``ValueError`` as the case is run.

    ``couplings`` lists the coupling nodes, outer to inner: each names its
    exchanges (each in exactly one node), its convergence, maxIterations, an
    acceleration per exchange and where its op wraps each region. Without it one
    node ``coupling`` holds every exchange, with the top-level
    ``acceleration`` (of every exchange), ``convergence`` and ``maxIterations``;
    with it those top-level entries are an error. ``scheme`` is the scheme of
    every node that names none.

    ``transportTimeout`` (seconds, default 600) bounds how long a region waits
    for a message, in case the comm channel does not tell a deadlock at once;
    ``0`` or ``null`` waits without end. The environment variable
    ``NEOFOAM_COUPLING_TIMEOUT`` overrides it.

    Example::

        exchanges:
          - sides: {fluid: wall, solid: right}
            mapping: nearest-neighbor
        scheme: parallel
        acceleration:
          type: aitken
          relaxation: 0.5
        convergence:
          - type: relative
            data: T
            limit: 1.0e-5

    or, with the nodes::

        exchanges:
          - {name: wall-T, sides: {fluid: wall, solid: right}}
          - {name: wall-p, sides: {fluid: inlet, pipe: outlet}}
        couplings:
          - name: wall
            maxIterations: 50
            exchanges:
              - {name: wall-T, acceleration: {type: aitken, relaxation: 0.5}}
              - wall-p
            convergence:
              - {type: relative, data: T, limit: 1.0e-5}
            wraps:
              fluid: {operations: [solve_T], downstream: true}
    """

    model_config = ConfigDict(extra="forbid")

    exchanges: list[ExchangeConfig] = Field(min_length=1)
    scheme: Literal["serial", "parallel"] = "parallel"
    acceleration: Optional[AccelerationConfig] = None
    convergence: Optional[list[ConvergenceConfig]] = Field(default=None, min_length=1)
    maxIterations: Optional[int] = Field(default=None, gt=0)
    couplings: Optional[list[CouplingNodeConfig]] = Field(default=None, min_length=1)
    transportTimeout: Optional[float] = Field(default=DEFAULT_TIMEOUT, ge=0, allow_inf_nan=False)

    def exchange_names(self) -> list[str]:
        """Every exchange's name, in order: its ``name``, by default its regions joined by ``-``.

        A default name another exchange has is numbered (``a-b-2``). Raises
        ``ValueError`` when two exchanges are named alike.
        """
        names: list[str] = []
        for exchange in self.exchanges:
            name = exchange.name or _free("-".join(exchange.sides), names)
            if name in names:
                raise ValueError(f"two exchanges are named '{name}'")
            names.append(name)
        return names

    def nodes(self) -> list[CouplingNodeConfig]:
        """The coupling nodes, outer to inner: ``couplings``, or the one default node.

        A node that names no ``scheme`` gets the file's.
        """
        if self.couplings is not None:
            return [
                node if node.scheme else node.model_copy(update={"scheme": self.scheme})
                for node in self.couplings
            ]
        return [
            CouplingNodeConfig(
                name=DEFAULT_NODE,
                scheme=self.scheme,
                maxIterations=100 if self.maxIterations is None else self.maxIterations,
                exchanges=[
                    NodeExchangeConfig(name=name, acceleration=self.acceleration)
                    for name in self.exchange_names()
                ],
                convergence=self.convergence or [],
            )
        ]

    @model_validator(mode="after")
    def _nodes_place_every_exchange_once(self) -> "RegionCouplingConfig":
        names = self.exchange_names()
        if self.couplings is None:
            if self.convergence is None:
                raise ValueError(
                    "convergence: required without couplings (the convergence of the one "
                    "coupling node)"
                )
            return self
        top = [
            key
            for key in ("acceleration", "convergence", "maxIterations")
            if getattr(self, key) is not None
        ]
        if top:
            raise ValueError(
                f"couplings: with couplings the nodes set {top}; remove the top-level {top} "
                "(they set the one default node of a file without couplings)"
            )
        seen: dict[str, str] = {}
        nodes: list[str] = []
        for node in self.couplings:
            if node.name in nodes:
                raise ValueError(f"couplings: two coupling nodes are named '{node.name}'")
            nodes.append(node.name)
            for exchange in node.exchanges:
                if exchange.name not in names:
                    raise ValueError(
                        f"couplings: node '{node.name}' names the exchange '{exchange.name}', "
                        f"which does not exist; the exchanges are {names}"
                    )
                if exchange.name in seen:
                    raise ValueError(
                        f"couplings: exchange '{exchange.name}' is in the nodes "
                        f"'{seen[exchange.name]}' and '{node.name}'; an exchange is in exactly "
                        "one node"
                    )
                seen[exchange.name] = node.name
            regions = sorted(
                {
                    region
                    for name, exchange in zip(names, self.exchanges)
                    if name in seen and seen[name] == node.name
                    for region in exchange.sides
                }
            )
            foreign = sorted(set(node.wraps) - set(regions))
            if foreign:
                raise ValueError(
                    f"couplings: node '{node.name}' wraps the regions {foreign}, which it does "
                    f"not couple; its regions are {regions}"
                )
        missing = [name for name in names if name not in seen]
        if missing:
            raise ValueError(
                f"couplings: the exchanges {missing} are in no coupling node; each exchange is "
                f"in exactly one node (the nodes: {nodes})"
            )
        return self


def _free(name: str, taken: list[str]) -> str:
    """``name``, numbered if ``taken`` has it."""
    free, number = name, 1
    while free in taken:
        number += 1
        free = f"{name}-{number}"
    return free
