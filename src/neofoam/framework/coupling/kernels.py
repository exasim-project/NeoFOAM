# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The strategies of a coupling: mapping, acceleration and convergence of the exchanged data.

Each is a family of NeoFOAM's :class:`~neofoam.core.plugin_system.PluginSystem`:
a pydantic model per strategy, selected by its ``type``, its options its
fields (an unknown option is an error). The mapping is per exchange and
receiving side (:class:`CouplingMapping`); relaxation and convergence belong to a
coupling node (its leader's
:class:`~neofoam.framework.coupling.controller.CouplingController`): a list of convergence
measures (:class:`CouplingConvergence`), all of which must hold, and at most one
acceleration (:class:`CouplingAcceleration`) per exchange::

    CouplingNode(
        "thermal",
        ("interface",),
        convergence=(RelativeConvergence(data="T", limit=1e-5),),
        acceleration={"interface": AitkenRelaxation(relaxation=0.5)},
    )

The case names them by ``type`` (``system/regionCoupling.yaml``;
:func:`strategy` builds one from its options). A strategy of one's own
subclasses its family with a ``type: Literal[...]`` and registers with it
(``@CouplingMapping.register``). A package adds its strategies through the entry-point
group ``neofoam.coupling`` (:data:`ENTRY_POINT_GROUP`): each entry point names
a module that registers them as it is imported; :func:`load_plugins` imports
them, once, before the regions start::

    # pyproject.toml of neofoam-precice:
    #   [project.entry-points."neofoam.coupling"]
    #   precice = "neofoam_precice.coupling"
    @CouplingMapping.register
    class RadialBasisMapping(CouplingMapping):
        type: Literal["rbf"] = "rbf"
        support_radius: float
"""

from __future__ import annotations

import importlib.metadata
import threading
from typing import Any, Callable, Literal, NamedTuple, Optional, TypeVar, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, ValidationError

from neofoam.core.plugin_system import PluginSystem

from .transport import Data


class SideChange(NamedTuple):
    """The :data:`~neofoam.framework.coupling.Array` data one side sent in the previous and
    in the current pass.

    For a convergence check; plain fields are not compared.
    """

    field: str
    previous: Data
    current: Data


#: The entry-point group of the packages that add strategies to the families.
ENTRY_POINT_GROUP = "neofoam.coupling"

S = TypeVar("S", bound=BaseModel)


def strategy(family: type[S], options: Any) -> S:
    """The strategy of ``family`` (:class:`CouplingMapping`, :class:`CouplingAcceleration`,
    :class:`CouplingConvergence`) that ``options`` describe: ``{type: ..., <option>: ...}``, or
    the ``type`` alone.

    Raises ``ValueError`` listing the registered types when ``type`` is missing
    or none of them, and naming the strategy's options when one is unknown,
    missing or invalid. A strategy is passed through.

    Example::

        strategy(CouplingAcceleration, {"type": "aitken", "relaxation": 0.5})
        strategy(CouplingMapping, "nearest-neighbor")
    """
    if isinstance(options, family):
        return options
    what = family.__name__.removeprefix("Coupling").lower()
    if isinstance(options, str):
        options = {"type": options}
    members = _members(family)
    kind = options.get("type") if isinstance(options, dict) else None
    if kind not in members:
        problem = f"unknown {what} type {kind!r}" if kind else f"{what} {options!r} has no type"
        raise ValueError(f"{problem}; registered {what} types: {sorted(members)}")
    variable = _family(family).discriminator_variable
    try:
        built = cast(Any, family).create(**{variable: options})
    except ValidationError as error:
        problems = "; ".join(_problem(problem) for problem in error.errors())
        names = [name for name in members[kind].model_fields if name != "type"]
        raise ValueError(f"{what} {kind!r}: {problems}; its options: {names}") from None
    return cast(S, getattr(built, variable))


def _family(family: type[BaseModel]) -> Any:
    registry = PluginSystem.get_registered(family.__name__)
    if registry is None:
        raise TypeError(f"{family.__name__} is no plugin family")
    return registry


def _members(family: type[BaseModel]) -> dict[str, type[BaseModel]]:
    """The registered strategies of ``family``, by type."""
    return {
        str(member.model_fields["type"].default): member
        for member in _family(family).plugin_registry
        if "type" in member.model_fields
    }


def _problem(error: Any) -> str:
    """One pydantic error of a strategy's options, named by its option."""
    option = ".".join(str(part) for part in error["loc"][2:])
    if error["type"] == "extra_forbidden":
        return f"unknown option '{option}'"
    if error["type"] == "missing":
        return f"missing option '{option}'"
    return f"{option}: {error['msg']}"


_load_lock = threading.Lock()
_loaded = False


def load_plugins() -> None:
    """Import the module of every plugin of :data:`ENTRY_POINT_GROUP`, once; it registers its
    strategies with their families as it is imported.

    Call it before the regions start (``multiRun`` does): a case's config names
    a plugin's strategies by their type only once they are registered. Safe to
    call from several threads; the first call imports, the others wait for it.

    Raises ``ImportError`` naming the plugin when it fails to import (e.g. it
    registers a type that is registered already); the next call tries again.
    """
    global _loaded
    with _load_lock:
        if _loaded:
            return
        entry_points = importlib.metadata.entry_points(group=ENTRY_POINT_GROUP)
        for entry_point in sorted(entry_points, key=lambda e: e.name):
            try:
                entry_point.load()
            except Exception as error:
                raise ImportError(
                    f"coupling plugin '{entry_point.name}' ({entry_point.value}) of "
                    f"entry-point group '{ENTRY_POINT_GROUP}' failed to load: {error}"
                ) from error
        _loaded = True


# The elements of the target-by-source distance block :func:`nearest_neighbor` computes at once.
_BLOCK_ELEMENTS = 1 << 20


def nearest_neighbor(source: Any, target: Any) -> Any:
    """For every target point, the index of the nearest source point (the lowest on a tie).

    Points are rows of ``(n_points, dim)`` arrays. Compares every pair of points,
    a block of target points at a time, so that memory grows with the points of
    one side, not with the pairs: O(n + m) for ``m`` source and ``n`` target
    points, time O(n * m).

    Example::

        values_on_target = values_on_source[nearest_neighbor(source, target)]
    """
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    indices = np.empty(len(target), dtype=np.intp)
    rows = max(1, _BLOCK_ELEMENTS // max(1, source.size))
    for start in range(0, len(target), rows):
        block = target[start : start + rows]
        distances = np.linalg.norm(block[:, None, :] - source[None, :, :], axis=-1)
        indices[start : start + rows] = np.argmin(distances, axis=1)
    return indices


#: A mapping bound to the points of both sides: the data on the source points -> on the target's.
BoundMapping = Callable[[Data], Data]


@PluginSystem.register(discriminator_variable="mapping", discriminator="type")
class CouplingMapping(BaseModel):
    """Maps the :data:`~neofoam.framework.coupling.Array` data a side sends onto the other side.

    The mapping of an :class:`~neofoam.framework.coupling.data.Exchange`, a plugin
    family (``system/regionCoupling.yaml``: ``mapping:``, its ``type`` and
    options). The receiving side's coupling op binds it once, when it connects,
    to the partner's points (``source``) and the side's own (``target``; each a
    ``(n_points, dim)`` array, or ``None`` without coordinates), and maps the
    data of every pass with the bound mapping. A mapping of one's own subclasses
    it with a ``type: Literal[...]``, its options as fields, overrides
    :meth:`bind` and registers (``@CouplingMapping.register``).

    Example::

        mapper = exchange.mapping.bind(partner_points, points)
        mapped = mapper({"T": values})
    """

    model_config = ConfigDict(extra="forbid")

    def bind(self, source: Any, target: Any) -> BoundMapping:
        """The mapping from the points ``source`` onto ``target``.

        Raises ``ValueError`` when it cannot map between them.
        """
        raise NotImplementedError


@CouplingMapping.register
class IdentityMapping(CouplingMapping):
    """Hands the data over unchanged: the sides share their points."""

    type: Literal["identity"] = "identity"

    def bind(self, source: Any, target: Any) -> BoundMapping:
        """Raises ``ValueError`` when both sides have points and their numbers differ."""
        if source is not None and target is not None and len(source) != len(target):
            raise ValueError(
                f"mapping '{self.type}' needs as many points on both sides: "
                f"{len(source)} sent, {len(target)} received"
            )
        return _unchanged


def _unchanged(data: Data) -> Data:
    return data


@CouplingMapping.register
class NearestNeighborMapping(CouplingMapping):
    """Maps every point of the receiving side to the nearest point of the sending side.

    Both sides need coordinates. :meth:`bind` finds the nearest points
    (:func:`nearest_neighbor`) once, so the points must not move.
    """

    type: Literal["nearest-neighbor"] = "nearest-neighbor"

    def bind(self, source: Any, target: Any) -> BoundMapping:
        if source is None or target is None:
            raise ValueError(f"mapping '{self.type}' needs the coordinates of both sides")
        indices = nearest_neighbor(source, target)

        def mapper(data: Data) -> Data:
            return {name: np.asarray(values)[indices] for name, values in data.items()}

        return mapper


@PluginSystem.register(discriminator_variable="acceleration", discriminator="type")
class CouplingAcceleration(BaseModel):
    """Relaxes the Array data carried into a pass: (data of the carried sides, iteration) -> data.

    The acceleration of an exchange of a
    :class:`~neofoam.framework.coupling.CouplingNode`, a plugin family
    (``system/regionCoupling.yaml``: ``acceleration:``, its ``type`` and
    options). It keeps the data of its last passes in private attributes: give
    each exchange its own. An acceleration of one's own subclasses it with a
    ``type: Literal[...]``, its options as fields, overrides :meth:`__call__`
    and registers (``@CouplingAcceleration.register``).
    """

    model_config = ConfigDict(extra="forbid")

    def __call__(self, sent: list[Data], iteration: int) -> list[Data]:
        """The relaxed ``sent`` for pass ``iteration``."""
        raise NotImplementedError


@CouplingAcceleration.register
class ConstantRelaxation(CouplingAcceleration):
    """Under-relaxes the data carried into a pass by a fixed factor, ``relaxation``.

    ``x = x_last + relaxation * (sent - x_last)`` for all carried :data:`Array`
    data at once, with ``x_last`` what the last pass received. ``relaxation = 1``
    changes nothing; the first data passes unchanged. Not thread safe.

    Example::

        CouplingNode(
            "n", ("interface",), acceleration={"interface": ConstantRelaxation(relaxation=0.5)}
        )
    """

    type: Literal["constant"] = "constant"
    relaxation: float = Field(gt=0, le=1)
    _last: Optional[Any] = PrivateAttr(default=None)

    def __call__(self, sent: list[Data], iteration: int) -> list[Data]:
        """The relaxed ``sent`` for pass ``iteration``."""
        values = _flatten(sent)
        if self._last is not None:
            residual = values - self._last
            values = self._last + self._factor(residual, iteration) * residual
        self._last = values
        return _unflatten(values, sent)

    def _factor(self, residual: Any, iteration: int) -> float:
        return self.relaxation


@CouplingAcceleration.register
class AitkenRelaxation(ConstantRelaxation):
    """Under-relaxes the data carried into a pass by Aitken's dynamic factor.

    Use it instead of :class:`ConstantRelaxation` when a fixed factor converges
    slowly; on a linear fixed-point problem of one value it reaches the solution
    in the second relaxed pass. The factor follows from the residuals of the last
    two passes; the first pass of a time step uses ``relaxation``. Not thread safe.

    Example::

        CouplingNode(
            "n", ("interface",), acceleration={"interface": AitkenRelaxation(relaxation=0.5)}
        )
    """

    type: Literal["aitken"] = "aitken"  # type: ignore[assignment]
    _residual: Optional[Any] = PrivateAttr(default=None)
    _omega: float = PrivateAttr(default=0.0)

    def _factor(self, residual: Any, iteration: int) -> float:
        previous = self._residual if iteration > 1 else None
        self._residual = residual
        # A previous residual that is round-off against this one (e.g. a time step
        # that starts converged) would drive the factor to zero and stall the
        # loop: start over from the initial factor instead.
        if previous is None or np.linalg.norm(previous) <= 1e-6 * np.linalg.norm(residual):
            self._omega = self.relaxation
            return self._omega
        change = residual - previous
        squared = float(change @ change)
        if squared > 0.0:
            self._omega *= -float(previous @ change) / squared
        return self._omega


def _flatten(sent: list[Data]) -> Any:
    """All arrays of ``sent`` as one vector, in order."""
    arrays = [np.ravel(data[name]) for data in sent for name in sorted(data)]
    return np.concatenate(arrays) if arrays else np.zeros(0)


def _unflatten(values: Any, like: list[Data]) -> list[Data]:
    """``values`` cut into the arrays of ``like``."""
    result: list[Data] = []
    start = 0
    for data in like:
        unflattened: Data = {}
        for name in sorted(data):
            shape = np.shape(data[name])
            size = int(np.prod(shape))
            unflattened[name] = values[start : start + size].reshape(shape)
            start += size
        result.append(unflattened)
    return result


@PluginSystem.register(discriminator_variable="convergence", discriminator="type")
class CouplingConvergence(BaseModel):
    """Whether the coupling has converged, from the change of every side's data in the last pass.

    A convergence measure of a :class:`~neofoam.framework.coupling.CouplingNode`,
    a plugin family (``system/regionCoupling.yaml``: ``convergence:``, a list of
    ``type`` and options). A measure on the data of one field names it in its
    ``data`` option. A measure of one's own subclasses it with a
    ``type: Literal[...]``, its options as fields, overrides :meth:`__call__` and
    registers (``@CouplingConvergence.register``).
    """

    model_config = ConfigDict(extra="forbid")

    def __call__(self, changes: list[SideChange]) -> bool:
        """Whether the data converged, from ``changes``."""
        raise NotImplementedError


@CouplingConvergence.register
class RelativeConvergence(CouplingConvergence):
    """Converged once the data of field ``data`` changes by less than ``limit`` relative to it.

    It checks every array sent by a side of the field ``data`` (2-norm of the
    change over the 2-norm of the array) and ignores the sides of other fields.

    Example::

        CouplingNode("n", ("interface",), convergence=(RelativeConvergence(data="T", limit=1e-5),))
    """

    type: Literal["relative"] = "relative"
    data: str
    limit: float = Field(gt=0)

    def __call__(self, changes: list[SideChange]) -> bool:
        """Whether every array of field ``data`` changed by less than ``limit`` in the last pass."""
        return all(
            np.linalg.norm(np.asarray(change.current[name]) - np.asarray(change.previous[name]))
            <= self.limit * np.linalg.norm(change.current[name])
            for change in changes
            if change.field == self.data
            for name in change.current
        )
