# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""numpy kernels of a coupling: mapping, relaxation and convergence of the exchanged data.

The mapping is per exchange and receiving side (:func:`mapping`); relaxation and
convergence belong to the leader's
:class:`~neofoam.framework.coupling.CouplingController`, which a region's
coupling builds with its ``controller`` factory::

    def controller(sides, label, parallel):
        return CouplingController(
            sides,
            label,
            parallel=parallel,
            converged=RelativeConvergence(data="T", limit=1e-5),
            acceleration=AitkenRelaxation(initial=0.5),
        )

The registries :data:`MAPPINGS`, :data:`ACCELERATIONS` and :data:`CONVERGENCES`
name them for the case (``system/regionCoupling.yaml``); a kernel of one's own
is added to its registry. A package adds its kernels through the entry-point
group ``neofoam.coupling`` (:data:`ENTRY_POINT_GROUP`): each entry point names a
``register(mappings, accelerations, convergences)`` that fills the three dicts
it is given; the registries take them in before they are first read
(:func:`load_plugins`)::

    # pyproject.toml of neofoam-precice:
    #   [project.entry-points."neofoam.coupling"]
    #   precice = "neofoam_precice:register"
    def register(mappings, accelerations, convergences):
        mappings["rbf"] = RadialBasisMapping
        accelerations["iqn-ils"] = IQNILSAcceleration
"""

from __future__ import annotations

import importlib.metadata
from typing import Any, Callable, Iterator, NamedTuple, Optional, TypeVar

import numpy as np

from .transport import Data


class SideChange(NamedTuple):
    """The :data:`~neofoam.framework.coupling.Array` data one side sent in the previous and
    in the current pass.

    For a convergence check; plain fields are not compared.
    """

    field: str
    previous: Data
    current: Data


V = TypeVar("V")
#: The entry-point group of the packages that add kernels to the registries.
ENTRY_POINT_GROUP = "neofoam.coupling"


class Registry(dict[str, V]):
    """A registry of kernels by name: a dict that takes in the plugins' kernels before it is read.

    Reading it (``[]``, ``in``, ``get``, iterating, ``len``, ``keys``, ``values``,
    ``items``) loads the plugins of :data:`ENTRY_POINT_GROUP` first, once
    (:func:`load_plugins`); writing to it does not.
    """

    def __getitem__(self, name: str) -> V:
        load_plugins()
        return super().__getitem__(name)

    def __contains__(self, name: object) -> bool:
        load_plugins()
        return super().__contains__(name)

    def __iter__(self) -> Iterator[str]:
        load_plugins()
        return super().__iter__()

    def __len__(self) -> int:
        load_plugins()
        return super().__len__()

    def get(self, name: str, default: Any = None) -> Any:
        load_plugins()
        return super().get(name, default)

    def keys(self) -> Any:
        load_plugins()
        return super().keys()

    def values(self) -> Any:
        load_plugins()
        return super().values()

    def items(self) -> Any:
        load_plugins()
        return super().items()


# Whether the coupling has converged, from the change of every side's data in the last pass.
Convergence = Callable[[list[SideChange]], bool]
# Relaxes the Array data carried into a pass: (data of the carried sides, iteration) -> data.
Acceleration = Callable[[list[Data], int], list[Data]]


def nearest_neighbor(source: Any, target: Any) -> Any:
    """For every target point, the index of the nearest source point.

    Points are rows of ``(n_points, dim)`` arrays. Compares every pair of points,
    so it suits interfaces of a few thousand points.

    Example::

        values_on_target = values_on_source[nearest_neighbor(source, target)]
    """
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    distances = np.linalg.norm(target[:, None, :] - source[None, :, :], axis=-1)
    return np.argmin(distances, axis=1)


class Mapping:
    """Maps the :data:`~neofoam.framework.coupling.Array` data a side sends onto the other side.

    The mapping of an :class:`~neofoam.framework.coupling.Exchange`, run by the
    receiving side's coupling op in its loop test: from the partner's points onto
    the side's own (each a ``(n_points, dim)`` array, or ``None`` without
    coordinates). ``kind`` names it (``system/regionCoupling.yaml``:
    ``mapping:``); :data:`MAPPINGS` holds the kinds the framework offers. This
    base class hands the data over unchanged (kind ``identity``): the sides share
    their points. A mapping of one's own subclasses it and sets ``kind``.

    Example::

        mapped = exchange.mapping({"T": values}, partner_points, points)
    """

    kind = "identity"

    def check(self, source: Any, target: Any) -> None:
        """Raise ``ValueError`` when it cannot map from the points ``source`` onto ``target``."""

    def __call__(self, data: Data, source: Any, target: Any) -> Data:
        """``data`` given on the points ``source``, on the points ``target``."""
        return data


class NearestNeighborMapping(Mapping):
    """Maps every point of the receiving side to the nearest point of the sending side.

    Both sides need coordinates. It keeps the indices of each pair of point
    arrays, so the points must not move. Not thread safe.
    """

    kind = "nearest-neighbor"

    def __init__(self) -> None:
        self._indices: dict[tuple[int, int], Any] = {}

    def check(self, source: Any, target: Any) -> None:
        if source is None or target is None:
            raise ValueError(f"mapping '{self.kind}' needs the coordinates of both sides")

    def __call__(self, data: Data, source: Any, target: Any) -> Data:
        pair = (id(source), id(target))
        if pair not in self._indices:
            self._indices[pair] = nearest_neighbor(source, target)
        return {name: np.asarray(values)[self._indices[pair]] for name, values in data.items()}


#: The mappings by their ``kind``: a new instance per exchange.
MAPPINGS: Registry[Callable[[], Mapping]] = Registry(
    {
        Mapping.kind: Mapping,
        NearestNeighborMapping.kind: NearestNeighborMapping,
    }
)


def mapping(kind: str) -> Mapping:
    """A new mapping of ``kind`` (a key of :data:`MAPPINGS`)."""
    if kind not in MAPPINGS:
        raise ValueError(f"unknown mapping '{kind}'; known: {sorted(MAPPINGS)}")
    return MAPPINGS[kind]()


class ConstantRelaxation:
    """Under-relaxes the data carried into a pass by a fixed factor.

    Use it as the ``acceleration`` of a
    :class:`~neofoam.framework.coupling.CouplingController`:
    ``x = x_last + omega * (sent - x_last)`` for all carried :data:`Array` data at once, with
    ``x_last`` what the last pass received. ``omega = 1`` changes nothing; the
    first data passes unchanged. Not thread safe.

    Example::

        CouplingController(**group, acceleration=ConstantRelaxation(omega=0.5))
    """

    def __init__(self, omega: float) -> None:
        self.omega = omega
        self._last: Optional[Any] = None

    def __call__(self, sent: list[Data], iteration: int) -> list[Data]:
        """The relaxed ``sent`` for pass ``iteration``."""
        values = _flatten(sent)
        if self._last is not None:
            residual = values - self._last
            values = self._last + self._factor(residual, iteration) * residual
        self._last = values
        return _unflatten(values, sent)

    def _factor(self, residual: Any, iteration: int) -> float:
        return self.omega


class AitkenRelaxation(ConstantRelaxation):
    """Under-relaxes the data carried into a pass by Aitken's dynamic factor.

    Use it instead of :class:`ConstantRelaxation` when a fixed factor converges
    slowly; on a linear fixed-point problem of one value it reaches the solution
    in the second relaxed pass. The factor follows from the residuals of the last
    two passes; the first pass of a time step uses ``initial``. Not thread safe.

    Example::

        CouplingController(**group, acceleration=AitkenRelaxation(initial=0.5))
    """

    def __init__(self, initial: float) -> None:
        super().__init__(initial)
        self._residual: Optional[Any] = None
        self._omega = initial

    def _factor(self, residual: Any, iteration: int) -> float:
        previous = self._residual if iteration > 1 else None
        self._residual = residual
        # A previous residual that is round-off against this one (e.g. a time step
        # that starts converged) would drive the factor to zero and stall the
        # loop: start over from the initial factor instead.
        if previous is None or np.linalg.norm(previous) <= 1e-6 * np.linalg.norm(residual):
            self._omega = self.omega
            return self._omega
        change = residual - previous
        squared = float(change @ change)
        if squared > 0.0:
            self._omega *= -float(previous @ change) / squared
        return self._omega


#: The accelerations by their name (``system/regionCoupling.yaml``: ``acceleration.type``),
#: built with the relaxation factor; a new instance per coupling, as it keeps its last passes.
ACCELERATIONS: Registry[Callable[[float], Acceleration]] = Registry(
    {
        "constant": ConstantRelaxation,
        "aitken": AitkenRelaxation,
    }
)


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


class RelativeConvergence:
    """Converged once the data of field ``data`` changes by less than ``limit`` relative to it.

    Use it as the ``converged`` of a
    :class:`~neofoam.framework.coupling.CouplingController`.
    It checks every array sent by a side of the field ``data`` (2-norm of the
    change over the 2-norm of the array) and ignores the sides of other fields.

    Example::

        CouplingController(**group, converged=RelativeConvergence(data="T", limit=1e-5))
    """

    def __init__(self, data: str, limit: float) -> None:
        self.data = data
        self.limit = limit

    def __call__(self, changes: list[SideChange]) -> bool:
        """Whether every array of field ``data`` changed by less than ``limit`` in the last pass."""
        return all(
            np.linalg.norm(np.asarray(change.current[name]) - np.asarray(change.previous[name]))
            <= self.limit * np.linalg.norm(change.current[name])
            for change in changes
            if change.field == self.data
            for name in change.current
        )


#: The convergence measures by their name (``system/regionCoupling.yaml``:
#: ``convergence[].type``), built with the field whose data they check and the limit.
CONVERGENCES: Registry[Callable[[str, float], Convergence]] = Registry(
    {
        "relative": RelativeConvergence,
    }
)

# None: the plugins are not loaded yet; False: they are loading; True: loaded.
_plugins_loaded: Optional[bool] = None


def load_plugins() -> None:
    """Add the kernels of every plugin of :data:`ENTRY_POINT_GROUP` to the registries, once.

    Each entry point names a callable ``register(mappings, accelerations,
    convergences)`` that fills the three empty dicts it is given; they join
    :data:`MAPPINGS`, :data:`ACCELERATIONS` and :data:`CONVERGENCES` when every
    plugin registered (a read of a registry while the plugins load sees the
    built-in kernels only). The registries call it before they are read; a case
    config that checks a name against a registry reads it.

    Raises ``ImportError`` naming the plugin when it fails to load or to
    register, ``ValueError`` naming both sides when a plugin registers a name
    that is built in or registered by another plugin. Nothing is added then, and
    the next read tries again.
    """
    global _plugins_loaded
    if _plugins_loaded is not None:
        return
    _plugins_loaded = False
    try:
        registries: dict[str, Registry[Any]] = {
            "mapping": MAPPINGS,
            "acceleration": ACCELERATIONS,
            "convergence": CONVERGENCES,
        }
        # who registered every name, by registry: NeoFOAM or a plugin
        sources: dict[str, dict[str, str]] = {
            what: dict.fromkeys(dict.keys(r), "NeoFOAM (built-in)")
            for what, r in registries.items()
        }
        added: dict[str, dict[str, Any]] = {what: {} for what in registries}
        entry_points = importlib.metadata.entry_points(group=ENTRY_POINT_GROUP)
        for entry_point in sorted(entry_points, key=lambda e: e.name):
            plugin = f"plugin '{entry_point.name}' ({entry_point.value})"
            try:
                register = entry_point.load()
            except Exception as error:
                raise ImportError(
                    f"coupling {plugin} of entry-point group '{ENTRY_POINT_GROUP}' "
                    f"failed to load: {error}"
                ) from error
            kernels: dict[str, dict[str, Any]] = {what: {} for what in registries}
            try:
                register(kernels["mapping"], kernels["acceleration"], kernels["convergence"])
            except Exception as error:
                raise ImportError(
                    f"coupling {plugin} of entry-point group '{ENTRY_POINT_GROUP}' "
                    f"failed to register: {error}"
                ) from error
            for what, entries in kernels.items():
                for name, kernel in entries.items():
                    if name in sources[what]:
                        raise ValueError(
                            f"coupling {plugin} and {sources[what][name]} both register the "
                            f"{what} '{name}'; a name is registered once"
                        )
                    sources[what][name] = plugin
                    added[what][name] = kernel
        for what, registry in registries.items():
            dict.update(registry, added[what])
    except BaseException:
        _plugins_loaded = None
        raise
    _plugins_loaded = True
