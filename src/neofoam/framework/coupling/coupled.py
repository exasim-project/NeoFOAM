# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""``@coupled``: what a class sends and receives; :class:`CoupledPort`, where its owner binds it."""

from __future__ import annotations

import inspect
import typing
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

from .data import End, Stream, check_model

T = TypeVar("T", bound=type)
F = TypeVar("F", bound=Callable[..., Any])

_ROLE = "__coupled_role__"
_ROLES = ("sends", "receives")


@dataclass(frozen=True)
class _Declaration:
    """What ``@coupled`` read off a class, kept on it as ``__coupled__``: ``(method, model)``s."""

    sends: tuple[tuple[str, type[Any]], ...]
    receives: tuple[tuple[str, type[Any]], ...]

    def streams(self, instance: Any, role: str) -> tuple[Stream, ...]:
        """The streams of ``instance`` in ``role``, bound to its methods."""
        return tuple(Stream(model, getattr(instance, name)) for name, model in getattr(self, role))


def _marked(cls: type) -> dict[str, str]:
    """The methods of ``cls`` (and its bases) marked with a role: ``{name: role}``."""
    roles: dict[str, str] = {}
    for klass in reversed(cls.__mro__):
        for name, member in vars(klass).items():
            role = getattr(member, _ROLE, None)
            if role is not None:
                roles[name] = role
    return roles


def _hints(cls: type, name: str) -> dict[str, Any]:
    func = getattr(cls, name)
    try:
        return typing.get_type_hints(func, include_extras=False)
    except Exception as error:  # a name the annotation uses that is not defined
        raise TypeError(f"{cls.__qualname__}.{name}: cannot read its annotations ({error})")


def _model(cls: type, name: str, role: str) -> type[Any]:
    hints = _hints(cls, name)
    if role == "sends":
        model = hints.get("return")
    else:
        params = [p for p in inspect.signature(getattr(cls, name)).parameters if p != "self"]
        model = hints.get(params[0]) if params else None
    return check_model(model, f"{cls.__qualname__}.{name} (@coupled.{role})")


def _check_once(cls: type, role: str, declared: list[tuple[str, type[Any]]]) -> None:
    seen: dict[str, str] = {}
    for _name, model in declared:
        for field in model.model_fields:
            if field in seen:
                raise TypeError(
                    f"two @coupled.{role} of {field} in {cls.__name__} ({seen[field]} and "
                    f"{model.__qualname__}); values that go together share one model"
                )
            seen[field] = model.__qualname__


class _Coupled:
    """``@coupled`` on a class, and its method markers ``@coupled.sends`` / ``@coupled.receives``.

    A class sends, receives, or both; at most once per data field. It knows no
    location, geometry or context: its owner binds it with
    ``coupling(ctx).add(location, instance)``. Checked at decoration.

    Example::

        @coupled
        class CoupledHeatFlux:
            @coupled.sends
            def wall_temperature(self) -> WallTemperature: ...

            @coupled.receives
            def heat_in(self, data: WallHeatFlux) -> None: ...

        coupling(ctx).add(Patch("top"), CoupledHeatFlux())
    """

    def __call__(self, cls: T) -> T:
        marked = _marked(cls)
        if not marked:
            raise TypeError(
                f"{cls.__name__} is @coupled but marks no @coupled.sends or "
                "@coupled.receives method"
            )
        declared: dict[str, list[tuple[str, type[Any]]]] = {role: [] for role in _ROLES}
        for name, role in marked.items():
            declared[role].append((name, _model(cls, name, role)))
        for role in _ROLES:
            _check_once(cls, role, declared[role])
        cls.__coupled__ = _Declaration(  # type: ignore[attr-defined]
            tuple(declared["sends"]), tuple(declared["receives"])
        )
        return cls

    @staticmethod
    def sends(func: F) -> F:
        """Mark a method returning data the instance sends; its return annotation is the model."""
        setattr(func, _ROLE, "sends")
        return func

    @staticmethod
    def receives(func: F) -> F:
        """Mark a method taking data the instance receives; its parameter's type is the model."""
        setattr(func, _ROLE, "receives")
        return func


coupled = _Coupled()


@dataclass(frozen=True)
class CoupledPort:
    """A coupled location of a region: what it sends and receives there, by data key.

    Built by :meth:`~neofoam.framework.coupling.Participant.add`; several instances
    on one location merge into one port.
    """

    region: str
    location: Any
    sends: tuple[Stream, ...] = ()
    receives: tuple[Stream, ...] = ()

    @property
    def name(self) -> str:
        """The port's name, its location's."""
        return str(self.location.name)

    @property
    def end(self) -> End:
        """The port's end of its exchange."""
        return (self.region, self.name)

    @property
    def label(self) -> str:
        """``<region>.<port>``."""
        return f"{self.region}.{self.name}"

    @property
    def names(self) -> set[str]:
        """The data field names the port sends or receives."""
        return {name for stream in (*self.sends, *self.receives) for name in stream.names}

    def merged(self, other: CoupledPort) -> CoupledPort:
        """This port with ``other``'s streams; ``ValueError`` when a field name repeats."""
        if other.location != self.location:
            raise ValueError(
                f"{self.label} is bound to {self.location} and {other.location}; a port's "
                "name names one location"
            )
        for role in ("sends", "receives"):
            mine = {name for stream in getattr(self, role) for name in stream.names}
            for stream in getattr(other, role):
                twice = sorted(mine & set(stream.names))
                if twice:
                    raise ValueError(
                        f"two @coupled.{role} of {', '.join(twice)} on {self.label}; a port "
                        f"{role} each data field once"
                    )
        return CoupledPort(
            self.region,
            self.location,
            self.sends + other.sends,
            self.receives + other.receives,
        )
