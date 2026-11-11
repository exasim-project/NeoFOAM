# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""``@coupled``: a boundary condition declares its side of an exchange by annotations.

Example::

    @coupled
    class CoupledHeatFlux(Patch):     # sets self.field and self.patch
        def __init__(self, ctx, ...):
            ...
            Coupling.of(ctx).add(self)

        @coupled.coordinates
        def points(self) -> Array: ...

        @coupled.sends
        def send(self) -> WallTemperature: ...

        @coupled.receives
        def receive(self, data: WallHeatFlux) -> None: ...
"""

from __future__ import annotations

import inspect
import typing
from dataclasses import dataclass
from typing import Any, Callable, Optional, TypeVar

from .data import ExchangeSide, _is_model

T = TypeVar("T", bound=type)
F = TypeVar("F", bound=Callable[..., Any])

_ROLE = "__coupled_role__"


@dataclass(frozen=True)
class _Declaration:
    """What ``@coupled`` read off a class, kept on it as ``__coupled__``."""

    sends: type[Any]
    receives: type[Any]
    send: str
    receive: str
    coordinates: Optional[str]

    def side(self, instance: Any, region: str) -> ExchangeSide:
        """The coupled side of ``instance`` in ``region``, its methods bound."""
        points = getattr(instance, self.coordinates)() if self.coordinates else None
        return ExchangeSide(
            region=region,
            patch=instance.patch,
            field=instance.field,
            sends=self.sends,
            receives=self.receives,
            send=getattr(instance, self.send),
            receive=getattr(instance, self.receive),
            coordinates=points,
        )


def _marked(cls: type) -> dict[str, str]:
    """The methods of ``cls`` (and its bases) marked with a role: ``{role: name}``."""
    roles: dict[str, str] = {}
    for klass in reversed(cls.__mro__):
        for name, member in vars(klass).items():
            role = getattr(member, _ROLE, None)
            if role is not None:
                roles[role] = name
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
        where = "its return annotation"
    else:
        params = [p for p in inspect.signature(getattr(cls, name)).parameters if p != "self"]
        model = hints.get(params[0]) if params else None
        where = "the annotation of its data parameter"
    if not _is_model(model):
        raise TypeError(
            f"{cls.__qualname__}.{name} (@coupled.{role}) declares what the side {role} "
            f"by {where}, a pydantic model; got {model!r}"
        )
    return model  # type: ignore[return-value]


class _Coupled:
    """``@coupled`` on a boundary condition class, and its method markers."""

    def __call__(self, cls: T) -> T:
        """Declare the decorated class a coupled side."""
        roles = _marked(cls)
        missing = [role for role in ("sends", "receives") if role not in roles]
        if missing:
            raise TypeError(
                f"@coupled class {cls.__qualname__} needs a method marked "
                + " and one marked ".join(f"@coupled.{role}" for role in missing)
            )
        declaration = _Declaration(
            sends=_model(cls, roles["sends"], "sends"),
            receives=_model(cls, roles["receives"], "receives"),
            send=roles["sends"],
            receive=roles["receives"],
            coordinates=roles.get("coordinates"),
        )
        cls.__coupled__ = declaration  # type: ignore[attr-defined]
        return cls

    @staticmethod
    def sends(func: F) -> F:
        """Mark the method returning what the side sends; its return annotation is the model."""
        setattr(func, _ROLE, "sends")
        return func

    @staticmethod
    def receives(func: F) -> F:
        """Mark the method taking what the side receives; its parameter annotation is the model."""
        setattr(func, _ROLE, "receives")
        return func

    @staticmethod
    def coordinates(func: F) -> F:
        """Mark the method returning the patch's points, one row each (optional)."""
        setattr(func, _ROLE, "coordinates")
        return func


coupled = _Coupled()
