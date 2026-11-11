# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""``@coupled``: a class declares what it sends and receives; its owner binds it to a location."""

from __future__ import annotations

import inspect
import typing
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

from .data import Stream, check_model

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
