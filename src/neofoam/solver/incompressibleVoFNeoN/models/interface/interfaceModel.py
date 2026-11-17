# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The interface model family of incompressibleVoFNeoN (pick one; only gradAlpha for now)."""

from typing import Any, Protocol

import pybFoam as pyf

from .grad_alpha import gradAlphaInterface

_DEFAULT_MODEL = "gradAlpha"
_MODELS: dict[str, Any] = {"gradAlpha": gradAlphaInterface}


class InterfaceModel(Protocol):
    """The interface geometry the surface forces read: face normal flux and curvature.

    ``n_hat_f`` is the unit-normal face flux ``nHat & Sf`` and ``curvature`` the cell
    curvature ``K``; both are valid after ``update`` and stay fixed until the next call.
    Implement it to add an interface reconstruction to :class:`interfaceModelNeoN`.

    Example::

        interface: InterfaceModel = gradAlphaInterface(alpha1, runtime)
        interface.update(alpha1, runtime)
    """

    n_hat_f: Any
    curvature: Any

    def update(self, alpha1: Any, runtime: Any) -> None:
        """Recompute the interface geometry from the phase fraction ``alpha1``."""
        ...


class interfaceModelNeoN:
    """Model family of the interface geometry, keyed by ``interfaceModel``.

    The member is the optional ``interfaceModel`` entry of ``constant/transportProperties``
    (default ``gradAlpha``, interFoam's ``interfaceProperties``). Select it with
    :meth:`create`; :meth:`select` maps a name to its class. Holds no state.

    Example::

        interface = interfaceModelNeoN.create(alpha1, runtime)
    """

    @classmethod
    def registered_names(cls) -> list[str]:
        """The ``interfaceModel`` names the family accepts."""
        return list(_MODELS)

    @classmethod
    def select(cls, name: str) -> Any:
        """The member class for ``interfaceModel`` ``name``; unknown names raise ValueError."""
        member = _MODELS.get(name)
        if member is None:
            raise ValueError(
                f"Unknown interfaceModel '{name}'; registered: {cls.registered_names()}"
            )
        return member

    @classmethod
    def name_from_case(cls) -> str:
        """The ``interfaceModel`` of the case in the working directory."""
        transport = pyf.dictionary.read("constant/transportProperties")
        return str(transport.getOrDefault[str]("interfaceModel", _DEFAULT_MODEL))

    @classmethod
    def create(cls, alpha1: Any, runtime: Any) -> InterfaceModel:
        """The active member for the case in the working directory, built on ``alpha1``."""
        member: InterfaceModel = cls.select(cls.name_from_case())(alpha1, runtime)
        return member
