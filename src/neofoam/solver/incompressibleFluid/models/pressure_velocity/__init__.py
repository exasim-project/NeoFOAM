# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Pressure-velocity model plugin for incompressibleFluid solver."""

from typing import Annotated, Any

from neofoam.framework.context import FieldUpdates

from ..incompressibleFluidModel import Model, incompressibleFluidModel
from .model import PressureVelocityModelImpl


pressure_velocity = Model("pressureVelocity").register_with(incompressibleFluidModel)
pressure_velocity.algorithm_type = "PIMPLE"
pressure_velocity.fv_solution = None
pressure_velocity.pRefCell = None
pressure_velocity.pRefValue = None
pressure_velocity.use_boussinesq = False

_impl = PressureVelocityModelImpl()


@pressure_velocity.detect
def detect_model() -> bool:
    return _impl.detect()


@pressure_velocity.load
def load_config() -> None:
    _impl.load(pressure_velocity)


@pressure_velocity.resolve
def resolve(_config: Any) -> None:
    _impl.resolve(_config)


@pressure_velocity.build
def build() -> list[Any]:
    return _impl.build(pressure_velocity)


def inner_loop(ctx: Any) -> bool:
    return _impl.inner_loop(pressure_velocity, ctx)


pressure_velocity.inner_loop = inner_loop


@pressure_velocity.operation(operation_number="2.1")
def momentum(
    U: Any,
    phi: Any,
    p: Any,
    turbulence: Annotated[Any, "models"],
    pimple_control: Annotated[Any, "models"],
    p_rgh: Any = None,
    rhok: Any = None,
    ghf: Any = None,
) -> FieldUpdates:
    return _impl.momentum(
        pressure_velocity,
        U=U,
        phi=phi,
        p=p,
        turbulence=turbulence,
        pimple_control=pimple_control,
        p_rgh=p_rgh,
        rhok=rhok,
        ghf=ghf,
    )


@pressure_velocity.operation(operation_number="2.2", depends_on=["momentum"])
def continuity(
    U: Any,
    p: Any,
    phi: Any,
    UEqn: Any,
    pimple_control: Annotated[Any, "models"],
    cumulativeContErr: Annotated[list[float], "models"],
    p_rgh: Any = None,
    rhok: Any = None,
    gh: Any = None,
    ghf: Any = None,
) -> FieldUpdates:
    return _impl.continuity(
        pressure_velocity,
        U=U,
        p=p,
        phi=phi,
        UEqn=UEqn,
        pimple_control=pimple_control,
        cumulativeContErr=cumulativeContErr,
        p_rgh=p_rgh,
        rhok=rhok,
        gh=gh,
        ghf=ghf,
    )
