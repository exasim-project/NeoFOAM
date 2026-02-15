# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

from typing import Annotated, Any, Protocol

import pybFoam as pyf

from neofoam.framework.context import FieldUpdates

from ..incompressibleFluidModel import Model, incompressibleFluidModel


class PressureVelocityAlgorithm(Protocol):
    def build(self) -> list[Any]: ...

    def inner_loop(self, ctx: Any) -> bool: ...

    def momentum(self, model_state: Any, **kwargs: Any) -> FieldUpdates: ...

    def continuity(self, model_state: Any, **kwargs: Any) -> FieldUpdates: ...


class pressureVelocityAlgorithm:
    @classmethod
    def register(cls, plugin_cls: type) -> type:
        return plugin_cls

    @classmethod
    def create(cls, *, algorithm_type: str) -> Any:
        from . import pimple, simple, piso

        algorithms = {
            "Pimple": pimple,
            "Simple": simple,
            "Piso": piso,
        }
        if algorithm_type not in algorithms:
            raise ValueError(
                f"Unsupported pressure-velocity algorithm model: {algorithm_type}. "
                f"Available: {list(algorithms.keys())}"
            )
        return algorithms[algorithm_type]


def _detect_algorithm_type(fv_solution: Any) -> str:
    toc = fv_solution.toc()
    if "PIMPLE" in toc:
        return "PIMPLE"
    if "PISO" in toc:
        return "PISO"
    if "SIMPLE" in toc:
        return "SIMPLE"
    raise ValueError("No supported algorithm found in system/fvSolution")


def _algorithm_model_name(algorithm_type: str) -> str:
    if algorithm_type == "PIMPLE":
        return "Pimple"
    if algorithm_type == "PISO":
        return "Piso"
    if algorithm_type == "SIMPLE":
        return "Simple"
    raise ValueError(f"Unsupported pressure-velocity algorithm: {algorithm_type}")


def _active_algorithm() -> PressureVelocityAlgorithm:
    algorithm = getattr(pressure_velocity, "algorithm", None)
    if algorithm is None:
        raise RuntimeError("pressureVelocity algorithm not initialized")
    return algorithm


pressure_velocity = Model("pressureVelocity").register_with(incompressibleFluidModel)
pressure_velocity.algorithm_type = "PIMPLE"
pressure_velocity.fv_solution = None
pressure_velocity.pRefCell = None
pressure_velocity.pRefValue = None
pressure_velocity.use_boussinesq = False
pressure_velocity.algorithm = None


@pressure_velocity.detect
def detect_model() -> bool:
    return True


@pressure_velocity.load
def load_config() -> None:
    pressure_velocity.use_boussinesq = False
    pressure_velocity.pRefCell = None
    pressure_velocity.pRefValue = None

    fv_solution = pyf.dictionary.read("system/fvSolution")
    pressure_velocity.algorithm_type = _detect_algorithm_type(fv_solution)
    pressure_velocity.fv_solution = fv_solution

    model_name = _algorithm_model_name(pressure_velocity.algorithm_type)
    pressure_velocity.algorithm = pressureVelocityAlgorithm.create(algorithm_type=model_name)


@pressure_velocity.resolve
def resolve(_config: Any) -> None:
    pass


@pressure_velocity.build
def build() -> list[Any]:
    algorithm = _active_algorithm()
    return algorithm.build()


def inner_loop(ctx: Any) -> bool:
    algorithm = _active_algorithm()
    return bool(algorithm.inner_loop(ctx))


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
    algorithm = _active_algorithm()
    return algorithm.momentum(
        model_state=pressure_velocity,
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
    algorithm = _active_algorithm()
    return algorithm.continuity(
        model_state=pressure_velocity,
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
