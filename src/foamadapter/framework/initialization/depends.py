# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""FastAPI-style dependency injection helpers for Architecture 5b."""

from typing import Callable, Any, Union


class Depends:
    """
    Declare a dependency for initialization steps and runtime operations (FastAPI-style).

    Init-time usage (existing):
        from typing import Annotated

        init = Init("solver")

        @init.step
        def runTime() -> pyf.Time:
            return pyf.Time.read(argv)

        @init.step
        def mesh(runTime: Annotated[pyf.Time, Depends(runTime)]) -> pyf.fvMesh:
            return pyf.fvMesh.read(runTime)

    Runtime usage (new):
        @solver.operation
        def solve_momentum(
            self,
            U: Annotated[dict, Depends("fields.U")],
            turbulence: Annotated[Any, Depends("models.turbulence")]
        ) -> FieldUpdates:
            ...

    Provider usage (new):
        def get_nu_eff(turbulence: Annotated[Any, Depends("models.turbulence")]):
            return turbulence.nu() + turbulence.nut()

        @solver.operation
        def momentum(
            self,
            nu_eff: Annotated[float, Depends(get_nu_eff)]
        ):
            ...
    """

    def __init__(
        self,
        dependency: Union[str, Callable[..., Any]],
        *,
        scope: str = "time_step",  # time_step, iteration, operation
        cache: bool = True,
        optional: bool = False,
    ):
        """
        Args:
            dependency: The function to call or string path to resolve.
                       Can be another @init.step function, a provider function,
                       or a string like "fields.U" or "models.turbulence".
            scope: Caching scope - "time_step", "iteration", or "operation"
            cache: Whether to cache the resolved value within the scope
            optional: If True, don't raise error when dependency is missing
        """
        self.dependency = dependency
        self.scope = scope
        self.cache = cache
        self.optional = optional

    def __repr__(self) -> str:
        if isinstance(self.dependency, str):
            name = self.dependency
        else:
            name = getattr(self.dependency, "__name__", str(self.dependency))
        return f"Depends({name}, scope={self.scope})"

    def __call__(self) -> Any:
        """Call the dependency function to get the value (for backward compatibility)."""
        if callable(self.dependency):
            return self.dependency()
        raise TypeError(f"Cannot call string dependency: {self.dependency}")
