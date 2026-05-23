# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""Dependency marker used with ``typing.Annotated`` for injection."""

from typing import Callable, Any, Union


class Depends:
    """
    Declare a dependency for DI-enabled callables.

    Supported usage includes initializer injection, e.g.:

        @solver.initializer
        def initialize(
            init: Annotated[StagedInitRunner, Depends(create_init)],
        ) -> Context:
            return init.run()
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
                       Can be a provider function or a string path
                       like "fields.U" or "models.turbulence".
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
