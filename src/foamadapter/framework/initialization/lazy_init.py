# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

"""
Lazy Initialization

Provides the LazyInit dataclass for deferred initialization with dependency tracking.
"""

import inspect
from dataclasses import dataclass, field
from typing import Callable, Any, List, Union, cast


def _num_args(init: Callable[..., Any]) -> int:
    sig = inspect.signature(init)
    return len(sig.parameters)


@dataclass
class LazyInit:
    """
    A deferred initialization that declares its dependencies.

    LazyInit objects are returned from BUILD stage methods and collected
    by the initializer. They are then topologically sorted by dependencies
    and executed in the correct order.

    Attributes:
        name: Unique identifier (e.g., "fields.U", "operators.momentum")
        depends_on: List of dependency names that must be initialized first
        initializer: Lazy function that produces the runtime object
        category: Optional category for grouping (e.g., "fields", "operators")

    Example:
        LazyInit(
            name="fields.U",
            depends_on=["mesh"],
            initializer=lambda: create_vector_field(mesh, U0)
        )
    """

    name: str
    depends_on: List[str] = field(default_factory=list)
    initializer: Union[Callable[[], Any], Callable[[dict[str, Any]], Any], None] = None
    category: str | None = None

    def execute(self, context: dict[str, Any] | None = None) -> Any:
        """Execute the deferred initialization and return the result.

        Args:
            context: Dictionary of already-initialized objects for dependencies
        """
        if self.initializer is None:
            raise ValueError(f"LazyInit '{self.name}' has no initializer function")

        n_arguments = _num_args(self.initializer)
        if n_arguments == 1:
            if context is None:
                raise ValueError(
                    f"LazyInit '{self.name}' requires context but None was provided"
                )
            context_callable = cast(Callable[[dict[str, Any]], Any], self.initializer)
            return context_callable(context)

        no_arg_callable = cast(Callable[[], Any], self.initializer)
        return no_arg_callable()

    def __post_init__(self) -> None:
        """Validate LazyInit after creation."""
        if not self.name:
            raise ValueError("LazyInit must have a non-empty name")
        if self.initializer is None:
            raise ValueError(
                f"LazyInit '{self.name}' must have an initializer function"
            )
