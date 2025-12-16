# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

"""
Lazy Initialization

Provides the LazyInit dataclass for deferred initialization with dependency tracking.
"""

from dataclasses import dataclass, field
from typing import Callable, Any, List


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
    initializer: Callable[[], Any] | None = None
    category: str | None = None

    def execute(self, context: dict[str, Any] | None = None) -> Any:
        """Execute the deferred initialization and return the result.

        Args:
            context: Dictionary of already-initialized objects for dependencies
        """
        if self.initializer is None:
            raise ValueError(f"LazyInit '{self.name}' has no initializer function")

        # Try calling with context parameter first
        if context is not None:
            try:
                return self.initializer(context)  # type: ignore[call-arg]
            except TypeError:
                # If initializer doesn't accept context, call without it
                return self.initializer()
        else:
            return self.initializer()

    def __post_init__(self) -> None:
        """Validate LazyInit after creation."""
        if not self.name:
            raise ValueError("LazyInit must have a non-empty name")
        if self.initializer is None:
            raise ValueError(
                f"LazyInit '{self.name}' must have an initializer function"
            )
