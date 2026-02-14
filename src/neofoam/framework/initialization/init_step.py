# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

"""
Lazy Initialization

Provides the InitStep dataclass for deferred initialization with dependency tracking.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Literal


InitCategory = Literal["fields", "models", "operators", "resource"]


class InitStepExecutionError(RuntimeError):
    """Raised when an InitStep's initializer fails during execution.

    Attributes:
        step_name: Name of the InitStep that failed.
        depends_on: Dependencies declared by the failed step.
    """

    def __init__(self, step_name: str, depends_on: list[str], cause: Exception) -> None:
        self.step_name = step_name
        self.depends_on = list(depends_on)
        deps = ", ".join(depends_on) if depends_on else "(none)"
        super().__init__(
            f"InitStep '{step_name}' failed during execution "
            f"(depends_on: [{deps}]): {cause}"
        )


@dataclass
class InitStep:
    """
    A deferred initialization step that declares its dependencies.

    InitStep objects are returned from BUILD stage methods and collected
    by the initializer. They are then topologically sorted by dependencies
    and executed in the correct order.

    Attributes:
        name: Unique identifier (e.g., "fields.U", "operators.momentum")
        depends_on: List of dependency names that must be initialized first
        initializer: Lazy function that produces the runtime object using context
        category: Optional category for grouping (e.g., "fields", "operators")

    Example:
        InitStep(
            name="fields.U",
            depends_on=["mesh"],
            initializer=lambda ctx: create_vector_field(ctx["mesh"], U0)
        )
    """

    name: str
    depends_on: list[str] = field(default_factory=list)
    initializer: Callable[[dict[str, Any]], Any] = None  # type: ignore[assignment]
    category: InitCategory = "resource"

    def execute(self, context: dict[str, Any]) -> Any:
        """Execute the deferred initialization and return the result.

        Args:
            context: Dictionary of already-initialized objects for dependencies

        Raises:
            ValueError: If the initializer is ``None``.
            InitStepExecutionError: If the initializer itself raises.
        """
        if self.initializer is None:
            raise ValueError(f"InitStep '{self.name}' has no initializer function")

        try:
            return self.initializer(context)
        except (ValueError, TypeError):
            raise
        except Exception as exc:
            raise InitStepExecutionError(self.name, self.depends_on, exc) from exc

    def __post_init__(self) -> None:
        """Validate InitStep after creation."""
        if not self.name:
            raise ValueError("InitStep must have a non-empty name")
        if self.initializer is None:
            raise ValueError(
                f"InitStep '{self.name}' must have an initializer function"
            )
        if self.category not in {"fields", "models", "operators", "resource"}:
            raise ValueError(
                f"InitStep '{self.name}' has invalid category '{self.category}'"
            )
