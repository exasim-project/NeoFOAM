# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
SolverRuntime — mutable per-instance state for a SolverSpec.

Created by SolverSpec.instantiate(). Each call produces an independent
runtime so multiple solver runs never share mutable state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .spec import SolverSpec
    from neofoam.framework.context import Context
    from neofoam.framework.operations import Operations, StepBuilder


@dataclass
class SolverState:
    """Mutable state container for a single solver run."""

    core_models: list[Any] = field(default_factory=list)
    optional_models: list[Any] = field(default_factory=list)
    configs: dict[str, Any] = field(default_factory=dict)


@dataclass
class SolverRuntime:
    """
    One instantiation of a SolverSpec with its own mutable state.

    Owns per-instance state (models, configs, argv). All decorator-
    registered callables live on the spec; this object provides the
    public execution API (initialize, execution_graph, operations).
    """

    spec: "SolverSpec"
    name: str
    argv: list[Any] = field(default_factory=list)
    state: SolverState = field(default_factory=SolverState)
    _config_instance: Any = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # Execution API
    # ------------------------------------------------------------------

    def initialize(self) -> "Context":
        """Execute the registered initialization step."""
        return self.spec._run_initialize(self)

    def execution_graph(
        self, domain_name: Optional[str] = None
    ) -> tuple["StepBuilder", "Operations"]:
        """Execute the registered execution graph step."""
        return self.spec._run_execution_graph(self, domain_name)

    @property
    def operations(self) -> "Operations":
        """Build operations with this runtime as the self binding."""
        from neofoam.framework.operations import Operations

        return Operations(self.spec._build_operations_for(self))

    def get_config(self) -> Any:
        """
        Get solver configuration instance (lazy initialization).

        Used as a dependency provider:
            config: Annotated[MyConfig, Depends(solver.get_config)]
        """
        if self._config_instance is None:
            if self.spec._config_class is None:
                raise RuntimeError(f"No config class defined for solver '{self.name}'")
            self._config_instance = self.spec._config_class()
        return self._config_instance
