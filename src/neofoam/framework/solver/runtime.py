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
    from neofoam.framework.operations import OperationCollection, StepBuilder


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
    config: Any = None
    _config_instance: Any = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # State proxy properties
    # ------------------------------------------------------------------

    @property
    def core_models(self) -> list[Any]:
        """Access core_models from state."""
        return self.state.core_models

    @core_models.setter
    def core_models(self, value: list[Any]) -> None:
        self.state.core_models = value

    @property
    def optional_models(self) -> list[Any]:
        """Access optional_models from state."""
        return self.state.optional_models

    @optional_models.setter
    def optional_models(self, value: list[Any]) -> None:
        self.state.optional_models = value

    @property
    def configs(self) -> dict[str, Any]:
        """Access configs from state."""
        return self.state.configs

    @configs.setter
    def configs(self, value: dict[str, Any]) -> None:
        self.state.configs = value

    # ------------------------------------------------------------------
    # Execution API
    # ------------------------------------------------------------------

    def initialize(self) -> "Context":
        """Execute the registered initialization step."""
        return self.spec._run_initialize(self)

    def execution_graph(
        self, domain_name: Optional[str] = None, ctx: Optional["Context"] = None
    ) -> tuple["StepBuilder", "OperationCollection"]:
        """Execute the registered execution graph step.

        ``ctx`` (the initialized :class:`Context`) is forwarded so a step that
        declares a ``Context`` parameter can reach the built models; it is passed
        transiently, never stored on the runtime.
        """
        return self.spec._run_execution_graph(self, domain_name, ctx)

    @property
    def operations(self) -> "OperationCollection":
        """Build operations with this runtime as the self binding."""
        return self.spec._build_operations_for(self)

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
