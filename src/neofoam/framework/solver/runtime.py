# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
SolverRuntime — mutable per-instance state for a SolverSpec.

Created by SolverSpec.instantiate(). Each call produces an independent
runtime so multiple solver runs never share mutable state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional, Protocol, Sequence, runtime_checkable

from neofoam.framework.dependency_resolver import DependencyResolver

if TYPE_CHECKING:
    from neofoam.framework.context import Context
    from neofoam.framework.coupling.exchanges import RegionCoupling
    from neofoam.framework.operations import (
        Operation,
        OperationCollection,
        Operations,
        StepBuilder,
    )

    from .spec import SolverSpec


@runtime_checkable
class GraphContribution(Protocol):
    """A model of ``ctx.models`` that adds operations to its region's graph."""

    def graph_operations(self, ctx: Context, graph: Sequence[Operation]) -> Operations:
        """The operations to add to ``graph``, the solver's operations."""
        ...

    def check_graph(self, graph: Operations) -> None:
        """Raise ``ValueError`` when the resolved ``graph`` is wrong for the model."""
        ...


@dataclass
class SolverState:
    """Mutable state container for a single solver run."""

    core_models: list[Any] = field(default_factory=list)
    optional_models: list[Any] = field(default_factory=list)
    configs: dict[str, Any] = field(default_factory=dict)


def core_model(state: Any, spec_name: str) -> Any:
    """The core model of ``state`` (a :class:`SolverState`) whose spec is named ``spec_name``.

    A solver's execution graph step takes its core models' operations from it.
    """
    return next(
        m for m in state.core_models if getattr(getattr(m, "spec", None), "name", None) == spec_name
    )


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
    # None for a single-region case
    region: Optional[str] = None
    # None: the case does not couple the region
    coupling: Optional[RegionCoupling] = None
    state: SolverState = field(default_factory=SolverState)
    config: Any = None
    _config_instance: Any = field(default=None, repr=False)
    # per run, so two runtimes of one spec (two regions) share no Depends() cache
    dependency_resolver: DependencyResolver = field(default_factory=DependencyResolver, repr=False)

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
        """Execute the registered initialization step; the context gets the runtime's region."""
        ctx = self.spec._run_initialize(self)
        ctx.region = self.region
        return ctx

    def execution_graph(
        self, domain_name: Optional[str] = None, ctx: Optional["Context"] = None
    ) -> tuple["StepBuilder", "OperationCollection"]:
        """Execute the registered execution graph step.

        ``ctx`` (the initialized :class:`Context`) is forwarded so a step that
        declares a ``Context`` parameter can reach the built models; it is passed
        transiently, never stored on the runtime.
        """
        return self.spec._run_execution_graph(self, domain_name, ctx)

    def resolve_operations(self, ctx: "Context") -> "Operations":
        """The runnable execution graph: ``execution_graph`` resolved with its model operations.

        Each :class:`GraphContribution` of ``ctx.models`` checks the result.
        Reported to the installed execution observer before it is returned.
        """
        from neofoam.framework import execution_observer  # noqa: PLC0415
        from neofoam.framework.graph import DAGResolver  # noqa: PLC0415

        builder, model_ops = self.execution_graph(ctx=ctx)
        operations = DAGResolver().resolve(builder, model_ops).operations
        for contribution in self.graph_contributions(ctx):
            contribution.check_graph(operations)
        observer = execution_observer.current()
        if observer is not None:
            observer.operations(operations, ctx)
        return operations

    def graph_contributions(self, ctx: "Context") -> list[GraphContribution]:
        """The models of ``ctx`` that contribute to the region's graph."""
        contributions = [m for m in ctx.models.values() if isinstance(m, GraphContribution)]
        if self.coupling is not None and not contributions:
            raise RuntimeError(
                f"region {self.region} is coupled in regionCoupling.yaml, but solver "
                f"{self.name} added no coupling model; add "
                "ctx.models['coupling'] = Coupling(self.coupling, geometry) in initialize"
            )
        return contributions

    def run(self) -> "Context":
        """Solve the runtime's region: initialize, resolve and run; return the context."""
        ctx = self.initialize()
        self.resolve_operations(ctx).run(ctx)
        return ctx

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
