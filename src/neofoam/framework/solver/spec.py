# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
SolverSpec — immutable solver definition registered once at module import.

Stores only callables; all mutable state lives on SolverRuntime created
per instantiate().
"""

from __future__ import annotations

import inspect
import re
from types import SimpleNamespace
from typing import Any, Callable, Optional, TypeVar, cast

from neofoam.framework.context import Context
from neofoam.framework.dependency_resolver import (
    DependencyResolver,
    wrap_with_dependency_resolution,
)
from neofoam.framework.operations import Operation, OperationCollection, SequentialOp
from neofoam.framework.types import OperationMetadata, OperationNumber

from .runtime import SolverRuntime


_ConfigT = TypeVar("_ConfigT", bound=type)


def _snake_case(name: str) -> str:
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


class SolverSpec:
    """
    Immutable solver definition. Read-only after module import.

    Decorator methods store callables only; execution is delegated to
    SolverRuntime so that multiple runtimes can coexist independently.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._config_classes: list[type] = []
        self._core_model_specs: list[Any] = []
        self._optional_model_specs: list[Any] = []
        self._operations: list[tuple[Any, dict[str, Any]]] = []
        self._initialize_func: Optional[Callable[..., Context]] = None
        self._execution_graph_func: Optional[Callable[..., tuple[Any, Any]]] = None
        self._dependency_resolver = DependencyResolver()

    @property
    def _config_class(self) -> Optional[type]:
        return self._config_classes[0] if self._config_classes else None

    @_config_class.setter
    def _config_class(self, cls: Optional[type]) -> None:
        if cls is None:
            self._config_classes = []
        elif not self._config_classes:
            self._config_classes = [cls]
        else:
            self._config_classes[0] = cls

    # ------------------------------------------------------------------
    # Config registration
    # ------------------------------------------------------------------

    def config(self, cls: _ConfigT) -> _ConfigT:
        """Register a config class. Callable multiple times.

        When more than one class is registered, ``runtime.config`` becomes a
        ``SimpleNamespace`` keyed by snake-case class name; a single
        registration leaves ``runtime.config`` as the instance itself.

        Classes opting in via ``_synthesize_per_spec = True`` (e.g.
        ``neofoam.foam.fvSchemes``) get a fresh per-spec subclass.
        """
        if getattr(cls, "_synthesize_per_spec", False):
            subclass = cast(_ConfigT, type(f"{self.name}_{cls.__name__}", (cls,), {}))
            subclass._pending_sections = {}  # type: ignore[attr-defined]
            subclass._finalized = False  # type: ignore[attr-defined]
            self._config_classes.append(subclass)
            return subclass

        if cls in self._config_classes:
            return cls
        self._config_classes.append(cls)
        return cls

    # ------------------------------------------------------------------
    # Model-family registration
    # ------------------------------------------------------------------

    def core_models(self, family: Any) -> Any:
        """Bind a *required* model family (exactly one member runs per case).

        ``family`` must expose ``all_specs()`` (member specs, case-free) and
        ``detect_and_create()`` (select the single member for a concrete
        case). Example: the pressure-velocity family (PIMPLE / SIMPLE /
        PISO). Every member's configs join the case-free schema; detection
        picks which one runs.
        """
        if family not in self._core_model_specs:
            self._core_model_specs.append(family)
        return family

    def optional_models(self, family: Any) -> Any:
        """Bind an *optional* model family (zero or more members per case).

        ``family`` must expose ``all_specs()`` (member specs, case-free) and
        ``detect_models(case_dir)`` (the active members for a concrete case).
        Example: the ``incompressibleFluidModel`` family (boussinesq, …).
        Every member's configs join the case-free schema; detection picks
        which (if any) are active.
        """
        if family not in self._optional_model_specs:
            self._optional_model_specs.append(family)
        return family

    @property
    def model_specs(self) -> list[Any]:
        """Every member of every bound family, case-free (no detection).

        The union that — together with the solver's own ``_config_classes``
        — defines the full config schema returned by :func:`configurations`.
        """
        specs: list[Any] = []
        for family in (*self._core_model_specs, *self._optional_model_specs):
            specs.extend(family.all_specs())
        return specs

    @property
    def optional_model_specs(self) -> list[Any]:
        """The bound *optional* model families (case-free, no detection).

        Each family's members are optional models; some may flag themselves as
        a UI toggle (e.g. buoyancy) — see
        :func:`neofoam.framework.solver.configurations.toggle_models`.
        """
        return list(self._optional_model_specs)

    def detect_core_models(self, case_dir: Optional[Any] = None) -> list[Any]:
        """Select the single active member of each bound core family."""
        return [family.detect_and_create() for family in self._core_model_specs]

    def detect_optional_models(self, case_dir: Optional[Any] = None) -> list[Any]:
        """Collect the active members of each bound optional family."""
        active: list[Any] = []
        for family in self._optional_model_specs:
            active.extend(family.detect_models(case_dir))
        return active

    def _aggregate_configs(self, instances: list[Any]) -> Any:
        """Build ``runtime.config`` from loaded instances."""
        if not self._config_classes:
            return None

        matched: dict[str, Any] = {}
        for cls in self._config_classes:
            for inst in instances:
                if isinstance(inst, cls):
                    matched[_snake_case(cls.__name__)] = inst
                    break

        if not matched:
            return None
        if len(self._config_classes) == 1:
            return next(iter(matched.values()))
        return SimpleNamespace(**matched)

    # ------------------------------------------------------------------
    # Decorator API
    # ------------------------------------------------------------------

    def initializer(self, func: Callable[..., Context]) -> Callable[..., Context]:
        """Decorator to register solver initializer with dependency injection support."""
        self._initialize_func = func
        return func

    def execution_graph_step(
        self, func: Callable[..., tuple[Any, Any]]
    ) -> Callable[..., tuple[Any, Any]]:
        """Decorator to register execution graph construction step."""
        self._execution_graph_func = func
        return func

    def operation(
        self,
        operation_number: Optional[str] = None,
        depends_on: Optional[list[str]] = None,
        before: Optional[list[str]] = None,
        name: Optional[str] = None,
    ) -> Callable[..., Any]:
        """Decorator to register a solver operation."""

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self._operations.append(
                (
                    func,
                    {
                        "operation_number": operation_number,
                        "depends_on": depends_on,
                        "before": before,
                        "name": name or func.__name__,
                    },
                )
            )
            return func

        return decorator

    # ------------------------------------------------------------------
    # Instantiation
    # ------------------------------------------------------------------

    def instantiate(self, argv: Optional[list[Any]] = None) -> SolverRuntime:
        """Create a fresh SolverRuntime for one run of this solver."""
        return SolverRuntime(
            spec=self,
            name=self.name,
            argv=argv or [],
        )

    # ------------------------------------------------------------------
    # Execution helpers (called by SolverRuntime)
    # ------------------------------------------------------------------

    def _run_initialize(self, runtime: SolverRuntime) -> Context:
        """Execute the registered initialization step with dependency injection."""
        if self._initialize_func is None:
            raise RuntimeError(
                f"No initialize function registered for solver {self.name}"
            )

        kwargs = self._dependency_resolver.resolve_arguments(
            self._initialize_func, None
        )

        sig = inspect.signature(self._initialize_func)
        if "self" in sig.parameters and "self" not in kwargs:
            kwargs["self"] = runtime

        first_param = next(iter(kwargs.values()), None) if kwargs else None
        if first_param is not None and hasattr(first_param, "argv"):
            first_param.argv = runtime.argv

        ctx = self._initialize_func(**kwargs)

        if first_param is not None:
            inj_state = getattr(first_param, "state", None)
            if inj_state is not None:
                runtime.state = inj_state
            else:
                if hasattr(first_param, "core_models"):
                    runtime.state.core_models = first_param.core_models
                if hasattr(first_param, "optional_models"):
                    runtime.state.optional_models = first_param.optional_models

        # Pull every registered config class instance out of core_models
        # onto runtime.config.
        if self._config_classes and runtime.config is None:
            aggregated = self._aggregate_configs(runtime.state.core_models)
            if aggregated is not None:
                runtime.config = aggregated

        return ctx

    def _run_execution_graph(
        self,
        runtime: SolverRuntime,
        domain_name: Optional[str] = None,
        ctx: Optional[Context] = None,
    ) -> tuple[Any, Any]:
        """Execute the registered execution graph step.

        Any ``BaseConfig``-annotated parameter in the callback signature is
        type-injected from ``runtime.config``; a ``Context``-annotated parameter
        is injected with the live ``ctx`` (so the step can reach the built models
        without the runtime holding a long-lived reference to the context).
        """
        if self._execution_graph_func is None:
            raise RuntimeError(
                f"No execution_graph function registered for solver {self.name}"
            )

        from neofoam.framework.config_injection import (
            _discover_configs_from_signature,
            _find_config_by_type,
        )

        sig = inspect.signature(self._execution_graph_func)
        kwargs: dict[str, Any] = {}

        if "self" in sig.parameters:
            kwargs["self"] = runtime
        if "domain_name" in sig.parameters:
            kwargs["domain_name"] = domain_name

        for name, param in sig.parameters.items():
            if param.annotation is Context:
                kwargs[name] = ctx

        for cfg_meta in _discover_configs_from_signature(self._execution_graph_func):
            kwargs[cfg_meta["param_name"]] = _find_config_by_type(
                runtime.config, cfg_meta["config_type"]
            )

        return self._execution_graph_func(**kwargs)

    # ------------------------------------------------------------------
    # Operation building (called by SolverRuntime.operations)
    # ------------------------------------------------------------------

    def _build_operations_for(self, runtime: SolverRuntime) -> OperationCollection:
        """Build OperationCollection with *runtime* as the ``self`` binding."""
        from neofoam.framework.config_injection import (
            _discover_configs_from_signature,
            _create_runtime_config_wrapper,
        )

        ops = OperationCollection()
        for func, metadata in self._operations:
            discovered = _discover_configs_from_signature(func)
            if discovered:
                wrapped = _create_runtime_config_wrapper(func, discovered, runtime)
            else:
                wrapped = wrap_with_dependency_resolution(
                    func, runtime, self._dependency_resolver
                )

            op = Operation(
                func=SequentialOp(wrapped),
                metadata=OperationMetadata(
                    op_name=metadata["name"],
                    operation_number=(
                        OperationNumber(metadata["operation_number"])
                        if metadata["operation_number"]
                        else None
                    ),
                    depends_on=metadata["depends_on"] or [],
                    before=metadata["before"] or [],
                ),
            )
            ops.add(op)
        return ops


def Solver(name: str) -> SolverSpec:
    """Create a new SolverSpec instance."""
    return SolverSpec(name)
