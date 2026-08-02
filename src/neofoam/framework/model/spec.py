# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
ModelSpec — immutable model definition registered once at module import.

Provides the same decorator API as the old ModelInstance but stores only
callables; all mutable state lives on ModelRuntime created per instantiate().
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Literal, Optional, Sequence, TypeVar, cast

from pydantic import BaseModel

from neofoam.core.plugin_system import PluginSystem
from neofoam.fields.decl import FieldDecl
from neofoam.framework.config_injection import (
    _create_runtime_config_wrapper,
    _discover_configs_from_signature,
)
from neofoam.framework.dependency_resolver import (
    DependencyResolver,
    wrap_with_dependency_resolution,
)
from neofoam.framework.operations import Operation, Operations, SequentialOp
from neofoam.framework.types import OperationMetadata, OperationNumber

from .extension import Extension, Hook
from .runtime import ModelRuntime

_ConfigT = TypeVar("_ConfigT", bound=type)
_T = TypeVar("_T")


def _snake_case(name: str) -> str:
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


class ModelSpec:
    """
    Immutable model definition.  Read-only after module import.

    Decorator methods store callables only; execution is delegated to
    ModelRuntime so that multiple runtimes can coexist independently.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.enabled = True

        # Human-readable display name for UIs (e.g. "Buoyancy (Boussinesq)").
        # Independent of required-ness; set at registration via :meth:`labeled`.
        self.label: str = name

        self._config_classes: list[type] = []
        self._field_decls: list[FieldDecl] = []

        self._load_func: Optional[Callable[..., Any]] = None
        self._resolve_func: Optional[Callable[..., Any]] = None
        self._build_func: Optional[Callable[..., Any]] = None
        self._detect_func: Optional[Callable[..., Any]] = None

        self._operations: list[tuple[Any, dict[str, Any]]] = []
        self._operation_collection_func: Optional[Callable[..., Any]] = None

        self._own_extension: Optional[Extension] = None

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
    # Field registration (declares a 0/<name> field this model owns)
    # ------------------------------------------------------------------

    def field(
        self,
        name: str,
        *,
        dimensions: list[int],
        value_type: type,
        allowed_bcs: "Sequence[type[BaseModel]]",
        write: bool = False,
        depends_on: "Sequence[str]" = ("mesh",),
        initial_value: Any = None,
    ) -> FieldDecl:
        """Declare an on-disk field (``0/<name>``) this model owns.

        Returns a :class:`FieldDecl` handle for cross-linking from the
        model's ``@build`` callback (``decl.create(read_fn)`` yields the
        :class:`InitStep` the runtime consumes). The declaration also
        seeds the per-field schema surfaced by ``configurations(solver)``
        — see :func:`neofoam.fields.schema.schema_for`.

        Args:
            name: Field name without prefix (``"U"`` / ``"p"`` / …).
            dimensions: OpenFOAM dimension exponents
                ``[M, L, T, Θ, N, I, J]``.
            value_type: :class:`~neofoam.fields.value_types.Scalar` /
                :class:`~neofoam.fields.value_types.Vector` marker.
            allowed_bcs: Permitted boundary-condition arms; include
                :class:`~neofoam.fields.bc.GenericBC` to admit unknown
                BC types via the smart-union fallback.
            write: Mark the field for runtime auto-persistence.
            depends_on: Runtime dependencies for the field's factory.
            initial_value: Optional default ``internalField`` baked
                into a scaffolded schema (Python literal or OpenFOAM
                uniform string).

        Returns:
            The :class:`FieldDecl` handle. Same object is stored on
            ``self._field_decls`` so the loader can discover it.

        Raises:
            ValueError: If a field of the same name is already declared
                on this spec.
        """
        if any(d.name == name for d in self._field_decls):
            raise ValueError(f"ModelSpec '{self.name}': field '{name}' already declared")
        decl = FieldDecl(
            name=name,
            dimensions=list(dimensions),
            value_type=value_type,
            allowed_bcs=tuple(allowed_bcs),
            write=write,
            depends_on=tuple(depends_on),
            initial_value=initial_value,
        )
        self._field_decls.append(decl)
        return decl

    @property
    def field_decls(self) -> "tuple[FieldDecl, ...]":
        """Read-only view of declared fields, in registration order."""
        return tuple(self._field_decls)

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
    # Stage decorators — store only, no side-effects
    # ------------------------------------------------------------------

    def load(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """
        Register the LOAD function.

        Signature: ``def load(case_dir: Path, instance_id: str) -> SomeConfig``
        """
        self._load_func = func
        return func

    def resolve(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """
        Register the RESOLVE function.

        Signature: ``def resolve(config: MyConfig, ctx: ConfigContext) -> MyConfig``
        """
        self._resolve_func = func
        return func

    def build(self, func: Callable[..., list[Any]]) -> Callable[..., list[Any]]:
        """
        Register the BUILD function.

        Signature: ``def build(config: MyConfig) -> list[InitStep]``
        """
        self._build_func = func
        return func

    def detect(self, func: Callable[[], bool]) -> Callable[[], bool]:
        """Register the DETECT predicate."""
        self._detect_func = func
        return func

    def run_detect(self) -> bool:
        """Return True if the detect predicate passes (default: True)."""
        return self._detect_func() if self._detect_func is not None else True

    # ------------------------------------------------------------------
    # Operation decorators
    # ------------------------------------------------------------------

    def operation(
        self,
        operation_number: Optional[str] = None,
        depends_on: Optional[list[str]] = None,
        before: Optional[list[str]] = None,
        name: Optional[str] = None,
        fallback: bool = False,
    ) -> Callable[..., Any]:
        """Decorator to register a model operation.

        ``fallback=True`` marks the op as belonging to the model's *fallback*
        backend (partitioned by :meth:`ModelRuntime.fallback_operations`); by
        default an op is native.
        """

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self._operations.append(
                (
                    func,
                    {
                        "operation_number": operation_number,
                        "depends_on": depends_on,
                        "before": before,
                        "name": name or func.__name__,
                        "fallback": fallback,
                    },
                )
            )
            return func

        return decorator

    def operation_collection(self, func: Callable[..., Operations]) -> Callable[..., Operations]:
        """Decorator for conditional operation dispatch."""
        self._operation_collection_func = func
        return func

    # ------------------------------------------------------------------
    # Interface ownership (model-owned extension points)
    # ------------------------------------------------------------------

    def interface(self, declaration: Callable[..., Any]) -> Hook:
        """Declare a gather hook owned by this model.

        Sugar over :class:`~neofoam.framework.model.extension.Extension`: the
        hook is declared on a model-private extension named after this model,
        so an interface *is* an extension :class:`Hook` — same contribution
        decorator, same dispatch, same parameter resolution. The decorated
        function's ``__name__`` is the interface name and its body receives the
        active contributions' results and combines them (defining the empty
        case)::

            @solutionLoop.interface
            def maxTimeStep(ceilings: Iterable[float]) -> float:
                return min(ceilings, default=VGREAT)

        Returns the :class:`Hook` handle: the ``@<model>.contributes`` target,
        and — used directly as a parameter annotation — the consumer's
        injection marker (the resolver injects the hook bound to the live
        Context; calling it folds the active contributions).
        """
        if self._own_extension is None:
            self._own_extension = Extension(self.name)
        if declaration.__name__ in self._own_extension.hooks:
            raise RuntimeError(
                f"Model '{self.name}': interface '{declaration.__name__}' is already declared."
            )
        return self._own_extension.defines(declaration)

    def contributes(self, target: Hook) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Register an operation-style contribution to *target*, owned by self.

        *target* is a :class:`Hook` — declared via ``@<model>.interface`` or
        ``@<extension>.defines``. The function is recorded against *target*
        and tagged with this contributing model, then returned unchanged. It
        participates in *target*'s dispatch iff this model is active for the
        case.
        """
        if not isinstance(target, Hook):
            raise TypeError(
                f"Model '{self.name}': contributes(...) target must be a "
                "hook declared via @<model>.interface or "
                f"@<extension>.defines, got {type(target).__name__}."
            )

        def decorator(func: Callable[..., _T]) -> Callable[..., _T]:
            return target._register_contribution(func, owner=self)

        return decorator

    @property
    def declared_interfaces(self) -> dict[str, Hook]:
        """The gather hooks this model owns, by name (reachable from the owner)."""
        return dict(self._own_extension.hooks) if self._own_extension is not None else {}

    # ------------------------------------------------------------------
    # Instantiation
    # ------------------------------------------------------------------

    def instantiate(
        self,
        case_dir: Path,
        instance_id: Optional[str] = None,
    ) -> ModelRuntime:
        """
        Create a fresh ModelRuntime for one instance of this spec.

        Loading priority:

        1. If ``@load`` registered → call it (with ``case_dir`` + ``instance_id``).
        2. Else if one or more ``@config(Cls)`` registered → auto-load every
           class via its ``@IOStrategy`` binding. Aggregates to a
           ``SimpleNamespace`` when ≥ 2 are registered.
        3. Else → error.
        """
        if self._load_func is not None:
            config = self._load_func(case_dir, instance_id)
            rt_name = f"{self.name}_{instance_id}" if instance_id else self.name
            return ModelRuntime(spec=self, name=rt_name, config=config)

        if self._config_classes:
            loaded = [
                cls.load(case_dir=case_dir, validate=False)  # type: ignore[attr-defined]
                for cls in self._config_classes
            ]
            config = self._aggregate_configs(loaded)
            if config is None:
                config = loaded[0] if loaded else None
            return ModelRuntime(spec=self, name=self.name, config=config)

        raise ValueError(
            f"ModelSpec '{self.name}': cannot instantiate. "
            "Register @load or @config(Cls) on the spec."
        )

    # ------------------------------------------------------------------
    # Operation building (called by ModelRuntime.operations)
    # ------------------------------------------------------------------

    def _build_operations_for(self, runtime: ModelRuntime) -> list[Operation]:
        """
        Build Operation objects with *runtime* as the ``self`` binding.

        Returns a fresh list per call so multiple runtimes never share
        wrapper state.
        """
        if self._operation_collection_func is not None:
            result = self._operation_collection_func(runtime)
            if isinstance(result, Operations):
                return list(result)
            return result  # type: ignore[no-any-return]

        ops: list[Operation] = []
        for func, metadata in self._operations:
            discovered = _discover_configs_from_signature(func)
            if discovered:
                wrapped = _create_runtime_config_wrapper(func, discovered, runtime)
            else:
                wrapped = wrap_with_dependency_resolution(func, runtime, self._dependency_resolver)

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
                    fallback=metadata.get("fallback", False),
                ),
            )
            ops.append(op)
        return ops

    # ------------------------------------------------------------------
    # Public spec-as-runtime helpers (used by solver entrypoints)
    # ------------------------------------------------------------------

    def build_steps(self) -> list[Any]:
        """Run the registered ``@build`` function with this spec as its binding.

        For specs used as their own runtime (core models composed without
        ``instantiate``): emits the spec's InitSteps, or an empty list when no
        ``@build`` is registered.
        """
        return list(self._build_func(self)) if self._build_func is not None else []

    def build_operations_for(self, runtime: Any) -> list[Operation]:
        """Build this spec's Operations with ``runtime`` as the ``self`` binding.

        ``runtime`` is a ModelRuntime — or the bare spec itself for specs used
        as their own runtime.
        """
        return self._build_operations_for(runtime)

    def wrap_operation(self, func: Callable[..., Any], runtime: Any) -> Callable[..., Any]:
        """Wrap ``func`` with this spec's dependency resolution.

        For ``@operation_collection`` bodies, which bypass the spec's default
        operation wrapping and must wrap each exposed operation themselves.
        """
        return wrap_with_dependency_resolution(func, runtime, self._dependency_resolver)

    # ------------------------------------------------------------------
    # Plugin registration
    # ------------------------------------------------------------------

    def register_with(self, plugin_interface: type) -> "ModelSpec":
        """Register this ModelSpec with a PluginSystem interface (idempotent by name)."""
        registry = PluginSystem.get_registered(plugin_interface.__name__)
        if registry is not None and self.name in {
            plugin_cls.__name__ for plugin_cls in registry.plugin_registry
        }:
            # Already registered (e.g. a module re-import): keep a single catalog
            # entry so the discriminated union does not collide on the model_type
            # literal.
            return self

        wrapper_class = type(
            self.name,
            (BaseModel,),
            {
                "__module__": plugin_interface.__module__,
                "__annotations__": {"model_type": Literal[self.name]},
                "model_type": self.name,
                "get_model_instance": lambda self_wrapper: self,
            },
        )

        if not hasattr(plugin_interface, "register"):
            raise TypeError(
                f"Plugin interface '{plugin_interface}' does not have a 'register' method"
            )
        plugin_interface.register(wrapper_class)
        return self

    def labeled(self, label: str) -> "ModelSpec":
        """Set this model's human-readable display label for UIs.

        Chainable (e.g. after :meth:`register_with`). Independent of whether the
        model's family is required; defaults to the model name.
        """
        self.label = label
        return self

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_snake_case(name: str) -> str:
        return _snake_case(name)


def Model(name: str) -> ModelSpec:
    """Factory: create a named ModelSpec."""
    return ModelSpec(name)
