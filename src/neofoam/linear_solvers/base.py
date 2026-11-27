# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The linear-solver and preconditioner plugin families, shared by every backend."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Callable, ClassVar, Iterator, Optional, Union

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    RootModel,
    SerializationInfo,
    model_serializer,
    model_validator,
)

from neofoam.core.field_metadata import advanced
from neofoam.core.plugin_system import PluginSystem

_LEADING_KEYS = ("solver", "preconditioner", "configFile")
_CONTROL_KEYS = ("tolerance", "relTol", "maxIter", "minIter")

# Filled by each backend module.
_FILE_READERS: dict[str, Callable[[Any], Any]] = {}
_ROLE_RULES: dict[tuple[str, str], Callable[[Any], Any]] = {}
_ROLE_SCHEMAS: dict[str, Callable[[str], dict[str, Any]]] = {}
# Called after a plugin change, so configs built on the families can rebuild.
_refresh_listeners: list[Callable[[], None]] = []
_entry_types: dict[tuple[str, Optional[str]], type[RootModel[Any]]] = {}


class LinearSolverConversionWarning(UserWarning):
    """A key of an existing ``fvSolution`` block was dropped when it was read.

    Only keys NeoN already ignores are dropped; a value it honours raises instead. Catch
    it with :func:`warnings.catch_warnings` to show it next to the loaded case. Warnings
    filters are process-global, so this is not thread-safe.

    Example::

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", LinearSolverConversionWarning)
            config = Slice.load(case_dir=case)
    """


def _without_null_defaults(schema: Any, handler: Any) -> Any:
    """The object schema without ``default: null`` on its properties."""
    for prop in handler.resolve_ref_schema(schema).get("properties", {}).values():
        if "default" in prop and prop["default"] is None:
            del prop["default"]
    return schema


@PluginSystem.register(discriminator_variable="entry", discriminator="solver")
class LinearSolver(BaseModel):
    """A linear solver of one backend, selected by the ``solver`` key.

    Subclass a backend base such as :class:`GinkgoSolver` and decorate it with
    ``@LinearSolver.register``. ``backend`` is fixed by the class and never appears in a
    form or file. The fields here mean the same on every backend; the backend's
    dictionary is built only at run start, by ``runtime_settings``. Registration is not
    thread-safe.

    Example::

        @LinearSolver.register
        class Fgmres(GinkgoSolver):
            solver: Literal["Fgmres"] = "Fgmres"
    """

    model_config = ConfigDict(extra="forbid")

    backend: ClassVar[str]
    shared_backends: ClassVar[tuple[str, ...]] = ()
    register: ClassVar[Callable[[type[Any]], type[Any]]]

    solver: str
    tolerance: Optional[float] = None
    relTol: Optional[float] = None
    maxIter: Optional[int] = None
    minIter: Optional[int] = advanced(None)

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema: Any, handler: Any) -> Any:
        """Drop ``default: null``, which JSONForms would write into the data."""
        return _without_null_defaults(handler(core_schema), handler)

    @model_serializer(mode="wrap")
    def _dump(self, handler: Any) -> dict[str, Any]:
        """Dump the set keys, the selecting keys first."""
        data = {key: value for key, value in handler(self).items() if value is not None}
        first = {key: data[key] for key in _LEADING_KEYS if key in data}
        return {**first, **data}

    def runtime_settings(self, case_dir: Path) -> dict[str, Any]:
        """The backend's in-memory settings for this block."""
        raise NotImplementedError(f"{type(self).__name__} has no runtime translation")


@PluginSystem.register(discriminator_variable="entry", discriminator="type")
class Preconditioner(BaseModel):
    """A preconditioner of one backend, selected by ``type``.

    Register one subclass to add a preconditioner: it joins the ``preconditioner`` union of
    every solver of its backend, also on slices built earlier. Declared fields are its
    options. With none set it is written as a word, otherwise as a dictionary; form data
    is always the object. Registration is not thread-safe.

    Example::

        @Preconditioner.register
        class Sor(GinkgoPreconditioner):
            mapper_word: ClassVar[str] = "SOR"
            type: Literal["Sor"] = "Sor"
    """

    model_config = ConfigDict(extra="forbid")

    backend: ClassVar[str]
    # The family's tags are unique, so a backend with the same word reuses the plugin.
    shared_backends: ClassVar[tuple[str, ...]] = ()
    register: ClassVar[Callable[[type[Any]], type[Any]]]
    mapper_word: ClassVar[str]
    native_block: ClassVar[Optional[dict[str, Any]]] = None
    drops_unknown_keys: ClassVar[bool] = False
    # Options an OpenFOAM-named block may set; the rest stay dropped so its behaviour holds.
    foam_options: ClassVar[tuple[str, ...]] = ()

    type: str

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema: Any, handler: Any) -> Any:
        """Drop ``default: null``, which JSONForms would write into the data."""
        return _without_null_defaults(handler(core_schema), handler)

    @model_serializer(mode="wrap")
    def _dump(self, handler: Any, info: SerializationInfo) -> Any:
        """The object form, or the bare word in a file when no option is set."""
        is_file = isinstance(info.context, dict) and info.context.get("format") == "openfoam"
        if is_file and not self.mapper_entries():
            return self.type
        return {key: value for key, value in handler(self).items() if value not in (None, {})}

    def mapper_entries(self) -> dict[str, Any]:
        """The set options, as the flat solver-block keys the mapper reads."""
        fields = type(self).model_fields
        values = {key: _plain(getattr(self, key)) for key in _option_keys(type(self))}
        return {
            key: value
            for key, value in values.items()
            if value != {} and value != fields[key].default
        }

    def runtime_settings(self) -> dict[str, Any]:
        """The solver-block keys the run-start mapper reads for this preconditioner."""
        return {"preconditioner": self.mapper_word, **self.mapper_entries()}

    @classmethod
    def incompatibility(cls, solver: str) -> Optional[str]:
        """Why this preconditioner cannot run with ``solver``, or ``None``."""
        return None


def _family_plugins(family: str, backend: Optional[str] = None) -> list[Any]:
    """A family's plugins registered now, of one backend when given, in registration order."""
    registry = PluginSystem.get_registered(family)
    assert registry is not None
    return [
        plugin
        for plugin in registry.plugin_registry
        if backend is None or backend in (plugin.backend, *plugin.shared_backends)  # type: ignore[attr-defined]
    ]


def _registered_plugins(backend: Optional[str] = None) -> list[type[LinearSolver]]:
    """The linear-solver plugins registered now."""
    return _family_plugins("LinearSolver", backend)


def registered_preconditioners(backend: Optional[str] = None) -> list[type[Preconditioner]]:
    """The preconditioner plugins registered now."""
    return _family_plugins("Preconditioner", backend)


def _plugin_tag(plugin: type[BaseModel], key: str = "solver") -> str:
    """The plugin's discriminator value."""
    tag = plugin.model_fields[key].default
    if not isinstance(tag, str):
        raise ValueError(f"{plugin.__name__} needs a Literal `{key}` with a string default")
    return tag


def _option_keys(plugin: type[Preconditioner]) -> tuple[str, ...]:
    """The option fields a preconditioner declares."""
    return tuple(key for key in plugin.model_fields if key != "type")


def _plain(value: Any) -> Any:
    """An option value with a nested model as its set keys."""
    return value.model_dump(exclude_none=True) if isinstance(value, BaseModel) else value


def _role_rule(backend: str, role: Optional[str]) -> Optional[Callable[[Any], Any]]:
    """The per-field block rule of a backend, if it has one for this role."""
    return None if role is None else _ROLE_RULES.get((backend, role))


def _role_schema(backend: str, role: str) -> dict[str, Any]:
    """The schema keywords the wizard reads for one field role."""
    return _ROLE_SCHEMAS[backend](role)


def _role_entry(backend: str, role: str) -> type[RootModel[Any]]:
    """The entry type of one field role, applying that role's rule before validation."""
    rule = _role_rule(backend, role)
    assert rule is not None

    def apply_rule(cls: type, block: Any) -> Any:
        return rule(block)

    base = linear_solver_entry(backend)
    namespace = {
        "__module__": __name__,
        # The form keeps its "add a solver" row only while all entries read the same.
        "__doc__": base.__doc__,
        "model_config": ConfigDict(json_schema_extra=_role_schema(backend, role)),
        "_apply_rule": model_validator(mode="before")(classmethod(apply_rule)),  # type: ignore[arg-type]
    }
    metaclass: Any = type(base)
    entry: type[RootModel[Any]] = metaclass(
        f"{backend.capitalize()}{role.capitalize()}Entry", (base,), namespace
    )
    return entry


def linear_solver_entry(backend: str, role: Optional[str] = None) -> type[RootModel[Any]]:
    """The typed ``solvers.<field>`` entry over one backend's registered plugins.

    Use it to validate or dump one block outside a config; the plugin is ``.root``. A
    ``RootModel``, not a bare union, so the OpenFOAM reader does not reorder the block by
    the first arm's fields. ``role`` (``"pressure"``/``"momentum"``) adds that field's
    rule where the backend has one. The class is replaced when a plugin registers.

    Example::

        entry = linear_solver_entry("ginkgo").model_validate({"solver": "Cg"})
    """
    if _role_rule(backend, role) is None:
        role = None
    if (backend, role) in _entry_types:
        return _entry_types[(backend, role)]
    if role is not None:
        _entry_types[(backend, role)] = _role_entry(backend, role)
        return _entry_types[(backend, role)]
    if (backend, None) not in _entry_types:
        arms: Any = Annotated[
            Union[tuple(_registered_plugins(backend))],  # noqa: UP007
            Field(discriminator="solver"),
        ]
        reader = _FILE_READERS.get(backend, lambda block: block)

        class LinearSolverEntry(RootModel[Annotated[arms, BeforeValidator(reader)]]):
            """One linear-solver block."""

        _entry_types[(backend, None)] = LinearSolverEntry
    return _entry_types[(backend, None)]


def _subclasses(cls: type[LinearSolver]) -> Iterator[type[LinearSolver]]:
    """Every subclass of ``cls``."""
    for subclass in cls.__subclasses__():
        yield subclass
        yield from _subclasses(subclass)


def _retype_solver(solver: type[LinearSolver]) -> None:
    """Point a solver's ``preconditioner`` at its backend's registered plugins."""
    info = solver.model_fields["preconditioner"]
    arms: Any = Union[tuple(registered_preconditioners(solver.backend))]  # noqa: UP007
    info.annotation = arms
    info.discriminator = "type"
    solver.model_rebuild(force=True)


def refresh_linear_solver_slices() -> None:
    """Rebuild the typed NeoN sections from the registered plugins.

    Covers both the solver and the preconditioner family. Registration calls this; call it
    yourself only after removing a plugin with ``PluginSystem.remove_plugin_model``. Not
    thread-safe.

    Example::

        PluginSystem.remove_plugin_model("LinearSolver", MySolver)
        refresh_linear_solver_slices()
    """
    for solver in _subclasses(LinearSolver):
        if "preconditioner" in solver.model_fields:
            _retype_solver(solver)
    _entry_types.clear()
    for listener in _refresh_listeners:
        listener()


def _checked_register(family: Any, key: str) -> Callable[[type[Any]], type[Any]]:
    """A family's ``register`` with a family-unique tag and a rebuild of the typed sections."""
    add_plugin = family.register

    def register(plugin: type[Any]) -> type[Any]:
        tag = _plugin_tag(plugin, key)
        for existing in _family_plugins(family.__name__):
            if _plugin_tag(existing, key) == tag:
                raise ValueError(
                    f"{family.__name__} {key} {tag!r} of {plugin.__name__} is already "
                    f"registered by {existing.__name__}"
                )
        add_plugin(plugin)
        refresh_linear_solver_slices()
        return plugin

    return register


LinearSolver.register = _checked_register(LinearSolver, "solver")
Preconditioner.register = _checked_register(Preconditioner, "type")


def _keep_compatible_arms(schema: dict[str, Any], backend: str, solver: str) -> None:
    """Narrow a solver schema's ``preconditioner`` union to the arms that accept ``solver``."""
    field = schema["properties"]["preconditioner"]
    usable = {
        _plugin_tag(plugin, "type")
        for plugin in registered_preconditioners(backend)
        if plugin.incompatibility(solver) is None
    }
    mapping = {tag: ref for tag, ref in field["discriminator"]["mapping"].items() if tag in usable}
    field["discriminator"]["mapping"] = mapping
    field["oneOf"] = [arm for arm in field["oneOf"] if arm["$ref"] in mapping.values()]


class _PreconditionedSolver(LinearSolver):
    """A solver whose ``preconditioner`` is its backend's plugin union."""

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        """Type the new solver's ``preconditioner`` over the registered plugins."""
        super().__pydantic_init_subclass__(**kwargs)
        _retype_solver(cls)

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema: Any, handler: Any) -> Any:
        """Publish only the preconditioners this solver can run with."""
        schema = super().__get_pydantic_json_schema__(core_schema, handler)
        tag = cls.model_fields["solver"].default
        if isinstance(tag, str):
            _keep_compatible_arms(handler.resolve_ref_schema(schema), cls.backend, tag)
        return schema
