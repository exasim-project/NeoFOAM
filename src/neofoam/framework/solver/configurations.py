# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""A solver's config schema, case-free.

:func:`configurations` returns a :class:`Configurations` view over every
``BaseConfig`` class a solver may consume — its own declared configs plus
the configs of every member of every bound model family — **without a case
directory**. The classes are pydantic models, so the view doubles as the
schema an agent (e.g. via Pydantic AI) fills and saves to scaffold a case.

Solver-agnostic: it walks ``solver._config_classes`` and
``solver.model_specs`` through :func:`neofoam.io.collect_config_classes`
and runs no detection.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Union, cast

from pydantic import BaseModel, create_model

from neofoam.io.base import BaseConfig


def _snake_case(name: str) -> str:
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def _is_field_schema(cls: type[BaseConfig]) -> bool:
    """True if ``cls`` is a synthesised ``0/<name>`` field schema.

    The check is on the registered IO path: per-field schemas bind to
    ``0/<name>`` via :func:`neofoam.fields.schema.schema_for`; dictionary
    configs bind to ``constant/...`` / ``system/...`` paths. A class with
    no ``io_config`` cannot be a field schema.
    """
    io = getattr(cls, "io_config", None)
    if io is None:
        return False
    return bool(io.file.startswith("0/"))


@dataclass(frozen=True)
class Configurations:
    """Case-free view over a solver's config classes (all pydantic models)."""

    solver: Any
    classes: list[type[BaseConfig]]

    # -- listing -------------------------------------------------------

    @property
    def names(self) -> list[str]:
        """Class names, in declaration order."""
        return [cls.__name__ for cls in self.classes]

    @property
    def fields(self) -> list[type[BaseConfig]]:
        """Synthesised ``0/<name>`` field schemas only.

        These are the per-field schemas surfaced via ``Model.field(...)``
        registrations; they bind to ``0/<name>`` and carry the
        ``(dimensions, internalField, boundaryField)`` shape. Use this
        when a caller only wants to walk the field side (e.g. the
        ``load_fields`` / ``save_fields`` orchestrators).
        """
        return [cls for cls in self.classes if _is_field_schema(cls)]

    @property
    def dicts(self) -> list[type[BaseConfig]]:
        """Dictionary configs only — ``constant/...`` / ``system/...``.

        The complement of :attr:`fields`. Useful when the caller wants
        to keep the original ``configurations(solver)`` semantics
        (multi-owner dict files merged via ``write_configs``).
        """
        return [cls for cls in self.classes if not _is_field_schema(cls)]

    def __iter__(self) -> Iterator[type[BaseConfig]]:
        return iter(self.classes)

    def __len__(self) -> int:
        return len(self.classes)

    def __getitem__(self, name: str) -> type[BaseConfig]:
        """Look a config class up by its ``__name__``."""
        for cls in self.classes:
            if cls.__name__ == name:
                return cls
        raise KeyError(f"{name!r} not in {self.solver.name} configs: {self.names}")

    # -- build & validate ---------------------------------------------

    def new(self, name: str, **values: Any) -> Any:
        """Construct + pydantic-validate one config from values alone."""
        return self[name](**values)

    def json_schema(self) -> dict[str, dict[str, Any]]:
        """JSON Schema per config class, keyed by class name."""
        return {cls.__name__: cls.model_json_schema() for cls in self.classes}

    def as_output_model(self, model_name: str = "CaseSpec") -> type[BaseModel]:
        """An aggregate pydantic model with one field per config class.

        Suitable as a Pydantic AI ``output_type``: each field is named by
        the snake-case class name and typed as that config class, so an
        agent fills the whole case in one structured response.
        """
        fields: dict[str, Any] = {
            _snake_case(cls.__name__): (cls, ...) for cls in self.classes
        }
        return create_model(model_name, **fields)

    # -- persist -------------------------------------------------------

    def save(self, configs: Any, *, case_dir: Union[Path, str]) -> list[Path]:
        """Write config instances to ``case_dir`` (delegates to ``save_configs``)."""
        from neofoam.io import save_configs

        return save_configs(configs, case_dir=case_dir)


def configurations(solver: Any) -> Configurations:
    """The case-free config schema of ``solver`` as a :class:`Configurations`.

    Walks the solver's own declared configs and every member of every bound
    model family (``solver.model_specs``), deduped by identity. Runs no
    detection and needs no case directory.
    """
    from neofoam.io import collect_config_classes

    classes = collect_config_classes([solver, *solver.model_specs])
    return Configurations(
        solver=solver, classes=cast("list[type[BaseConfig]]", classes)
    )


@dataclass(frozen=True)
class ModelEntry:
    """One model a solver consumes, for a "select models" UI.

    ``required`` models (the active member of each required family) are always
    on and can't be deselected; optional models can be deselected. ``dicts`` /
    ``fields`` are the config classes the model owns (derived from its spec).
    """

    name: str
    label: str
    required: bool
    dicts: list[type[BaseConfig]]
    fields: list[type[BaseConfig]]


def model_catalog(solver: Any) -> list[ModelEntry]:
    """Every model of ``solver`` with its required flag + owned configs/fields.

    Members of required families (:attr:`SolverSpec.required_model_specs`) yield
    ``required=True`` entries; members of optional families
    (:attr:`SolverSpec.optional_model_specs`) yield ``required=False``. ``label``
    comes from ``spec.label``. Ownership comes from each model spec — no
    name-matching. Case-free; runs no detection. The single required/optional
    source for UIs. (The solver's own configs — e.g. ``controlDict`` — are not
    models; they always apply and are listed by :func:`configurations`.)
    """
    from neofoam.io import collect_config_classes

    out: list[ModelEntry] = []
    for required, families in (
        (True, solver.required_model_specs),
        (False, solver.optional_model_specs),
    ):
        for family in families:
            for spec in family.all_specs():
                owned = cast("list[type[BaseConfig]]", collect_config_classes([spec]))
                out.append(
                    ModelEntry(
                        name=spec.name,
                        label=spec.label,
                        required=required,
                        dicts=[c for c in owned if not _is_field_schema(c)],
                        fields=[c for c in owned if _is_field_schema(c)],
                    )
                )
    return out
