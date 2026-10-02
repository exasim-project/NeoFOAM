# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The declarative front door: the case's ``system/setFields`` spec file.

A region is declared as an open mapping resolved against the ``Node`` registry at
resolve time — the same mechanism
:data:`~neofoam.postprocess.nodes.selectors.NestedSelector` uses — so a selector a
case script registers is usable from the spec file and no union is frozen at
import. YAML, YML and JSON are the same tree;
:class:`~neofoam.io.dictfile.DictFile` picks the backend by suffix::

    resolve_regions(load_config(case_dir))
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from neofoam.core.plugin_system import PluginSystem
from neofoam.io import YAML, BaseConfig, IOStrategy
from neofoam.postprocess.nodes.selectors import NestedSelector, Selector

#: The spec files tried in order; the first that exists is the case's.
SPEC_FILES = (
    "system/setFields.yaml",
    "system/setFields.yml",
    "system/setFields.json",
)

#: What one field is set to: a scalar (a ``volScalarField``) or a 3-vector (a
#: ``volVectorField``). The value's shape picks the field type, so a field set to
#: ``0`` and a field set to ``[0, 0, 0]`` are two different fields.
RegionValue = Union[float, tuple[float, float, float]]

#: One region mapping, resolved against the ``Node`` union as it stands at
#: validation time — never at import.
_SELECTOR = TypeAdapter(NestedSelector)


class RegionSpec(BaseModel):
    """One declared region and the values every field takes inside it.

    The ``region`` mapping is validated only when it is resolved (against the
    ``Node`` plugin family), which is what keeps a case's own selector usable from
    the spec file. The entry's own keys are fixed, so an unknown one is refused
    here::

        RegionSpec(region={"type": "box", "min": [0, 0, 0], "max": [1, 1, 1]},
                   values={"alpha.water": 1.0})
    """

    model_config = ConfigDict(extra="forbid")

    region: dict[str, Any]
    values: dict[str, RegionValue]


@IOStrategy(YAML("system/setFields.yaml"))
class SetFieldsConfig(BaseConfig):
    """The case's declared field initialisation; no file ⇒ nothing declared here.

    ``defaults`` is written over the whole internal field first, then each entry of
    ``regions`` over the cells its region selects, in order. Those are the file's
    only keys, so an unknown one is refused rather than read as an empty
    declaration.
    """

    model_config = ConfigDict(extra="forbid")

    defaults: dict[str, RegionValue] = {}
    regions: list[RegionSpec] = []


def spec_file(case_dir: Path) -> Optional[Path]:
    """The case's spec file — the first of the YAML/YML/JSON names that exists."""
    for name in SPEC_FILES:
        path = Path(case_dir) / name
        if path.is_file():
            return path
    return None


def load_config(case_dir: Path) -> SetFieldsConfig:
    """The case's declared regions, or an empty config when it declares no file."""
    path = spec_file(case_dir)
    if path is None:
        return SetFieldsConfig()
    return SetFieldsConfig.load(case_dir=path)


def resolve_regions(config: SetFieldsConfig) -> list[tuple[Selector, dict[str, RegionValue]]]:
    """Turn the declared regions into the ``(selector, values)`` pairs apply consumes.

    Each mapping picks its class through the ``Node`` family's discriminated union,
    so an unknown or non-selector ``type`` is reported against the position it was
    declared in, not as a bare pydantic union error.
    """
    return [
        (_resolve_selector(index, spec), spec.values) for index, spec in enumerate(config.regions)
    ]


def _resolve_selector(index: int, spec: RegionSpec) -> Selector:
    """One region, selected from the ``Node`` family by its ``type``."""
    try:
        return _SELECTOR.validate_python(spec.region)
    except (ValidationError, ValueError) as exc:
        raise ValueError(
            f"setFields regions[{index}]: cannot resolve region "
            f"{spec.region.get('type')!r}; registered selector types: "
            f"{_registered_selectors()}"
        ) from exc


def _registered_selectors() -> list[str]:
    """The discriminator strings the ``Node`` family's selectors answer to."""
    registry = PluginSystem.get_registered("Node")
    if registry is None:
        return []
    return sorted(
        str(plugin_cls.model_fields[registry.discriminator].default)
        for plugin_cls in registry.plugin_registry
        if issubclass(plugin_cls, Selector) and registry.discriminator in plugin_cls.model_fields
    )
