# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The declarative front door: the case's ``system/postProcess`` spec file.

A table is declared as open mappings
(``source``/``pipeline``/``write_control``/``writer``) resolved against their
registries at resolve time — mirroring
:class:`~neofoam.framework.tools.graph.PreprocessConfig` — so registering a new
node needs no change here and no union is frozen at import. YAML, YML and JSON
are the same tree; :class:`~neofoam.io.dictfile.DictFile` picks the backend by suffix::

    resolve_table(load_config(case_dir).tables[0])
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, cast

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from neofoam.algorithms.field_writer.write_control import WriteControl
from neofoam.core.plugin_system import PluginSystem
from neofoam.io import YAML, BaseConfig, IOStrategy
from neofoam.postprocess.node import Node, Pipeline, Source
from neofoam.postprocess.table import Table
from neofoam.postprocess.writers.writer import TableWriter

#: The spec files tried in order; the first that exists is the case's.
SPEC_FILES = (
    "system/postProcess.yaml",
    "system/postProcess.yml",
    "system/postProcess.json",
)

#: Evaluate every step unless the table says otherwise.
DEFAULT_WRITE_CONTROL: dict[str, Any] = {"write_control_type": "timeStep", "interval": 1}

#: Write a CSV file unless the table says otherwise.
DEFAULT_WRITER: dict[str, Any] = {"type": "csv"}


class TableSpec(BaseModel):
    """One declared table: where the data comes from, what happens to it, how often, in what format.

    The four mappings are validated only when they are resolved (against the
    ``Source``, ``Node``, ``WriteControl`` and ``TableWriter`` plugin families), which
    is what keeps a user's own node usable from the spec file. The table's own
    keys are fixed, so an unknown one is refused here::

        TableSpec(name="volume_p", source={"type": "internal", "field": "p"},
                  pipeline=[{"type": "volIntegrate"}])
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    source: dict[str, Any]
    pipeline: list[dict[str, Any]] = []
    write_control: dict[str, Any] = DEFAULT_WRITE_CONTROL
    writer: dict[str, Any] = DEFAULT_WRITER

    @field_validator("name")
    @classmethod
    def _one_file_name(cls, name: str) -> str:
        """Refuse a name carrying a directory part, which would write outside the case.

        :class:`~neofoam.postprocess.model.PostProcessor` opens a table's writer on
        ``<case>/postProcessing/<name>``, so an absolute name or a ``..`` segment
        escapes the case — and the workspace root that
        :func:`~neofoam.mcp.tools.save_post` confines an agent's spec to. The script
        front door already takes ``Path(filename).stem``; this is the same rule for a
        declared table, stated where the spec is read.
        """
        if name != Path(name).name or name in {"", ".", ".."} or "\\" in name:
            raise ValueError(
                f"postProcess table name {name!r} is not a file name — a table is "
                f"written to postProcessing/<name>.csv, so the name carries no "
                f"directory part"
            )
        return name


@IOStrategy(YAML("system/postProcess.yaml"))
class PostProcessConfig(BaseConfig):
    """The case's declared post-processing tables; no file ⇒ no tables.

    ``tables`` is the file's only key, so an unknown one is refused rather than
    read as an empty declaration.
    """

    model_config = ConfigDict(extra="forbid")

    tables: list[TableSpec] = []


def spec_file(case_dir: Path) -> Optional[Path]:
    """The case's spec file — the first of the YAML/YML/JSON names that exists."""
    for name in SPEC_FILES:
        path = Path(case_dir) / name
        if path.is_file():
            return path
    return None


def load_config(case_dir: Path) -> PostProcessConfig:
    """The case's declared tables, or an empty config when it declares no file."""
    path = spec_file(case_dir)
    if path is None:
        return PostProcessConfig()
    return PostProcessConfig.load(case_dir=path)


def resolve_table(spec: TableSpec) -> Table:
    """Turn one declared table into the very Table a case script builds.

    Each mapping picks its class through the family's discriminated union, so an
    unknown ``type`` is reported against the table (and the pipeline position)
    it was declared in, not as a bare pydantic union error.
    """
    steps = [_resolve_node(spec, index, entry) for index, entry in enumerate(spec.pipeline)]
    return Table(
        name=spec.name,
        pipeline=Pipeline(source=_resolve_source(spec), steps=steps),
        write_control=_resolve_write_control(spec),
        writer=_resolve_writer(spec),
    )


def _resolve_source(spec: TableSpec) -> Source:
    """The table's source, selected from the ``Source`` family by its ``type``."""
    try:
        return cast(Source, cast(Any, Source).create(source=spec.source).source)
    except ValidationError as exc:
        raise ValueError(
            f"postProcess table {spec.name!r}: cannot resolve source {spec.source!r}; "
            f"registered source types: {_registered_types('Source')}"
        ) from exc


def _resolve_node(spec: TableSpec, index: int, entry: dict[str, Any]) -> Node:
    """One pipeline step, selected from the ``Node`` family by its ``type``."""
    try:
        return cast(Node, cast(Any, Node).create(node=entry).node)
    except ValidationError as exc:
        raise ValueError(
            f"postProcess table {spec.name!r}: cannot resolve pipeline[{index}] {entry!r}; "
            f"registered node types: {_registered_types('Node')}"
        ) from exc


def _resolve_write_control(spec: TableSpec) -> WriteControl:
    """The table's cadence, selected from the ``WriteControl`` family."""
    try:
        return cast(WriteControl, cast(Any, WriteControl).create(policy=spec.write_control).policy)
    except ValidationError as exc:
        raise ValueError(
            f"postProcess table {spec.name!r}: cannot resolve write_control "
            f"{spec.write_control!r}; registered write control types: "
            f"{_registered_types('WriteControl')}"
        ) from exc


def _resolve_writer(spec: TableSpec) -> TableWriter:
    """The table's output format, selected from the ``TableWriter`` family."""
    try:
        return cast(TableWriter, cast(Any, TableWriter).create(writer=spec.writer).writer)
    except ValidationError as exc:
        raise ValueError(
            f"postProcess table {spec.name!r}: cannot resolve writer {spec.writer!r}; "
            f"registered writer types: {_registered_types('TableWriter')}"
        ) from exc


def _registered_types(family: str) -> list[str]:
    """The discriminator strings a plugin family currently answers to."""
    registry = PluginSystem.get_registered(family)
    if registry is None:
        return []
    return sorted(
        str(plugin_cls.model_fields[registry.discriminator].default)
        for plugin_cls in registry.plugin_registry
        if registry.discriminator in plugin_cls.model_fields
    )
