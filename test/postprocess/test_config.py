# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Spec for the declarative front door: system/postProcess.{yaml,yml,json}.

Every input is a checked-in case under ``cases/`` — the spec files are exactly
what a user writes. ``one_table`` ships the YAML and the JSON spelling of the
same table, which is what makes "the format is not part of the meaning"
testable: both must resolve to the very same
:class:`~neofoam.postprocess.node.Pipeline`, with the concrete node classes
(pydantic must not re-coerce a subclass to its base). No OpenFOAM: resolution
touches the registries only, never a mesh.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from pydantic import ValidationError

from neofoam.algorithms.field_writer.write_control import IntervalWriteControl
from neofoam.postprocess.config import (
    PostProcessConfig,
    TableSpec,
    load_config,
    resolve_table,
    spec_file,
)
from neofoam.postprocess.nodes.aggregators import VolIntegrate
from neofoam.postprocess.sources.fields import InternalField

CASES = Path(__file__).parent / "cases"


def _spec(rel_path: str, index: int = 0) -> TableSpec:
    """The table a checked-in spec file declares, loaded the way a case loads it."""
    return PostProcessConfig.load(case_dir=CASES / rel_path).tables[index]


def test_yaml_and_json_twins_resolve_to_the_same_pipeline() -> None:
    from_yaml = resolve_table(_spec("one_table/system/postProcess.yaml"))
    from_json = resolve_table(_spec("one_table/system/postProcess.json"))
    assert from_yaml.pipeline == from_json.pipeline


def test_resolved_pipeline_holds_the_concrete_plugin_classes() -> None:
    table = resolve_table(_spec("one_table/system/postProcess.yaml"))
    assert isinstance(table.pipeline.source, InternalField)
    assert [type(step) for step in table.pipeline.steps] == [VolIntegrate]


def test_resolved_source_keeps_the_declared_field() -> None:
    source = resolve_table(_spec("one_table/system/postProcess.yaml")).pipeline.source
    assert isinstance(source, InternalField)
    assert source.field == "p"


def test_table_without_write_control_runs_every_step() -> None:
    table = resolve_table(_spec("cadence/system/postProcess.yaml", index=0))
    assert table.write_control == IntervalWriteControl(interval=1)


def test_declared_write_control_maps_to_its_policy() -> None:
    table = resolve_table(_spec("cadence/system/postProcess.yaml", index=1))
    assert table.write_control == IntervalWriteControl(interval=2)


def test_unknown_node_type_names_the_table_and_the_pipeline_position() -> None:
    spec = _spec("unknown_node/system/postProcess.yaml")
    with pytest.raises(ValueError, match=r"'volume_p'.*pipeline\[1\]"):
        resolve_table(spec)


def test_unknown_node_error_lists_the_types_that_would_work() -> None:
    spec = _spec("unknown_node/system/postProcess.yaml")
    with pytest.raises(ValueError, match=r"notANode.*registered node types:.*volIntegrate"):
        resolve_table(spec)


@pytest.mark.parametrize(
    ("index", "message"),
    [
        (0, r"'typo_source'.*region"),
        (1, r"'typo_node'.*pipeline\[0\].*origins"),
        (2, r"'typo_writer'.*delimiter"),
    ],
    ids=["source", "node", "writer"],
)
def test_an_unknown_key_names_the_table_it_was_declared_in(index: int, message: str) -> None:
    spec = _spec("typo_key/system/postProcess.yaml", index=index)
    with pytest.raises(ValueError, match=message):
        resolve_table(spec)


@pytest.mark.parametrize(
    ("case", "unknown_key"),
    [("typo_table_key", "pipelines"), ("typo_file_key", "table")],
    ids=["table_key", "file_key"],
)
def test_an_unknown_key_outside_the_plugin_mappings_is_refused_when_the_file_is_read(
    case: str, unknown_key: str
) -> None:
    with pytest.raises(ValidationError, match=unknown_key):
        PostProcessConfig.load(case_dir=CASES / case / "system/postProcess.yaml")


def test_case_without_a_spec_file_declares_no_tables(tmp_path: Path) -> None:
    assert spec_file(tmp_path) is None
    assert load_config(tmp_path).tables == []


def test_json_is_picked_up_when_the_yaml_spelling_is_absent(tmp_path: Path) -> None:
    case_dir = tmp_path / "one_table"
    shutil.copytree(CASES / "one_table", case_dir)
    (case_dir / "system/postProcess.yaml").unlink()

    assert spec_file(case_dir) == case_dir / "system/postProcess.json"
    assert load_config(case_dir).tables == [_spec("one_table/system/postProcess.yaml")]
