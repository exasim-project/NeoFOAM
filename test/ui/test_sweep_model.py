# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Pure-model tests for the Parameters step (no trame server, no VueFlow)."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from neofoam.ui.sweep_model import (
    DimensionState,
    SweepModel,
    series_values,
    variant_name,
)


class _Transport(BaseModel):
    transportModel: str = "Newtonian"
    nu: float = 1e-5


_SCHEMA = {
    "type": "object",
    "properties": {"transportModel": {"type": "string"}, "nu": {"type": "number"}},
}
_CLASSES = {"transport": _Transport}


def _model_with_transport(fields=None) -> SweepModel:
    m = SweepModel()
    m.add_dimension(
        "transport",
        title="transportProperties",
        schema=_SCHEMA,
        seed={"transportModel": "Newtonian", "nu": 1e-5},
        fields=fields,
    )
    return m


def test_add_and_remove_dimension() -> None:
    m = SweepModel()
    assert not m.has("transport")
    m.add_dimension("transport", title="t", schema=_SCHEMA, seed={"nu": 1e-5})
    assert m.has("transport")
    assert m.dirty
    assert m.to_dimensions() == {"transport": {"base": {"nu": 1e-5}}}
    with pytest.raises(ValueError, match="already on the canvas"):
        m.add_dimension("transport", title="t", schema=_SCHEMA, seed={})
    m.remove_dimension("transport")
    assert not m.has("transport")
    with pytest.raises(ValueError, match="not on the canvas"):
        m.remove_dimension("transport")


class _Block(BaseModel):
    scale: float = 1.0


class _Snappy(BaseModel):
    refinement: int = 1


_MESH_CLASSES = {
    "transport": _Transport,
    "block_mesh_dict_config": _Block,
    "snappy_hex_mesh_dict_config": _Snappy,
}
_BLOCK_SCHEMA = {"type": "object", "properties": {"scale": {"type": "number"}}}
_SNAPPY_SCHEMA = {"type": "object", "properties": {"refinement": {"type": "integer"}}}


def test_add_mesh_source_creates_and_extends_keyed_mesh_dimension() -> None:
    m = SweepModel()
    assert m.mesh_sources() == []

    m.add_mesh_source(
        "block_mesh_dict_config",
        title="blockMeshDict",
        schema=_BLOCK_SCHEMA,
        seed={"scale": 1.0},
    )
    assert m.has("mesh")
    assert m.dims["mesh"].kind == "mesh"
    assert m.mesh_sources() == ["block_mesh_dict_config"]
    # The variant payload is config-name-keyed (the shape export/runner consume).
    assert m.to_dimensions()["mesh"] == {
        "base": {"block_mesh_dict_config": {"scale": 1.0}}
    }

    # A second source is grafted onto every existing variant + the combined schema.
    m.variant_add("mesh")  # base + variant-1, each carrying just blockMesh so far
    m.add_mesh_source(
        "snappy_hex_mesh_dict_config",
        title="snappyHexMeshDict",
        schema=_SNAPPY_SCHEMA,
        seed={"refinement": 2},
    )
    assert m.mesh_sources() == ["block_mesh_dict_config", "snappy_hex_mesh_dict_config"]
    assert set(m.dims["mesh"].schema["properties"]) == {
        "block_mesh_dict_config",
        "snappy_hex_mesh_dict_config",
    }
    for payload in m.to_dimensions()["mesh"].values():
        assert payload["snappy_hex_mesh_dict_config"] == {"refinement": 2}
        assert "block_mesh_dict_config" in payload

    with pytest.raises(ValueError, match="already a mesh source"):
        m.add_mesh_source(
            "block_mesh_dict_config",
            title="blockMeshDict",
            schema=_BLOCK_SCHEMA,
            seed={},
        )


def test_remove_mesh_source_drops_dimension_when_last() -> None:
    m = SweepModel()
    with pytest.raises(ValueError, match="not a mesh source"):
        m.remove_mesh_source("block_mesh_dict_config")

    m.add_mesh_source(
        "block_mesh_dict_config", title="b", schema=_BLOCK_SCHEMA, seed={"scale": 1.0}
    )
    m.add_mesh_source(
        "snappy_hex_mesh_dict_config", title="s", schema=_SNAPPY_SCHEMA, seed={}
    )
    # Removing one source keeps the dimension (its key drops from every variant).
    m.remove_mesh_source("block_mesh_dict_config")
    assert m.has("mesh")
    assert m.mesh_sources() == ["snappy_hex_mesh_dict_config"]
    assert "block_mesh_dict_config" not in m.to_dimensions()["mesh"]["base"]
    # Removing the last source removes the whole mesh dimension.
    m.remove_mesh_source("snappy_hex_mesh_dict_config")
    assert not m.has("mesh")


def test_mesh_dimension_validates_config_name_keyed_payloads() -> None:
    m = SweepModel()
    m.add_mesh_source(
        "block_mesh_dict_config", title="b", schema=_BLOCK_SCHEMA, seed={"scale": 1.0}
    )
    ov = m.overview(_MESH_CLASSES, warn_threshold=64)
    assert ov.valid
    # A bad inner payload is caught (the mesh dim is not exempt like cad).
    m.variant_edit("mesh", {"block_mesh_dict_config": {"scale": "not-a-number"}})
    ov = m.overview(_MESH_CLASSES, warn_threshold=64)
    assert not ov.valid
    assert "mesh" in ov.errors


def test_variant_add_rename_delete_select() -> None:
    m = _model_with_transport()
    new = m.variant_add("transport")
    assert new == "variant-1"
    assert m.dims["transport"].selected == "variant-1"

    with pytest.raises(ValueError, match="Invalid variant name"):
        m.variant_rename("transport", "bad name!")
    m.variant_rename("transport", "nu2")
    assert set(m.dims["transport"].entries) == {"base", "nu2"}
    assert m.dims["transport"].selected == "nu2"

    m.variant_select("transport", "base")
    assert m.dims["transport"].selected == "base"

    m.variant_select("transport", "nu2")
    m.variant_delete("transport")
    assert set(m.dims["transport"].entries) == {"base"}
    with pytest.raises(ValueError, match="at least one variant"):
        m.variant_delete("transport")


def test_variant_edit_merges_only_when_sliced() -> None:
    sliced = _model_with_transport(fields=["nu"])
    sliced.variant_edit("transport", {"nu": 5e-5})
    # Merge — the unrendered transportModel survives.
    assert sliced.dims["transport"].entries["base"] == {
        "transportModel": "Newtonian",
        "nu": 5e-5,
    }

    whole = _model_with_transport()
    whole.variant_edit("transport", {"nu": 5e-5})
    # Replace — whole-form edit is authoritative.
    assert whole.dims["transport"].entries["base"] == {"nu": 5e-5}


def test_generate_series_names_and_replace() -> None:
    m = _model_with_transport(fields=["nu"])
    count = m.generate_series(
        "transport",
        "nu",
        series_values("list", "1e-5, 2e-5 4e-5", "", "", ""),
        replace=True,
    )
    assert count == 3
    assert set(m.dims["transport"].entries) == {"nu1e-05", "nu2e-05", "nu4e-05"}
    # The unrendered key from the template survives on every generated variant.
    assert m.dims["transport"].entries["nu2e-05"]["transportModel"] == "Newtonian"


def test_generate_series_integer_coercion_and_collision() -> None:
    # An integer target rounds each value to int; two values rounding to the
    # same integer collide, so the name gets a de-dup suffix.
    m = SweepModel()
    m.add_dimension(
        "mesh",
        title="mesh",
        schema={"type": "object", "properties": {"cells": {"type": "integer"}}},
        seed={"cells": 1},
    )
    count = m.generate_series("mesh", "cells", [2.6, 3.4], replace=True)
    assert count == 2
    entries = m.dims["mesh"].entries
    assert set(entries) == {"cells3", "cells3-2"}
    for payload in entries.values():
        assert isinstance(payload["cells"], int)
        assert payload["cells"] == 3


def test_generate_series_name_collision_dedup() -> None:
    # Two identical values collide on the auto name → the second is suffixed.
    m = _model_with_transport(fields=["nu"])
    count = m.generate_series("transport", "nu", [1.0, 1.0], replace=True)
    assert count == 2
    assert set(m.dims["transport"].entries) == {"nu1", "nu1-2"}


def test_generate_series_replace_false_appends() -> None:
    # Without replace, the pre-existing variants survive alongside the new ones.
    m = _model_with_transport(fields=["nu"])
    m.variant_add("transport")  # base + variant-1
    before = set(m.dims["transport"].entries)
    m.generate_series("transport", "nu", [2.0, 3.0], replace=False)
    entries = set(m.dims["transport"].entries)
    assert before <= entries
    assert {"nu2", "nu3"} <= entries


def test_series_values_modes_and_errors() -> None:
    assert series_values("list", "1 2 3", "", "", "") == [1.0, 2.0, 3.0]
    assert series_values("linear", "", "1", "3", "3") == [1.0, 2.0, 3.0]
    with pytest.raises(ValueError, match="positive"):
        series_values("log", "", "0", "10", "3")
    with pytest.raises(ValueError, match="at least 2"):
        series_values("linear", "", "1", "3", "1")
    with pytest.raises(ValueError, match="not a number"):
        series_values("list", "abc", "", "", "")


def test_series_values_log_and_remaining_errors() -> None:
    # A valid log range spans the endpoints multiplicatively.
    assert series_values("log", "", "1", "100", "3") == [1.0, 10.0, 100.0]
    # An empty value list, non-numeric bounds and an unknown mode each raise.
    with pytest.raises(ValueError, match="at least one value"):
        series_values("list", "", "", "", "")
    with pytest.raises(ValueError, match="must be numbers"):
        series_values("linear", "", "", "", "3")
    with pytest.raises(ValueError, match="unknown spacing mode"):
        series_values("bogus", "", "1", "3", "3")


def test_variant_name_strips_plus() -> None:
    assert variant_name("nu", 1e-5) == "nu1e-05"
    assert variant_name("n", 1e5) == "n100000"


def test_overview_count_label_and_table() -> None:
    m = _model_with_transport()
    m.variant_add("transport")
    m.variant_rename("transport", "nu2")
    m.variant_edit("transport", {"transportModel": "Newtonian", "nu": 2e-5})

    ov = m.overview(_CLASSES, warn_threshold=64)
    assert ov.case_count == 2
    assert ov.count_label == "2 case(s)"
    assert not ov.warn
    assert ov.valid
    assert [(h["title"], h["key"]) for h in ov.headers] == [
        ("case", "case"),
        ("transportProperties", "transport"),
        ("nu", "transport__nu"),
        ("validation", "validation"),
    ]
    assert all(r["validation"] == "✓" for r in ov.rows)


def test_overview_factorizes_and_warns() -> None:
    m = _model_with_transport()
    m.add_dimension("control", title="controlDict", schema=_SCHEMA, seed={"nu": 1.0})
    m.variant_add("transport")  # 2 transport variants × 1 control = 2
    ov = m.overview(_CLASSES, warn_threshold=1)
    assert "×" in ov.count_label
    assert ov.count_label.endswith("= 2 case(s)")
    assert ov.warn  # 2 > threshold 1


def test_overview_flags_invalid_variant() -> None:
    m = _model_with_transport()
    m.variant_edit("transport", {"transportModel": "Newtonian", "nu": "abc"})
    ov = m.overview(_CLASSES, warn_threshold=64)
    assert not ov.valid
    assert "transport" in ov.errors
    assert m.variant_error("transport", _CLASSES)  # for the Configure alert
    assert ov.rows[0]["validation"] != "✓"


def test_table_qualifies_cross_dimension_columns() -> None:
    # When two dimensions vary the SAME key, the column titles are qualified
    # with the dimension title so they stay distinguishable.
    m = _model_with_transport()  # dim "transport", title "transportProperties"
    m.variant_add("transport")
    m.variant_edit("transport", {"transportModel": "Newtonian", "nu": 2e-5})
    m.add_dimension(
        "control",
        title="controlDict",
        schema=_SCHEMA,
        seed={"transportModel": "Newtonian", "nu": 1.0},
    )
    m.variant_add("control")
    m.variant_edit("control", {"transportModel": "Newtonian", "nu": 2.0})

    ov = m.overview(_CLASSES, warn_threshold=64)
    nu_headers = [h for h in ov.headers if h["key"].endswith("__nu")]
    assert len(nu_headers) == 2
    assert {h["title"] for h in nu_headers} == {
        "transportProperties · nu",
        "controlDict · nu",
    }
    assert {h["key"] for h in nu_headers} == {"transport__nu", "control__nu"}


def test_mutations_mark_dirty() -> None:
    # Every model mutation flips the dirty flag (staleness vs. the last export).
    m = SweepModel()
    states = [
        DimensionState(
            name="transport",
            title="transportProperties",
            schema=_SCHEMA,
            entries={
                "base": {"transportModel": "Newtonian", "nu": 1e-5},
                "second": {"transportModel": "Newtonian", "nu": 2e-5},
            },
            selected="base",
        )
    ]
    m.load(states, enabled=["all", "setup", "solve", "setup_mesh"])
    assert not m.dirty

    m.variant_add("transport")
    assert m.dirty
    m.mark_exported()

    m.variant_rename("transport", "nuX")
    assert m.dirty
    m.mark_exported()

    m.variant_delete("transport")
    assert m.dirty
    m.mark_exported()

    m.variant_edit("transport", {"transportModel": "Newtonian", "nu": 3e-5})
    assert m.dirty
    m.mark_exported()

    m.generate_series("transport", "nu", [1.0], replace=False)
    assert m.dirty
    m.mark_exported()

    m.set_enabled(["all", "setup", "solve"])
    assert m.dirty


def _cad_model() -> SweepModel:
    m = SweepModel()
    m.add_cad_dimension(
        "cad",
        title="CAD geometry",
        model_path="geometry/design.FCStd",
        params={"tube_d": 6.0, "n_tubes": 8},
    )
    return m


def test_add_cad_dimension_builds_numeric_schema_and_entries() -> None:
    m = _cad_model()
    dim = m.dims["cad"]
    assert dim.kind == "cad"
    assert dim.model_path == "geometry/design.FCStd"
    assert dim.selected == "base"
    assert dim.entries == {"base": {"tube_d": 6.0, "n_tubes": 8}}
    assert set(dim.schema["properties"]) == {"tube_d", "n_tubes"}
    assert dim.schema["properties"]["tube_d"] == {"type": "number", "title": "tube_d"}
    assert set(dim.fields) == {"tube_d", "n_tubes"}
    assert m.dirty
    with pytest.raises(ValueError, match="already on the canvas"):
        m.add_cad_dimension("cad", title="CAD geometry", model_path="x", params={})


def test_cad_dimensions_export_shape() -> None:
    m = _cad_model()
    assert m.cad_dimensions() == {
        "cad": {
            "model": "geometry/design.FCStd",
            "variants": {"base": {"tube_d": 6.0, "n_tubes": 8}},
        }
    }
    # A config-only model exposes no cad axis.
    assert _model_with_transport().cad_dimensions() == {}


def test_overview_does_not_flag_cad_variant_and_counts_product() -> None:
    m = _cad_model()
    m.add_dimension("transport", title="t", schema=_SCHEMA, seed={"nu": 1e-5})
    m.variant_add("transport")  # 2 transport variants
    ov = m.overview(_CLASSES, warn_threshold=64)
    # cad(1) × transport(2) = 2 — cad flows through the count.
    assert ov.case_count == 2
    # CAD has no config class, yet the sweep is valid: it is validation-exempt.
    assert ov.valid
    assert "cad" not in ov.errors
    assert m.variant_error("cad", _CLASSES) == ""


def test_overview_cad_times_config_case_count() -> None:
    m = SweepModel()
    m.add_cad_dimension(
        "cad", title="CAD geometry", model_path="m", params={"tube_d": 6.0}
    )
    m.variant_add("cad")  # 2 cad variants
    m.add_dimension("transport", title="t", schema=_SCHEMA, seed={"nu": 1e-5})
    m.variant_add("transport")  # 2 transport variants
    ov = m.overview(_CLASSES, warn_threshold=64)
    assert ov.case_count == 4  # 2 cad × 2 transport
    assert "×" in ov.count_label
    assert ov.valid


def test_add_cad_dimension_with_empty_params() -> None:
    # A degenerate CAD axis with no driven aliases: an empty schema + a single
    # empty base variant. It still contributes one case to the count.
    m = SweepModel()
    m.add_cad_dimension("cad", title="CAD geometry", model_path="m", params={})
    dim = m.dims["cad"]
    assert dim.kind == "cad"
    assert dim.entries == {"base": {}}
    assert dim.schema["properties"] == {}
    assert dim.fields == []
    ov = m.overview(_CLASSES, warn_threshold=64)
    assert ov.case_count == 1
    assert ov.valid


def test_overview_falls_back_when_cross_product_collides() -> None:
    # Two config dims whose variant names collapse to one case name (a<b, so
    # (p, q_r) and (p_q, r) both give 'p_q_r'): cross_product raises and overview
    # falls back to an empty table instead of crashing the panel.
    m = SweepModel()
    m.add_dimension("a", title="a", schema=_SCHEMA, seed={})
    m.add_dimension("b", title="b", schema=_SCHEMA, seed={})
    m.dims["a"].entries = {"p": {}, "p_q": {}}
    m.dims["b"].entries = {"q_r": {}, "r": {}}
    ov = m.overview(_CLASSES, warn_threshold=64)
    assert ov.rows == []
    assert ov.case_count == 0


def test_load_replaces_and_clears_dirty() -> None:
    m = _model_with_transport()
    assert m.dirty
    states = [
        DimensionState(
            name="control",
            title="controlDict",
            schema=_SCHEMA,
            entries={"a": {"nu": 1.0}, "b": {"nu": 2.0}},
            selected="a",
        )
    ]
    m.load(states, enabled=["all", "setup", "solve", "setup_mesh"])
    assert set(m.dims) == {"control"}
    assert m.enabled == ["all", "setup", "solve", "setup_mesh"]
    assert not m.dirty  # a freshly loaded sweep matches disk
