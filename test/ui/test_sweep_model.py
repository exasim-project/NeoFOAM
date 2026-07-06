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


def test_series_values_modes_and_errors() -> None:
    assert series_values("list", "1 2 3", "", "", "") == [1.0, 2.0, 3.0]
    assert series_values("linear", "", "1", "3", "3") == [1.0, 2.0, 3.0]
    with pytest.raises(ValueError, match="positive"):
        series_values("log", "", "0", "10", "3")
    with pytest.raises(ValueError, match="at least 2"):
        series_values("linear", "", "1", "3", "1")
    with pytest.raises(ValueError, match="not a number"):
        series_values("list", "abc", "", "", "")


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
