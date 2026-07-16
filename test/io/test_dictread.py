# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

from pathlib import Path

import pytest

from neofoam.io.dictread import (
    Unreadable,
    Value,
    _leaf,
    read_entry,
    read_keys,
    read_section,
    read_toplevel,
)

CASES = Path(__file__).parent / "cases"


def test_leaf_returns_value_for_a_readable_render() -> None:
    assert _leaf(lambda: "fixedValue") == Value(text="fixedValue")


def _raise() -> str:
    raise RuntimeError("uniform (0 0 0) is not a single string")


def test_leaf_marks_unreadable_when_render_raises() -> None:
    # A leaf that cannot be rendered becomes an explicit Unreadable — never a
    # swallowed skip that truncates its section.
    leaf = _leaf(_raise)
    assert isinstance(leaf, Unreadable)
    assert "not a single string" in leaf.reason


def test_read_section_reads_all_patches_and_types_leaves() -> None:
    pytest.importorskip("pybFoam")
    section = read_section(CASES / "field_U", "boundaryField")
    assert set(section) == {"frontBack", "inlet", "walls"}
    assert section["frontBack"]["type"] == Value(text="symmetry")
    assert section["inlet"]["type"] == Value(text="fixedValue")
    assert section["walls"]["type"] == Value(text="noSlip")


def test_read_section_does_not_truncate_on_a_vector_value_leaf() -> None:
    # 'inlet' carries a vector `value uniform (1 0 0)` leaf; it (and every later
    # patch) must survive — the swallow that hid a broken boundaryField is gone.
    pytest.importorskip("pybFoam")
    section = read_section(CASES / "field_U", "boundaryField")
    assert "value" in section["inlet"]
    assert "walls" in section  # a patch after the vector-valued one is not lost


def test_read_section_missing_file_is_empty() -> None:
    assert read_section(CASES / "does_not_exist", "boundaryField") == {}


def test_read_section_missing_section_is_empty() -> None:
    pytest.importorskip("pybFoam")
    assert read_section(CASES / "fvSchemes", "boundaryField") == {}


def test_read_section_skips_scalar_top_level_entries() -> None:
    # divSchemes holds only scalar leaves (default none; div(phi,U) Gauss linear;) —
    # a section with no dict sub-entries yields {} (the non-dict top-entry branch).
    pytest.importorskip("pybFoam")
    assert read_section(CASES / "fvSchemes", "divSchemes") == {}


def test_read_section_skips_nested_dict_leaves() -> None:
    # A sub-entry may itself contain a nested block; that leaf is skipped, not read as
    # text and not Unreadable, while its scalar siblings survive.
    pytest.importorskip("pybFoam")
    section = read_section(CASES / "nested_boundary", "boundaryField")
    assert "nested" not in section["inlet"]
    assert section["inlet"]["type"] == Value(text="fixedValue")
    assert "value" in section["inlet"]
    assert set(section) == {"inlet", "outlet"}


def test_read_entry_reads_a_scheme_value() -> None:
    pytest.importorskip("pybFoam")
    leaf = read_entry(CASES / "fvSchemes", "divSchemes", "div(phi,U)")
    assert leaf == Value(text="Gauss linear")


def test_read_entry_absent_vs_unreadable() -> None:
    # Absence is None (not Unreadable): missing file, missing section, missing key.
    pytest.importorskip("pybFoam")
    assert read_entry(CASES / "does_not_exist", "divSchemes", "div(phi,U)") is None
    assert read_entry(CASES / "fvSchemes", "noSuchSection", "x") is None
    assert read_entry(CASES / "fvSchemes", "divSchemes", "div(phi,k)") is None


def test_read_keys_lists_top_level_names() -> None:
    pytest.importorskip("pybFoam")
    keys = read_keys(CASES / "transportProperties")
    assert keys is not None and not isinstance(keys, Unreadable)
    assert {"transportModel", "nu", "beta", "TRef"} <= keys


def test_read_keys_missing_file_is_none() -> None:
    # Absence is None (a genuine absence), never Unreadable.
    assert read_keys(CASES / "does_not_exist") is None


def test_read_keys_does_not_render_dimensioned_values() -> None:
    # Presence-only: a dimensionedScalar (beta/TRef) is reported by name without a
    # value render, so a valid buoyant dict is never wrongly escalated to Unreadable.
    pytest.importorskip("pybFoam")
    keys = read_keys(CASES / "transportProperties")
    assert not isinstance(keys, Unreadable)


def test_read_toplevel_reads_a_word_entry() -> None:
    pytest.importorskip("pybFoam")
    assert read_toplevel(CASES / "turbulenceProperties", "simulationType") == Value(
        text="laminar"
    )


def test_read_toplevel_absent_key_and_file_are_none() -> None:
    pytest.importorskip("pybFoam")
    assert read_toplevel(CASES / "turbulenceProperties", "noSuchKey") is None
    assert read_toplevel(CASES / "does_not_exist", "simulationType") is None
