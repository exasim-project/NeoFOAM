# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Tests for the aggregate case model (``neofoam.io.case_spec``).

Two layers are exercised:

1. ``build_case_output_model`` — purely in-process: the aggregate ``CaseSpec``
   carries one ``Optional`` field per config class the solver may consume.
2. ``load_case_from_disk`` + ``save_case`` — the deterministic roundtrip through
   the OpenFOAM IO strategy, on checked-in cases (read-only).
"""

from __future__ import annotations

from pathlib import Path

from neofoam.io.case_spec import (
    build_case_output_model,
    case_spec_to_configs,
    load_case_from_disk,
    save_case,
)
from neofoam.solver.incompressibleFluid.configs import ControlDictConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_CASE = REPO_ROOT / "test" / "solver" / "incompressibleFluid" / "val_pitzDaily"
K_EPSILON = REPO_ROOT / "test" / "turbulence" / "setup_kEpsilon"
MOVING_ROW = REPO_ROOT / "test" / "solver" / "incompressibleFluid" / "cases" / "movingRow4"


# ---------------------------------------------------------------------------
# Pure-Python: schema shape
# ---------------------------------------------------------------------------


def test_case_output_model_has_one_optional_field_per_config() -> None:
    """The aggregate exposes every solver config as an Optional field."""
    model = build_case_output_model()
    # The five solver-core dictionaries pitzDaily needs are in there.
    expected = {
        "control_dict_config",
        "pimple_fv_schemes",
        "pimple_fv_solution",
        "transport_properties_config",
        "turbulence_properties_config",
    }
    assert expected.issubset(set(model.model_fields))
    # All fields default to None, so the agent can leave irrelevant ones out.
    instance = model()
    for name in model.model_fields:
        assert getattr(instance, name) is None


def test_case_spec_to_configs_filters_none_and_preserves_order() -> None:
    """Only populated BaseConfig fields are returned, in declaration order."""
    model = build_case_output_model()
    cd = ControlDictConfig(endTime=0.1, deltaT=0.01)
    spec = model(control_dict_config=cd)
    configs = case_spec_to_configs(spec)
    assert configs == [cd]


# ---------------------------------------------------------------------------
# Roundtrip via the OpenFOAM IO strategy
# ---------------------------------------------------------------------------


def test_load_case_from_disk_reads_every_present_config(tmp_path: Path) -> None:
    """``load_case_from_disk`` materialises the configs whose schemas match.

    ``ControlDictConfig`` / ``TransportPropertiesConfig`` /
    ``TurbulencePropertiesConfig`` round-trip cleanly through the
    OpenFOAM IO strategy. ``Pimple_fvSchemes`` / ``Pimple_fvSolution``
    fail validation against the bundled pitzDaily fixture (the schemas
    require fields the fixture doesn't carry — ``ddt(U)``, ``p_rgh``,
    etc.; see REVIEW.md for why the schemas are slices); they end up as
    ``None`` rather than raising, and ``fill_case``'s static-asset
    mirror keeps a runnable file on disk.
    """
    spec = load_case_from_disk(SOURCE_CASE)
    assert spec.control_dict_config is not None
    assert spec.transport_properties_config is not None
    assert spec.turbulence_properties_config is not None

    # Spot-check a few values to prove this is the *fixture's* data, not defaults.
    assert spec.transport_properties_config.transportModel == "Newtonian"
    assert spec.turbulence_properties_config.simulationType == "RAS"
    assert spec.turbulence_properties_config.RAS is not None
    assert spec.turbulence_properties_config.RAS.RASModel == "SpalartAllmaras"
    assert (spec.control_dict_config.endTime or 0.0) > 0.0


def test_load_case_from_disk_warns_about_a_partly_present_slice() -> None:
    """``"(U|nuTilda)"`` groups the ``U`` solver, while ``pFinal`` shows a PIMPLE case."""
    warnings: list[dict[str, str]] = []

    load_case_from_disk(SOURCE_CASE, warnings=warnings)

    assert warnings == [
        {
            "file": "system/fvSolution",
            "config": "Pimple_fvSolution",
            "reason": "missing solvers.U, solvers.UFinal",
        }
    ]


def test_load_case_from_disk_takes_a_regex_key_for_no_slice() -> None:
    """``"p.*"`` answers ``found("p_rgh")``, yet a case without buoyancy has no such slice."""
    warnings: list[dict[str, str]] = []

    load_case_from_disk(MOVING_ROW, warnings=warnings)

    assert "boussinesq_fvSolution" not in [warning["config"] for warning in warnings]


def test_load_case_from_disk_takes_default_none_for_no_boussinesq_slice() -> None:
    """``ddtSchemes { default none; }`` is no failed value of the slice's ``default`` key."""
    warnings: list[dict[str, str]] = []

    load_case_from_disk(K_EPSILON, warnings=warnings)

    assert "boussinesq_fvSchemes" not in [warning["config"] for warning in warnings]


def test_save_case_writes_files_via_registered_io_strategies(tmp_path: Path) -> None:
    """A CaseSpec roundtrips to disk and re-loads with the same values.

    Covers the three IO-strategy fixes:

    - ``TurbulencePropertiesConfig`` writes the nested ``RAS`` sub-dict
      from scratch (no longer raises "Cannot create nested dictionary
      'RAS' via pybFoam bindings").
    - The generated file carries a ``FoamFile`` block (previously
      stripped by the writer).
    - Every populated config ends up written — no warnings are emitted.
      Configs that co-own a file (``fvSolution`` + its ``PIMPLE`` block,
      ``transportProperties`` + ``Boussinesq``) are merged into a single
      write, so the invariant is per *target file*, not per config.
    """
    spec = load_case_from_disk(SOURCE_CASE)

    written = save_case(spec, tmp_path)
    populated = case_spec_to_configs(spec)
    assert {path.relative_to(tmp_path).as_posix() for path in written} == {
        config.file_name for config in populated
    }

    for path in written:
        assert "FoamFile" in path.read_text(), f"{path} missing FoamFile header"

    spec2 = load_case_from_disk(tmp_path)
    assert (
        spec2.transport_properties_config.transportModel
        == spec.transport_properties_config.transportModel
    )
    assert spec2.control_dict_config.endTime == spec.control_dict_config.endTime
    # The nested RAS sub-dict round-tripped fully:
    assert spec2.turbulence_properties_config.RAS is not None
    assert (
        spec2.turbulence_properties_config.RAS.RASModel
        == spec.turbulence_properties_config.RAS.RASModel
    )
