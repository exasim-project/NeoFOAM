# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Solver-core configs for incompressibleVoF: ``configs.py``.

Round-trips the OpenFOAM-decorated configs through the real ``DictFile`` /
``pybFoam`` IO path (never in-memory field access alone), using a real
interFoam ``damBreak`` case (``constant/transportProperties``, ``constant/g``,
``constant/turbulenceProperties``, ``system/controlDict`` copied verbatim from
``$FOAM_TUTORIALS/multiphase/interFoam/laminar/damBreak/damBreak``) under
``cases/damBreak/``. The pure ``phases``/``dimensions``/``value`` parse ↔
serialize logic is additionally exercised by direct model construction
(pydantic unit-test style, matching ``test_boussinesq.py``), since only one of
the two directions is observable by reading a single on-disk file.
"""

import shutil
from pathlib import Path

import pytest

pytest.importorskip("pybFoam")  # the OpenFOAM write/read path needs pybFoam

from neofoam.solver.incompressibleVoF.configs import (  # noqa: E402
    ControlDictConfig,
    GravityConfig,
    PhaseTransport,
    TransportPropertiesConfig,
    TurbulencePropertiesConfig,
    _fmt_component,
)

_CASES = Path(__file__).parent / "cases"


@pytest.fixture
def damBreak(tmp_path: Path) -> Path:
    """A private copy of the real damBreak case files (never mutate checked-in)."""
    shutil.copytree(_CASES / "damBreak", tmp_path, dirs_exist_ok=True)
    return tmp_path


def test_transport_properties_loads_two_phase_defaults_from_real_case(
    damBreak: Path,
) -> None:
    """The interFoam damBreak ``transportProperties`` is water/air, as modelled."""
    cfg = TransportPropertiesConfig.load(case_dir=damBreak)
    assert cfg.phases == ["water", "air"]
    assert cfg.water == PhaseTransport(nu=1e-6, rho=1000.0)
    assert cfg.air == PhaseTransport(nu=1.48e-5, rho=1.0)
    assert cfg.sigma == 0.07


def test_transport_properties_save_round_trips_through_the_real_file(
    damBreak: Path,
) -> None:
    """Loaded config, saved back and reloaded, reproduces the same values."""
    cfg = TransportPropertiesConfig.load(case_dir=damBreak)
    cfg.save(case_dir=damBreak)
    reloaded = TransportPropertiesConfig.load(case_dir=damBreak)
    assert reloaded == cfg


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("(water air)", ["water", "air"]),
        ("(oil water gas)", ["oil", "water", "gas"]),
        ("  (water air)  ", ["water", "air"]),
        (["water", "air"], ["water", "air"]),
    ],
)
def test_phases_parses_both_openfoam_string_and_list_forms(
    raw: object, expected: list[str]
) -> None:
    """``_parse_phases`` accepts the on-disk ``(a b)`` token and a plain list alike."""
    cfg = TransportPropertiesConfig(phases=raw)  # type: ignore[arg-type]
    assert cfg.phases == expected


def test_phases_serializes_list_back_to_the_openfoam_paren_string() -> None:
    """``_serialize_phases`` is the inverse of ``_parse_phases``."""
    cfg = TransportPropertiesConfig(phases=["oil", "water"])
    assert cfg.model_dump()["phases"] == "(oil water)"


def test_gravity_config_loads_earth_gravity_from_real_case(damBreak: Path) -> None:
    """The damBreak ``constant/g`` is Earth gravity acting in -y."""
    cfg = GravityConfig.load(case_dir=damBreak)
    assert cfg.dimensions == [0, 1, -2, 0, 0, 0, 0]
    assert cfg.value == [0.0, -9.81, 0.0]


def test_gravity_config_save_round_trips_through_the_real_file(damBreak: Path) -> None:
    """Loaded config, saved back and reloaded, reproduces the same values."""
    cfg = GravityConfig.load(case_dir=damBreak)
    cfg.save(case_dir=damBreak)
    reloaded = GravityConfig.load(case_dir=damBreak)
    assert reloaded == cfg


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("[0 1 -2 0 0 0 0]", [0, 1, -2, 0, 0, 0, 0]),
        ("[1 0 0 0 0 0 0]", [1, 0, 0, 0, 0, 0, 0]),
        ([0, 1, -2, 0, 0, 0, 0], [0, 1, -2, 0, 0, 0, 0]),
    ],
)
def test_dimensions_parses_both_openfoam_string_and_list_forms(
    raw: object, expected: list[int]
) -> None:
    """``_parse_dimensions`` accepts the on-disk ``[..]`` token and a plain list."""
    cfg = GravityConfig(dimensions=raw)  # type: ignore[arg-type]
    assert cfg.dimensions == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("(0 -9.81 0)", [0.0, -9.81, 0.0]),
        ("(0 0 -9.81)", [0.0, 0.0, -9.81]),
        ([0.0, 0.0, -9.81], [0.0, 0.0, -9.81]),
    ],
)
def test_value_parses_both_openfoam_string_and_list_forms(
    raw: object, expected: list[float]
) -> None:
    """``_parse_value`` accepts the on-disk ``(..)`` token and a plain list."""
    cfg = GravityConfig(value=raw)  # type: ignore[arg-type]
    assert cfg.value == expected


@pytest.mark.parametrize(
    ("component", "expected"),
    [
        (0.0, "0"),
        (-9.81, "-9.81"),
        (3.0, "3"),
        (2.5, "2.5"),
        (-1.0, "-1"),
    ],
)
def test_fmt_component_drops_gratuitous_trailing_zero(component: float, expected: str) -> None:
    assert _fmt_component(component) == expected


def test_value_serializes_list_back_to_the_openfoam_paren_string() -> None:
    """``_serialize_value`` is the inverse of ``_parse_value``, using ``_fmt_component``."""
    cfg = GravityConfig(value=[0.0, -9.81, 2.5])
    assert cfg.model_dump()["value"] == "(0 -9.81 2.5)"


def test_control_dict_config_loads_interfoam_adaptive_stepping_from_real_case(
    damBreak: Path,
) -> None:
    """Adds interFoam's adaptive-stepping keys over ``TimeControlConfig``."""
    cfg = ControlDictConfig.load(case_dir=damBreak)
    assert cfg.application == "interFoam"
    assert cfg.adjustTimeStep is True
    assert cfg.maxCo == 1.0
    assert cfg.maxAlphaCo == 1.0
    assert cfg.maxDeltaT == 1.0
    # Inherited from TimeControlConfig / WriteControlConfig.
    assert cfg.endTime == 1.0
    assert cfg.deltaT == 0.001
    assert cfg.startTime == 0.0


def test_turbulence_properties_config_loads_laminar_from_real_case(
    damBreak: Path,
) -> None:
    """damBreak is laminar: no RAS/LES sub-dictionary."""
    cfg = TurbulencePropertiesConfig.load(case_dir=damBreak)
    assert cfg.simulationType == "laminar"
    assert cfg.RAS is None
    assert cfg.LES is None
