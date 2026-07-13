# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Spec for the solver-owned telemetry opt-in: controlDict sub-dict + activation."""

# NOTE: no `from __future__ import annotations` — keep annotations live.

import importlib.util
import shutil
import sys
from pathlib import Path
from typing import Any, Iterator, cast

import pytest

from neofoam import telemetry as telemetry_shim
from neofoam.framework.solver import configurations
from neofoam.solver.incompressibleFluid.configs import TelemetryDictConfig
from neofoam.solver.incompressibleFluid.incompressibleFluid import (
    incompressibleFluid,
    maybe_configure_telemetry,
)
from neofoam.telemetry import TelemetryNotInstalledError

_CASES = Path(__file__).parent / "cases"

HAS_OTEL = importlib.util.find_spec("opentelemetry") is not None


@pytest.fixture(autouse=True)
def reset_telemetry() -> Iterator[None]:
    telemetry_shim.shutdown()
    yield
    telemetry_shim.shutdown()


# --- config ownership -----------------------------------------------------------


def test_solver_owns_the_control_dict_subdict_config() -> None:
    io_config = cast(Any, TelemetryDictConfig).io_config
    assert io_config.file == "system/controlDict"
    assert io_config.subdict == "telemetry"
    # part of the solver's case-free config schema (no model catalog entry)
    assert TelemetryDictConfig in set(configurations(incompressibleFluid))


def test_config_defaults() -> None:
    config = TelemetryDictConfig()
    assert config.enabled is True
    assert config.directory == "telemetry"
    assert config.summary is True


def test_config_loads_from_case_file() -> None:
    config = TelemetryDictConfig.load(case_dir=_CASES / "telemetry_enabled")
    assert config.enabled is True
    assert config.directory == "perf"


# --- activation helper ----------------------------------------------------------


def _case_copy(source: str, tmp_path: Path) -> Path:
    """Copy a fixture case so span files never pollute the checked-in cases."""
    target = tmp_path / source
    shutil.copytree(_CASES / source, target)
    return target


def test_maybe_configure_is_false_when_dict_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(_CASES / "controldict_base")
    assert maybe_configure_telemetry() is False
    assert telemetry_shim.is_active() is False


def test_maybe_configure_is_false_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(_CASES / "telemetry_disabled")
    assert maybe_configure_telemetry() is False
    assert telemetry_shim.is_active() is False


def test_maybe_configure_is_false_without_control_dict(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    assert maybe_configure_telemetry() is False
    assert telemetry_shim.is_active() is False


@pytest.mark.skipif(not HAS_OTEL, reason="requires the neofoam[telemetry] extra")
def test_maybe_configure_activates_with_configured_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    case = _case_copy("telemetry_enabled", tmp_path)
    monkeypatch.chdir(case)

    assert maybe_configure_telemetry() is True
    assert telemetry_shim.is_active() is True

    with telemetry_shim.span("probe"):
        pass
    telemetry_shim.shutdown()

    # `directory perf;` from the controlDict is honoured
    assert (case / "perf" / "rank0.spans.jsonl").is_file()


@pytest.mark.skipif(not HAS_OTEL, reason="requires the neofoam[telemetry] extra")
def test_maybe_configure_with_empty_dict_uses_defaults(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The dict alone opts in — ``enabled`` may be absent (defaults to true)."""
    case = _case_copy("telemetry_defaults", tmp_path)
    monkeypatch.chdir(case)

    assert maybe_configure_telemetry() is True
    with telemetry_shim.span("probe"):
        pass
    telemetry_shim.shutdown()

    assert (case / "telemetry" / "rank0.spans.jsonl").is_file()


def test_maybe_configure_without_extra_raises_informative_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    case = _case_copy("telemetry_enabled", tmp_path)
    monkeypatch.chdir(case)

    for name in list(sys.modules):
        if name == "opentelemetry" or name.startswith("opentelemetry."):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "opentelemetry", None)  # type: ignore[arg-type]
    monkeypatch.delitem(sys.modules, "neofoam.telemetry._sdk", raising=False)

    with pytest.raises(TelemetryNotInstalledError, match=r"neofoam\[telemetry\]"):
        maybe_configure_telemetry()
    assert telemetry_shim.is_active() is False
