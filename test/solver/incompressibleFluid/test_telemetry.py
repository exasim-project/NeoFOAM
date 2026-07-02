# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Spec for the telemetry optional model: controlDict opt-in + activation."""

# NOTE: no `from __future__ import annotations` — keep annotations live.

import importlib.util
import sys
from pathlib import Path
from typing import Any, Iterator, cast

import pytest

pytest.importorskip("pybFoam")

from neofoam import telemetry as telemetry_shim  # noqa: E402
from neofoam.solver.incompressibleFluid.models.incompressibleFluidModel import (  # noqa: E402
    incompressibleFluidModel,
)
from neofoam.solver.incompressibleFluid.models.telemetry import (  # noqa: E402
    TelemetryDictConfig,
    maybe_configure_telemetry,
    telemetry,
)
from neofoam.telemetry import TelemetryNotInstalledError  # noqa: E402

_CASES = Path(__file__).parent / "cases"

HAS_OTEL = importlib.util.find_spec("opentelemetry") is not None


@pytest.fixture(autouse=True)
def reset_telemetry() -> Iterator[None]:
    telemetry_shim.shutdown()
    yield
    telemetry_shim.shutdown()


# --- model registration + config ownership -----------------------------------


def test_model_is_registered_in_the_family_catalog() -> None:
    names = {spec.name for spec in incompressibleFluidModel.all_specs()}
    assert "telemetry" in names


def test_model_owns_the_control_dict_subdict_config() -> None:
    assert telemetry._config_class is TelemetryDictConfig
    io_config = cast(Any, TelemetryDictConfig).io_config
    assert io_config.file == "system/controlDict"
    assert io_config.subdict == "telemetry"


def test_config_defaults() -> None:
    config = TelemetryDictConfig()
    assert config.enabled is True
    assert config.directory == "telemetry"
    assert config.summary is True


def test_config_loads_from_case_file() -> None:
    config = TelemetryDictConfig.load(case_dir=_CASES / "telemetry_enabled")
    assert config.enabled is True
    assert config.directory == "perf"


# --- detection ----------------------------------------------------------------


def test_detect_active_when_dict_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(_CASES / "telemetry_enabled")
    detected = {rt.name for rt in incompressibleFluidModel.detect_models(Path("."))}
    assert "telemetry" in detected


def test_detect_active_when_enabled_key_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(_CASES / "telemetry_defaults")
    assert telemetry.run_detect() is True


def test_detect_inactive_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(_CASES / "telemetry_disabled")
    assert telemetry.run_detect() is False


def test_detect_inactive_when_dict_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(_CASES / "maxDeltaT_absent")
    assert telemetry.run_detect() is False


def test_detect_inactive_without_control_dict(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    assert telemetry.run_detect() is False


# --- activation helper ---------------------------------------------------------


def _case_copy(source: str, tmp_path: Path) -> Path:
    """Copy a fixture case so span files never pollute the checked-in cases."""
    import shutil

    target = tmp_path / source
    shutil.copytree(_CASES / source, target)
    return target


def test_maybe_configure_is_false_when_dict_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(_CASES / "maxDeltaT_absent")
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
