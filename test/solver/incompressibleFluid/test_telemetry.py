# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Spec for the solver-owned telemetry opt-in: controlDict sub-dict + activation.

Every case here is ``cases/controldict_base`` — a bare ``system/controlDict``
with no ``telemetry`` entry — plus the sub-dict the scenario opts in with,
written through the dictionary writer by :func:`_case`. The opt-in *is* that
sub-dict, so a scenario is a mapping, not a committed copy of the whole file;
building into ``tmp_path`` also keeps the span files a run emits out of the
checked-in case.
"""

# NOTE: no `from __future__ import annotations` — keep annotations live.

import importlib.util
import sys
from pathlib import Path
from typing import Any, Iterator, Optional, cast

import pytest

from neofoam import telemetry as telemetry_shim
from neofoam.framework.solver import configurations
from neofoam.solver.incompressibleFluid.configs import TelemetryDictConfig
from neofoam.solver.incompressibleFluid.incompressibleFluid import (
    incompressibleFluid,
    maybe_configure_telemetry,
)
from neofoam.telemetry import TelemetryNotInstalledError
from neofoam.tooling.casebuild import from_template, patch

_BASE = Path(__file__).parent / "cases" / "controldict_base"

HAS_OTEL = importlib.util.find_spec("opentelemetry") is not None


def _case(tmp_path: Path, telemetry: Optional[dict] = None) -> Path:
    """``controldict_base`` in *tmp_path*, with a ``telemetry`` sub-dict if given."""
    pipeline = from_template(_BASE)
    if telemetry is not None:
        pipeline = pipeline | patch("system/controlDict", telemetry=telemetry)
    return pipeline.build_at(tmp_path / "case").path


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


def test_config_loads_from_case_file(tmp_path: Path) -> None:
    case = _case(tmp_path, {"enabled": True, "directory": "perf", "summary": True})
    config = TelemetryDictConfig.load(case_dir=case)
    assert config.enabled is True
    assert config.directory == "perf"


# --- activation helper ----------------------------------------------------------


@pytest.mark.parametrize(
    ("telemetry", "with_control_dict"),
    [
        (None, True),  # the dict is absent: telemetry is opt-in
        ({"enabled": False}, True),  # present but switched off
        (None, False),  # no controlDict at all
    ],
    ids=["dict_absent", "disabled", "no_control_dict"],
)
def test_maybe_configure_is_false_unless_the_dict_opts_in(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    telemetry: Optional[dict],
    with_control_dict: bool,
) -> None:
    monkeypatch.chdir(_case(tmp_path, telemetry) if with_control_dict else tmp_path)
    assert maybe_configure_telemetry() is False
    assert telemetry_shim.is_active() is False


@pytest.mark.skipif(not HAS_OTEL, reason="requires the neofoam[telemetry] extra")
@pytest.mark.parametrize(
    ("telemetry", "span_dir"),
    [
        # `directory perf;` from the controlDict is honoured …
        ({"enabled": True, "directory": "perf", "summary": True}, "perf"),
        # … and the dict alone opts in: `enabled` may be absent (defaults to
        # true), as may `directory` (defaults to `telemetry`).
        ({}, "telemetry"),
    ],
    ids=["configured_directory", "empty_dict_defaults"],
)
def test_maybe_configure_activates_and_writes_to_the_configured_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, telemetry: dict, span_dir: str
) -> None:
    case = _case(tmp_path, telemetry)
    monkeypatch.chdir(case)

    assert maybe_configure_telemetry() is True
    assert telemetry_shim.is_active() is True

    with telemetry_shim.span("probe"):
        pass
    telemetry_shim.shutdown()

    assert (case / span_dir / "rank0.spans.jsonl").is_file()


def test_maybe_configure_without_extra_raises_informative_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(_case(tmp_path, {"enabled": True}))

    for name in list(sys.modules):
        if name == "opentelemetry" or name.startswith("opentelemetry."):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "opentelemetry", None)  # type: ignore[arg-type]
    monkeypatch.delitem(sys.modules, "neofoam.telemetry._sdk", raising=False)

    with pytest.raises(TelemetryNotInstalledError, match=r"neofoam\[telemetry\]"):
        maybe_configure_telemetry()
    assert telemetry_shim.is_active() is False
