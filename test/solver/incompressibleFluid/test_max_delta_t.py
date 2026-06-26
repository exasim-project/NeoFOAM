# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Spec for the maxDeltaT optional model and its timeStepConstraint contribution."""

# NOTE: no `from __future__ import annotations` — keep annotations live.

from pathlib import Path
from typing import Any, Iterator, cast

import pytest

# Importing the model pulls in neofoam.io, whose OpenFOAM strategy imports pybFoam.
pytest.importorskip("pybFoam")

from pydantic import ValidationError

from neofoam.algorithms.solution_loop.interfaces import VGREAT, timeStepConstraint
from neofoam.framework.context import Context
from neofoam.solver.incompressibleFluid.models.incompressibleFluidModel import (
    incompressibleFluidModel,
)
from neofoam.solver.incompressibleFluid.models.max_delta_t import (
    MaxDeltaTConfig,
    max_delta_t_limit,
    maxDeltaT,
)

_CASES = Path(__file__).parent / "cases"


@pytest.fixture(autouse=True)
def _restore_constraint_registry() -> Iterator[None]:
    """Snapshot/restore the process-global timeStepConstraint registry so tests
    that activate/register contributions cannot leak across test order."""
    saved_active = set(timeStepConstraint._active_contributions)
    saved_contribs = list(timeStepConstraint._contributions)
    try:
        yield
    finally:
        timeStepConstraint._active_contributions = set(saved_active)
        timeStepConstraint._contributions = list(saved_contribs)


def test_model_is_registered_in_the_family_catalog() -> None:
    names = {spec.name for spec in incompressibleFluidModel.all_specs()}
    assert "maxDeltaT" in names


def test_registering_the_model_twice_keeps_one_catalog_entry() -> None:
    maxDeltaT.register_with(incompressibleFluidModel)
    names = [spec.name for spec in incompressibleFluidModel.all_specs()]
    assert names.count("maxDeltaT") == 1


def test_model_owns_the_control_dict_config() -> None:
    assert maxDeltaT._config_class is MaxDeltaTConfig
    assert cast(Any, MaxDeltaTConfig).io_config.file == "system/controlDict"


def test_model_inactive_without_the_config_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)  # no system/controlDict present here
    assert maxDeltaT.run_detect() is False


def test_model_inactive_when_control_dict_lacks_the_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # a controlDict that exists but carries no maxDeltaT entry: detect must be
    # False via found()==False, not via the file-absent exception path.
    monkeypatch.chdir(_CASES / "maxDeltaT_absent")
    assert maxDeltaT.run_detect() is False


def test_model_active_when_the_config_entry_is_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(_CASES / "maxDeltaT_present")
    assert maxDeltaT.run_detect() is True


def test_contribution_rejects_a_context_typed_parameter() -> None:
    with pytest.raises(ValueError, match="Context"):

        @timeStepConstraint.contribute
        def bad(ctx: Context) -> float:
            return 1.0


@pytest.mark.parametrize("bad", [0.0, -0.1])
def test_config_rejects_non_positive_cap(bad: float) -> None:
    with pytest.raises(ValidationError):
        MaxDeltaTConfig(maxDeltaT=bad)


def test_active_contribution_caps_delta_t_from_config() -> None:
    ctx = Context(fields={"cfg": MaxDeltaTConfig(maxDeltaT=0.5)}, models={})
    timeStepConstraint.activate(max_delta_t_limit)
    assert timeStepConstraint.collect(ctx) == pytest.approx(0.5)


def test_cap_wins_the_min_fold_against_a_larger_limit() -> None:
    # the cap must win the min fold, not merely return its own value.
    def larger_limit(cfg: MaxDeltaTConfig) -> float:
        return 9.0

    timeStepConstraint.contribute(larger_limit)
    timeStepConstraint.activate(max_delta_t_limit)
    ctx = Context(fields={"cfg": MaxDeltaTConfig(maxDeltaT=0.5)}, models={})
    assert timeStepConstraint.collect(ctx) == pytest.approx(0.5)


def test_deactivated_contribution_is_excluded_from_the_fold() -> None:
    # default state is deactivated, so the fold sees no active contribution
    ctx = Context(fields={"cfg": MaxDeltaTConfig(maxDeltaT=0.5)}, models={})
    assert timeStepConstraint.collect(ctx) == VGREAT


def test_collect_raises_when_a_declared_param_is_unavailable() -> None:
    ctx = Context(fields={}, models={})
    timeStepConstraint.activate(max_delta_t_limit)
    with pytest.raises(ValueError) as exc:
        timeStepConstraint.collect(ctx)
    message = str(exc.value)
    assert "timeStepConstraint" in message
    assert "max_delta_t_limit" in message
    assert "cfg" in message
