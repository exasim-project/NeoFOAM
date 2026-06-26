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
    maxDeltaT,
)

_CASES = Path(__file__).parent / "cases"


@pytest.fixture(autouse=True)
def _restore_constraint_registry() -> Iterator[None]:
    """Snapshot/restore the process-global timeStepConstraint registry so tests
    that register extra contributions cannot leak across test order."""
    saved_contribs = list(timeStepConstraint._contributions)
    saved_owner = dict(timeStepConstraint._owner)
    try:
        yield
    finally:
        timeStepConstraint._contributions = list(saved_contribs)
        timeStepConstraint._owner = dict(saved_owner)


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


def test_contribution_rejects_a_context_typed_parameter() -> None:
    with pytest.raises(ValueError, match="Context"):

        @timeStepConstraint.contribute
        def bad(ctx: Context) -> float:
            return 1.0


@pytest.mark.parametrize("bad", [0.0, -0.1])
def test_config_rejects_non_positive_cap(bad: float) -> None:
    with pytest.raises(ValidationError):
        MaxDeltaTConfig(maxDeltaT=bad)


def test_contribution_caps_delta_t_when_model_active() -> None:
    ctx = Context(
        fields={"cfg": MaxDeltaTConfig(maxDeltaT=0.5)},
        models={"maxDeltaT": object()},
    )
    assert timeStepConstraint.collect(ctx) == pytest.approx(0.5)


def test_cap_wins_the_min_fold_against_a_larger_limit() -> None:
    # An unowned (always-active) larger limit plus the active maxDeltaT cap: the cap
    # must win the min fold, not merely return its own value.
    def larger_limit(cfg: MaxDeltaTConfig) -> float:
        return 9.0

    timeStepConstraint.contribute(larger_limit)
    ctx = Context(
        fields={"cfg": MaxDeltaTConfig(maxDeltaT=0.5)},
        models={"maxDeltaT": object()},
    )
    assert timeStepConstraint.collect(ctx) == pytest.approx(0.5)


def test_contribution_excluded_when_model_inactive() -> None:
    # maxDeltaT absent from ctx.models -> its contribution does not fold -> VGREAT.
    ctx = Context(fields={"cfg": MaxDeltaTConfig(maxDeltaT=0.5)}, models={})
    assert timeStepConstraint.collect(ctx) == VGREAT


def test_collect_raises_when_a_declared_param_is_unavailable() -> None:
    # Model active but its config field is missing from the Context -> explicit error.
    ctx = Context(fields={}, models={"maxDeltaT": object()})
    with pytest.raises(ValueError) as exc:
        timeStepConstraint.collect(ctx)
    message = str(exc.value)
    assert "timeStepConstraint" in message
    assert "max_delta_t_limit" in message
    assert "cfg" in message


def test_config_presence_activates_contribution_through_detection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A real case that declares maxDeltaT: detection makes the model active, its
    # name lands in ctx.models, and the contribution folds in collect().
    monkeypatch.chdir(_CASES / "maxDeltaT_present")
    detected = incompressibleFluidModel.detect_models(Path("."))
    models = {rt.name: rt for rt in detected}
    assert "maxDeltaT" in models

    cfg = models["maxDeltaT"].config
    ctx = Context(fields={"cfg": cfg}, models=models)
    assert timeStepConstraint.collect(ctx) == pytest.approx(0.5)


def test_absent_config_leaves_contribution_unfolded_through_detection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A real case without maxDeltaT: detection leaves the model inactive, so the
    # contribution is registered but never folds -> fixed step (VGREAT).
    monkeypatch.chdir(_CASES / "maxDeltaT_absent")
    detected = incompressibleFluidModel.detect_models(Path("."))
    models = {rt.name: rt for rt in detected}
    assert "maxDeltaT" not in models

    ctx = Context(fields={}, models=models)
    assert timeStepConstraint.collect(ctx) == VGREAT


def test_two_runs_in_one_process_flip_participation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Run 1: config present -> the cap folds.
    monkeypatch.chdir(_CASES / "maxDeltaT_present")
    m1 = {rt.name: rt for rt in incompressibleFluidModel.detect_models(Path("."))}
    ctx1 = Context(fields={"cfg": m1["maxDeltaT"].config}, models=m1)
    assert timeStepConstraint.collect(ctx1) == pytest.approx(0.5)

    # Run 2: same process, config removed -> the cap no longer folds, no re-import.
    monkeypatch.chdir(_CASES / "maxDeltaT_absent")
    m2 = {rt.name: rt for rt in incompressibleFluidModel.detect_models(Path("."))}
    ctx2 = Context(fields={}, models=m2)
    assert timeStepConstraint.collect(ctx2) == VGREAT


def test_model_inactive_when_no_control_dict_is_present(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A directory with no system/controlDict at all: detection degrades to
    # inactive via the isfile guard rather than raising.
    monkeypatch.chdir(tmp_path)
    assert maxDeltaT.run_detect() is False


def test_participation_reappears_when_config_returns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Reverse of the present->absent flip: absent first (fixed step), then the
    # config returns and the cap folds again — same process, no re-import.
    monkeypatch.chdir(_CASES / "maxDeltaT_absent")
    m1 = {rt.name: rt for rt in incompressibleFluidModel.detect_models(Path("."))}
    assert timeStepConstraint.collect(Context(fields={}, models=m1)) == VGREAT

    monkeypatch.chdir(_CASES / "maxDeltaT_present")
    m2 = {rt.name: rt for rt in incompressibleFluidModel.detect_models(Path("."))}
    ctx2 = Context(fields={"cfg": m2["maxDeltaT"].config}, models=m2)
    assert timeStepConstraint.collect(ctx2) == pytest.approx(0.5)
