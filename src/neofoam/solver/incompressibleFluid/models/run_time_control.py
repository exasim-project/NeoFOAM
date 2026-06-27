# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

# NOTE: no `from __future__ import annotations` — keep annotations live so the
# contribution's `U` / config params resolve by name (mirrors courant.py).

"""runTimeControl — dict-defined loop-stop conditions as a loopCondition contribution.

Solver-side (pybFoam): a ``RunTimeCondition`` PluginSystem registry (mirroring
``DeltaTConstraint``) with the ``minMax`` velocity/divergence guard and the
``equationInitialResidual`` convergence stop, plus a ``runTimeControl`` model whose
contribution folds the active conditions into a single ``ConditionVote``. It lives under
the solver, not the framework, so ``neofoam.algorithms.solution_loop`` stays pure-Python
(the only condition module importing pybFoam).

Gating is intrinsic: the contribution participates iff the ``runTimeControl`` model is
active for the case (matched to the owning ``loopCondition`` interface by ``ModelSpec``
identity), exactly like ``courant`` on ``timeStepConstraint``.
"""

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Annotated, Any, Literal, Mapping

from pybFoam import dictionary, mag, volVectorField
from pydantic import Field, model_validator

from neofoam.algorithms.solution_loop.conditions import (
    Action,
    ConditionVote,
    fold_conditions,
)
from neofoam.algorithms.solution_loop.interfaces import loopCondition
from neofoam.core.plugin_system import PluginSystem
from neofoam.framework.initialization import InitStep
from neofoam.framework.initialization import model as init_model
from neofoam.io import OF, BaseConfig, IOStrategy

from .incompressibleFluidModel import Model, incompressibleFluidModel
from .pressure_velocity.control_factory import read_residual_control

VGREAT = 1e300


@dataclass(frozen=True)
class ConditionContext:
    """The per-fold inputs a condition reads (NOT the framework Context)."""

    fields: Mapping[str, Any] = field(default_factory=dict)
    residuals: Mapping[str, float] = field(default_factory=dict)


@PluginSystem.register(discriminator_variable="condition", discriminator="type")
class RunTimeCondition(BaseConfig):
    """A dict-defined stop criterion (OpenFOAM ``runTimeControl`` condition)."""

    groupID: int = -1
    active: bool = True

    def vote(self, cctx: ConditionContext) -> ConditionVote:
        raise NotImplementedError


@RunTimeCondition.register
class MinMaxCondition(RunTimeCondition):
    """Satisfied when a field's magnitude extremum crosses ``value`` per ``mode``."""

    type: Literal["minMax"] = "minMax"
    mode: Literal["minimum", "maximum"]
    fields: list[str]
    value: float

    def vote(self, cctx: ConditionContext) -> ConditionVote:
        try:
            extrema = [_extremum(cctx.fields[name], self.mode) for name in self.fields]
        except KeyError as exc:
            missing = exc.args[0]
            raise ValueError(
                f"minMax condition: field {missing!r} is not available; "
                f"available fields: {sorted(cctx.fields)}"
            ) from exc
        overall = max(extrema) if self.mode == "maximum" else min(extrema)
        satisfied = (
            overall > self.value if self.mode == "maximum" else overall < self.value
        )
        return ConditionVote(satisfied=satisfied, group_id=self.groupID)


@RunTimeCondition.register
class EquationInitialResidualCondition(RunTimeCondition):
    """Satisfied when every listed field's initial residual is at or below ``value``."""

    type: Literal["equationInitialResidual"] = "equationInitialResidual"
    fields: list[str]
    value: float
    mode: Literal["minimum"] = "minimum"

    def vote(self, cctx: ConditionContext) -> ConditionVote:
        satisfied = all(
            cctx.residuals.get(name, VGREAT) <= self.value for name in self.fields
        )
        return ConditionVote(satisfied=satisfied, group_id=self.groupID)


def residual_conditions(
    residual_control: dict[str, float], group_id: int
) -> list[RunTimeCondition]:
    """Normalize an fvSolution ``residualControl`` into the registry's condition kind.

    One :class:`EquationInitialResidualCondition` per ``field -> tolerance`` (each
    over its single field, since tolerances differ per field), all sharing
    ``group_id`` so the whole ``residualControl`` is one AND group — the run/loop
    converges only when **every** field's published residual is at or below its
    tolerance. ``action`` stays the default ``"end"`` (a clean finish, never abort).
    """
    return [
        EquationInitialResidualCondition(
            fields=[name], value=tolerance, groupID=group_id
        )
        for name, tolerance in residual_control.items()
    ]


@IOStrategy(OF("system/controlDict"))
class RunTimeControlConfig(BaseConfig):
    """A ``functions.<name>`` runTimeControl block: its conditions + satisfiedAction."""

    conditions: dict[str, RunTimeCondition] = Field(default_factory=dict)
    satisfiedAction: Action = "end"

    @model_validator(mode="before")
    @classmethod
    def _coerce_conditions(cls, data: Any) -> Any:
        if isinstance(data, dict) and isinstance(data.get("conditions"), dict):
            coerced = {
                name: (
                    value
                    if isinstance(value, RunTimeCondition)
                    else RunTimeCondition.create(condition=value).condition  # type: ignore[attr-defined]
                )
                for name, value in data["conditions"].items()
            }
            data = {**data, "conditions": coerced}
        return data


def _extremum(field_obj: Any, mode: str) -> float:
    """The magnitude extremum of *field_obj* over its internal field (local reduction)."""
    magnitudes = _field_magnitudes(field_obj)
    return max(magnitudes) if mode == "maximum" else min(magnitudes)


def _field_magnitudes(field_obj: Any) -> list[float]:
    """The per-cell magnitudes of *field_obj* — the single pybFoam touch-point."""
    return list(mag(field_obj).ref().internalField())


runTimeControl = Model("runTimeControl").register_with(incompressibleFluidModel)
runTimeControl.config(RunTimeControlConfig)


@runTimeControl.detect
def detect_model() -> bool:
    """Active iff ``system/controlDict`` declares a ``functions`` entry of this type.

    Conservative: a case without a ``runTimeControl`` functionObject keeps its existing
    (fixed-step / endTime) stop behaviour, so the hotRoom/cavity parities are untouched.
    File-driven loading of the conditions themselves lands with the dict-loading work.
    """
    if not os.path.isfile("system/controlDict"):
        return False
    cd = dictionary.read("system/controlDict")
    if not cd.found("functions"):
        return False
    functions = cd.subDict("functions")
    for word in functions.toc():
        name = str(word)
        if functions.isDict(name) and functions.subDict(name).found("type"):
            if functions.subDict(name).get_word("type") == "runTimeControl":
                return True
    return False


@runTimeControl.contributes(loopCondition)
def evaluate_conditions(U: volVectorField, cfg: RunTimeControlConfig) -> ConditionVote:
    """Fold the active conditions into one stop verdict (gated to the active model)."""
    cctx = ConditionContext(fields={"U": U})
    votes = [
        replace(condition.vote(cctx), action=cfg.satisfiedAction)
        for condition in cfg.conditions.values()
        if condition.active
    ]
    return fold_conditions(votes)


class ResidualControlConfig(BaseConfig):
    """The fvSolution residualControl normalized into condition kinds."""

    conditions: list[RunTimeCondition] = Field(default_factory=list)


residualControl = Model("residualControl").register_with(incompressibleFluidModel)


def _read_fvsolution_residuals(case_dir: Path) -> dict[str, float]:
    """Merge the PIMPLE/SIMPLE residualControl of *case_dir*'s fvSolution."""
    fv = dictionary.read(str(case_dir / "system" / "fvSolution"))
    out: dict[str, float] = {}
    for algo in ("PIMPLE", "SIMPLE"):
        if fv.found(algo):
            out.update(read_residual_control(fv.subDict(algo)))
    return out


@residualControl.load
def load_residual_control(case_dir: Path, instance_id: str) -> ResidualControlConfig:
    """Init-time read: fvSolution residualControl -> one AND group of conditions."""
    residuals = _read_fvsolution_residuals(case_dir)
    return ResidualControlConfig(conditions=residual_conditions(residuals, group_id=0))


@residualControl.detect
def detect_residual_control() -> bool:
    """Active iff system/fvSolution has a PIMPLE/SIMPLE residualControl subdict."""
    if not os.path.isfile("system/fvSolution"):
        return False
    fv = dictionary.read("system/fvSolution")
    for algo in ("PIMPLE", "SIMPLE"):
        if fv.found(algo) and fv.subDict(algo).found("residualControl"):
            return True
    return False


@residualControl.build
def build_residual_control() -> list[InitStep]:
    """Publish the per-case live residual store the contribution reads.

    Empty until the solver publishes solve residuals (deferred — the member
    ``fvMatrix.solve()`` binding returns void). An empty store keeps every residual
    vote not-satisfied, so an activated residualControl is a no-op stop source until
    live publishing lands.
    """

    def make_residual_store(_ctx: dict[str, Any]) -> dict[str, float]:
        return {}

    return [init_model("residuals", make_residual_store)]


@residualControl.contributes(loopCondition)
def evaluate_residual_control(
    residuals: Annotated[Mapping[str, float], "models"],
    cfg: ResidualControlConfig,
) -> ConditionVote:
    """Fold the fvSolution residual conditions, reading the live residual store."""
    cctx = ConditionContext(residuals=residuals)
    return fold_conditions(c.vote(cctx) for c in cfg.conditions)
