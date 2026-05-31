# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""When to persist fields — a plugin family of write policies.

Deciding *when* to write is neither the loop's job nor the solver's. The
:class:`WriteControl` interface is a :class:`~neofoam.core.plugin_system.PluginSystem`
family: each concrete policy registers itself (discriminated by
``write_control_type``) and maps 1:1 to OpenFOAM's ``writeControl`` keyword, so a
plugin adds a new policy (e.g. ``clockTime``) by registering a new
``@WriteControl.register`` class — never by editing here. Every policy reads only
a narrow :class:`StepView` of the stepper.

The *data* that selects/parameterises a policy is a :class:`WriteControlConfig` —
a validated, file-backed :class:`~neofoam.io.BaseConfig` (the controlDict write
keys). The config is the file interface; the policy is the behaviour.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol, cast, runtime_checkable

from pydantic import Field, PrivateAttr

from neofoam.core.plugin_system import PluginSystem
from neofoam.io import OF, BaseConfig, IOStrategy

_EPS = 1e-10


@IOStrategy(OF("system/controlDict"))
class WriteControlConfig(BaseConfig):
    """The controlDict write keys, as a validated, file-backed config.

    Plugins extend the schema by subclassing (e.g. a ``clockTime`` interval),
    keeping the file interface the single source of truth.
    """

    writeControl: str = "timeStep"
    writeInterval: float = Field(default=1.0, gt=0)
    startTime: float = 0.0


@runtime_checkable
class StepView(Protocol):
    """The narrow slice of loop state a write policy reads to decide a write step.

    The :class:`~neofoam.algorithms.solution_loop.loop_state.LoopState` dataclass
    satisfies it structurally (so does any object exposing these attributes).
    """

    value: float  # current time
    index: int  # current step index
    write_time: bool  # the loop's own (Python-computed) write flag


@PluginSystem.register(
    discriminator_variable="policy", discriminator="write_control_type"
)
class WriteControl(BaseConfig):
    """Plugin interface: decide whether the current step is a write step.

    Concrete policies register with :meth:`WriteControl.register`; the
    ``write_control_type`` literal discriminates them.
    """

    def should_write(self, stepper: StepView) -> bool:
        raise NotImplementedError


@WriteControl.register
class StepperWriteControl(WriteControl):
    """Use the stepper's own (Python-computed) ``outputTime`` flag."""

    write_control_type: Literal["stepper"] = "stepper"

    def should_write(self, stepper: StepView) -> bool:
        return stepper.write_time


@WriteControl.register
class IntervalWriteControl(WriteControl):
    """``writeControl timeStep`` — write every ``interval`` steps."""

    write_control_type: Literal["timeStep"] = "timeStep"
    interval: int = 1

    def should_write(self, stepper: StepView) -> bool:
        return stepper.index % max(1, self.interval) == 0


@WriteControl.register
class RunTimeWriteControl(WriteControl):
    """``writeControl runTime``/``adjustable`` — write every ``interval`` of sim time."""

    write_control_type: Literal["runTime"] = "runTime"
    interval: float = Field(gt=0)
    start: float = 0.0
    _last: float = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        self._last = self.start

    def should_write(self, stepper: StepView) -> bool:
        if stepper.value - self._last >= self.interval - _EPS:
            self._last = stepper.value
            return True
        return False


def _policy_payload(config: WriteControlConfig) -> dict[str, Any]:
    """Map the controlDict write keys onto a policy's discriminated payload.

    The only keyword-specific glue: it normalises the ``writeControl`` keyword to
    a registered ``write_control_type`` discriminator and renames the controlDict
    keys to the policy's fields. Everything past this point is the discriminated
    union picking the class — registering a new policy is enough to select it.
    """
    if config.writeControl == "timeStep":
        return {
            "write_control_type": "timeStep",
            "interval": int(config.writeInterval),
        }
    # runTime / adjustable / adjustableRunTime all map to the runTime policy
    return {
        "write_control_type": "runTime",
        "interval": float(config.writeInterval),
        "start": config.startTime,
    }


def write_control_from_config(
    config: WriteControlConfig,
    *,
    stepper_decides: bool = False,
) -> WriteControl:
    """Build the write policy from a validated :class:`WriteControlConfig`.

    Selection resolves through the :class:`WriteControl` discriminated union (the
    ``PluginSystem`` extension point): the payload's ``write_control_type`` picks
    the registered policy. Pass ``stepper_decides=True`` to use the stepper's own
    ``outputTime`` flag (it already computes the OpenFOAM write semantics in
    Python).
    """
    if stepper_decides:
        return StepperWriteControl()
    selected = cast(Any, WriteControl).create(policy=_policy_payload(config))
    return cast(WriteControl, selected.policy)
