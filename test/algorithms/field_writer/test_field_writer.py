# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Spec for the framework fieldWriter *core* Model (backend-agnostic).

The framework default hook persists nothing (NullFieldHook); a solver injects
its backend hook by assigning ``writer.hook``. Here a recording fake hook stands
in. No OpenFOAM.
"""

from __future__ import annotations

from typing import Any, Mapping, cast

from pydantic import Field

from neofoam.algorithms.field_writer.field_writer import (
    build,
    fieldWriter,
    write_output,
)
from neofoam.algorithms.field_writer.write_control import WriteControlConfig
from neofoam.algorithms.field_writer.writer import FieldHook, FieldWriter, NullFieldHook
from neofoam.algorithms.solution_loop.loop_state import LoopState
from neofoam.framework.context import Context


class FakeFieldHook(FieldHook):
    """Records the fields it is asked to persist (must subclass FieldHook so
    pydantic keeps the instance on the ``FieldWriter.hook`` field)."""

    field_hook_type: str = "fake"
    calls: list[Mapping[str, Any]] = Field(default_factory=list)

    def write_fields(self, fields: Mapping[str, Any]) -> None:
        self.calls.append(dict(fields))


def _state(*, write: bool) -> LoopState:
    """A LoopState (the StepView) whose write flag drives the decision."""
    return LoopState(value=0.0, delta_t=0.1, end_time=1.0, write_time=write)


def _ctx(*, models: dict[str, Any], fields: dict[str, Any], time: LoopState) -> Context:
    """A real Context (all flagged fields eligible to write) — write_output reads
    ``models``/``fields``/``write_fields``/``time`` off it, so build the genuine
    object rather than a duck-typed stand-in."""
    return Context(models=models, fields=fields, write_fields=set(fields), time=time)


# --- the Model ------------------------------------------------------------


def test_modelspec_is_a_full_core_model() -> None:
    assert fieldWriter.name == "fieldWriter"
    assert fieldWriter._load_func is not None
    op_names = {meta["name"] for _, meta in fieldWriter._operations}
    assert op_names == {"write_output"}


def test_build_emits_writer_with_null_hook_by_default() -> None:
    steps = build(WriteControlConfig(writeControl="timeStep", writeInterval=1))
    assert [s.name for s in steps] == ["models.writer"]
    writer = steps[0].initializer({})
    assert isinstance(writer, FieldWriter)
    assert isinstance(writer.hook, NullFieldHook)  # framework default: no-op


# --- write_output through the injected hook -------------------------------


def _writer_with_hook() -> tuple[FieldWriter, FakeFieldHook]:
    writer = cast(FieldWriter, build(WriteControlConfig())[0].initializer({}))
    hook = FakeFieldHook()
    writer.hook = hook  # solver injects its backend hook post-build
    return writer, hook


def test_write_output_persists_flagged_fields_on_a_write_step() -> None:
    writer, hook = _writer_with_hook()
    reported: list[bool] = []
    ctx = _ctx(
        models={
            "writer": writer,
            "step_reporter": lambda: reported.append(True),
        },
        fields={"U": object(), "p": object()},
        time=_state(write=True),
    )
    write_output(None, ctx)
    assert len(hook.calls) == 1
    assert set(hook.calls[0]) == {"U", "p"}
    assert reported == [True]  # step_reporter consulted


def test_write_output_skips_on_a_non_write_step() -> None:
    writer, hook = _writer_with_hook()
    ctx = _ctx(
        models={"writer": writer},
        fields={"U": object()},
        time=_state(write=False),
    )
    write_output(None, ctx)
    assert hook.calls == []  # nothing written; missing step_reporter is fine
