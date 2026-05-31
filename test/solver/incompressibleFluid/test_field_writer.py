# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Spec for the fieldWriter Model — writing fields is its own concern.

Pure-Python: fakes stand in for the field hook / pybFoam Time, so no OpenFOAM
case is needed. The Model wraps the framework FieldWriter and selects a write
backend from the config.
"""

from __future__ import annotations

from typing import Any, Mapping, cast

import pybFoam
from pydantic import Field

from neofoam.algorithms.field_writer.write_control import StepperWriteControl
from neofoam.algorithms.field_writer.writer import FieldHook, FieldWriter, NullFieldHook
from neofoam.algorithms.solution_loop.loop_state import LoopState
from neofoam.framework.context import Context
from neofoam.solver.incompressibleFluid.configs import ControlDictConfig
from neofoam.solver.incompressibleFluid.models.field_writer import (
    PerFieldWriteHook,
    RuntimeWriteHook,
    build,
    fieldWriter,
    make_field_hook,
    write_output,
    writer_backend_steps,
)


def _config(**kw: object) -> ControlDictConfig:
    base: dict[str, object] = {"endTime": 0.3, "deltaT": 0.1}
    base.update(kw)
    return ControlDictConfig(**base)


def _state(*, write: bool) -> LoopState:
    """A LoopState (the StepView) with the write flag the decision reads."""
    return LoopState(value=0.0, delta_t=0.1, end_time=1.0, write_time=write)


class FakeFieldHook(FieldHook):
    calls: list[list[str]] = Field(default_factory=list)

    def write_fields(self, fields: Mapping[str, Any]) -> None:
        self.calls.append(sorted(fields))


class FakeRuntime:
    """Backend: registry write (write action) + per-step timing report."""

    def __init__(self) -> None:
        self.writes = 0
        self.printed = 0

    def write(self, force: bool = False) -> None:
        self.writes += 1

    def printExecutionTime(self) -> None:
        self.printed += 1


class FakeContext:
    def __init__(
        self,
        writer: FieldWriter,
        time: LoopState,
        runtime: FakeRuntime,
        fields: dict[str, Any],
        write_fields: set[str],
    ) -> None:
        # write_output reads ctx.time (the LoopState) for the decision and the
        # step_reporter seam (registered by writer_backend_steps) for timing.
        self.time = time
        self.models: dict[str, Any] = {
            "writer": writer,
            "step_reporter": runtime.printExecutionTime,
        }
        self.fields = fields
        self.write_fields = write_fields


# --- the Model ------------------------------------------------------------


def test_fieldwriter_is_a_model_owning_only_write() -> None:
    assert fieldWriter.name == "fieldWriter"
    assert fieldWriter._load_func is not None
    assert fieldWriter._build_func is not None
    op_names = {meta["name"] for _, meta in fieldWriter._operations}
    assert op_names == {"write_output"}


def test_build_emits_field_writer_with_null_hook() -> None:
    # the framework build is backend-agnostic: it emits the FieldWriter with a
    # no-op NullFieldHook; the pybFoam hook is injected by writer_backend_steps.
    steps = build(_config())
    assert len(steps) == 1
    assert steps[0].name == "models.writer"
    writer = steps[0].initializer({})
    assert isinstance(writer, FieldWriter)
    assert isinstance(writer.hook, NullFieldHook)


# --- backend selection from config ----------------------------------------


def test_default_backend_is_runtime_registry() -> None:
    hook = make_field_hook(_config(), FakeRuntime())
    assert isinstance(hook, RuntimeWriteHook)


def test_per_field_backend_selected_by_config() -> None:
    hook = make_field_hook(_config(writeBackend="perField"), FakeRuntime())
    assert isinstance(hook, PerFieldWriteHook)


# --- backend wiring into the framework model ------------------------------


def test_writer_backend_steps_inject_hook_and_reporter() -> None:
    # the framework writer starts with a NullFieldHook; the backend step swaps in
    # the config-selected pybFoam hook and registers the step_reporter.
    writer = cast(FieldWriter, build(_config())[0].initializer({}))
    assert isinstance(writer.hook, NullFieldHook)

    rt = FakeRuntime()
    steps = writer_backend_steps(_config())
    by_name = {s.name: s for s in steps}
    assert set(by_name) == {"models.writer_hook", "models.step_reporter"}

    ctx = {"models.writer": writer, "_foam_time": rt}
    by_name["models.writer_hook"].initializer(ctx)
    assert isinstance(writer.hook, RuntimeWriteHook)  # injected, no longer null

    reporter = by_name["models.step_reporter"].initializer(ctx)
    reporter()
    assert rt.printed == 1


# --- the two backends -----------------------------------------------------


def test_runtime_hook_writes_whole_registry() -> None:
    rt = FakeRuntime()
    # registry write captures everything registered (incl. turbulence k/epsilon),
    # ignoring the explicit fields handed in
    RuntimeWriteHook(registry=rt).write_fields({"U": object()})
    assert rt.writes == 1


def test_per_field_hook_writes_each_given_field(monkeypatch: Any) -> None:
    written: list[str] = []
    monkeypatch.setattr(pybFoam, "write", lambda f: written.append(f))
    PerFieldWriteHook().write_fields({"U": "u-field", "p": "p-field"})
    assert sorted(written) == ["p-field", "u-field"]


# --- the operation: hands over only the flagged fields --------------------


def _ctx(*, write: bool) -> tuple[FakeFieldHook, FakeRuntime, Context]:
    hook = FakeFieldHook()
    rt = FakeRuntime()
    writer = FieldWriter(write_control=StepperWriteControl(), hook=hook)
    # ctx has U, p, phi, UEqn but only U/p/phi are flagged for writing
    fields = {"U": object(), "p": object(), "phi": object(), "UEqn": object()}
    ctx = cast(
        Context,
        FakeContext(writer, _state(write=write), rt, fields, {"U", "p", "phi"}),
    )
    return hook, rt, ctx


def test_write_output_writes_only_flagged_fields() -> None:
    hook, rt, ctx = _ctx(write=True)
    write_output(None, ctx)
    assert hook.calls == [["U", "p", "phi"]]  # UEqn not flagged -> not handed over
    assert rt.printed == 1  # execution time reported every step


def test_write_output_skips_non_write_step() -> None:
    hook, rt, ctx = _ctx(write=False)
    write_output(None, ctx)
    assert hook.calls == []  # not a write step
    assert rt.printed == 1  # still reports timing
