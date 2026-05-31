# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

# NB: no ``from __future__ import annotations`` — see solution_loop.py.

"""fieldWriter — persisting fields as a framework *core* Model.

Writing is a separate concern from advancing the stepper, so it is a separate
core Model from ``solutionLoop`` (SRP). It is backend-agnostic and wraps a
:class:`~neofoam.algorithms.field_writer.writer.FieldWriter`:

* **SRP** — the writer decides *whether* this is a write step and, if so, writes
  the fields; it does not advance time.
* **OCP** — *when* to write is a
  :class:`~neofoam.algorithms.field_writer.write_control.WriteControl` policy selected through
  the discriminated union from the validated
  :class:`~neofoam.algorithms.field_writer.write_control.WriteControlConfig` (a ``BaseConfig``
  loaded straight from ``system/controlDict``).
* **DIP** — the write *action* goes through a
  :class:`~neofoam.algorithms.field_writer.writer.FieldHook`. The default is
  :class:`~neofoam.algorithms.field_writer.writer.NullFieldHook` (persists nothing); a solver
  injects its backend hook post-build (registry write, per-field write, …).

The optional ``ctx.models["step_reporter"]`` callable (``() -> None``) lets a
backend report per-step timing after a write; absent ⇒ no-op.
"""

from pathlib import Path
from typing import Any

from neofoam.algorithms.field_writer.write_control import (
    WriteControlConfig,
    write_control_from_config,
)
from neofoam.algorithms.field_writer.writer import FieldWriter, NullFieldHook
from neofoam.framework.context import Context
from neofoam.framework.initialization import InitStep, model
from neofoam.framework.model import Model

fieldWriter = Model("fieldWriter")


# -- load: the controlDict write keys parameterise the policy -------------
@fieldWriter.load
def load(case_dir: Path, instance_id: str) -> WriteControlConfig:
    return WriteControlConfig.load(case_dir=case_dir)


# -- build: the FieldWriter is this model's runtime state -----------------
@fieldWriter.build
def build(config: WriteControlConfig) -> list[InitStep]:
    def create_writer(ctx: dict[str, Any]) -> FieldWriter:
        # The write policy is selected through the WriteControl discriminated
        # union; stepper_decides=True uses the stepper's Python-computed
        # outputTime flag. NullFieldHook by default (persists nothing) — a solver
        # injects its backend hook afterwards by assigning ``writer.hook``.
        return FieldWriter(
            write_control=write_control_from_config(config, stepper_decides=True),
            hook=NullFieldHook(),
        )

    return [model("writer", create_writer)]


# -- operation: write on write steps --------------------------------------
# Stepped last in the solver's time loop, so builder-insertion order already
# places it after every field-mutating op (pressure-velocity, viscosity and
# turbulence correction); no explicit ``depends_on`` is needed and a generic
# writer must not name a specific solver's correction op.
@fieldWriter.operation()
def write_output(self: Any, ctx: Context) -> None:
    """Write the flagged fields on write steps; optionally report step timing.

    The decision reads the loop state (``ctx.time``, with the Python-computed
    ``write_time`` flag); the action writes through the injected backend hook.
    Only the ``write=True``-flagged fields are handed over (a registry backend
    ignores them and writes everything anyway).
    """
    writer: FieldWriter = ctx.models["writer"]
    to_write = {
        name: ctx.fields[name] for name in ctx.write_fields if name in ctx.fields
    }
    writer.write(ctx.time, to_write)
    reporter = ctx.models.get("step_reporter")
    if reporter is not None:
        reporter()
