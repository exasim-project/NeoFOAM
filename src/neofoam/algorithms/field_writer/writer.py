# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Per-step writers — persisting output, the symmetric twin of the SolutionLoop.

:class:`Writer` is the *abstraction*: anything that may persist something each
step (fields, probes, residual logs, …). It is a
:class:`~neofoam.core.plugin_system.PluginSystem` interface — new writers are new
``@Writer.register`` classes (discriminated by ``writer_type``), the extension
point.

:class:`FieldWriter` is the concrete writer for simulation fields. Like
:class:`~neofoam.algorithms.solution_loop.solution_loop.SolutionLoop` (advance + mirror to a
``LoopBackend``), it splits responsibilities:

* the **policy** — when to write — is a
  :class:`~neofoam.algorithms.field_writer.write_control.WriteControl` plugin (OCP), reading
  only a :class:`~neofoam.algorithms.field_writer.write_control.StepView` of the stepper;
* the **backend** — actually writing to disk — is a :class:`FieldHook` plugin, so
  the same FieldWriter serves multiple backends (registry write, per-field
  write, ``NullFieldHook`` for standalone).

The write *flag* is the policy decision; the per-field write to disk is the
hook's job (in OpenFOAM, ``pybFoam.write(field)`` per field).
"""

from __future__ import annotations

from typing import Any, Literal, Mapping

from pydantic import ConfigDict, Field

from neofoam.core.plugin_system import PluginSystem
from neofoam.algorithms.field_writer.write_control import StepView, WriteControl
from neofoam.io import BaseConfig


@PluginSystem.register(discriminator_variable="hook", discriminator="field_hook_type")
class FieldHook(BaseConfig):
    """Plugin interface: persist the given fields to disk (a swappable backend).

    Receives the fields to write (name -> field); the stepper has already synced
    the backend's time, so each field lands in the right time directory. Concrete
    backends register with :meth:`FieldHook.register`. ``arbitrary_types_allowed``
    lets a backend hold a live runtime handle.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def write_fields(self, fields: Mapping[str, Any]) -> None:
        raise NotImplementedError


@FieldHook.register
class NullFieldHook(FieldHook):
    """Default backend for standalone / pure-Python use: persists nothing."""

    field_hook_type: Literal["null"] = "null"

    def write_fields(self, fields: Mapping[str, Any]) -> None:
        return None


@PluginSystem.register(discriminator_variable="writer", discriminator="writer_type")
class Writer(BaseConfig):
    """Plugin interface: a per-step writer. Register one to add a new writer."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def write(self, stepper: StepView, fields: Mapping[str, Any]) -> bool:
        """Persist for the current step; return whether anything was written."""
        raise NotImplementedError


@Writer.register
class FieldWriter(Writer):
    """Writes simulation fields to disk on write steps (a :class:`Writer`).

    Usage (the ``fieldWriter`` Model drives this once per step)::

        writer.write(stepper, ctx.fields)   # writes via the hook iff a write step
    """

    writer_type: Literal["field"] = "field"
    write_control: WriteControl
    hook: FieldHook = Field(default_factory=NullFieldHook)

    def should_write(self, stepper: StepView) -> bool:
        """Is the current step a write step? (delegated to the policy)."""
        return self.write_control.should_write(stepper)

    def write(self, stepper: StepView, fields: Mapping[str, Any]) -> bool:
        """Persist ``fields`` if this is a write step; return whether it wrote."""
        if self.write_control.should_write(stepper):
            self.hook.write_fields(fields)
            return True
        return False
