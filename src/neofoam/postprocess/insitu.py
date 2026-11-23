# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

# NB: no ``from __future__ import annotations`` — Context injection matches
# ``param.annotation is Context`` literally (see field_writer.py).

"""The InSitu plugin family: hand the live solver state to an in-situ tool.

A case declares in-situ entries next to its tables (``insitu:`` in
``system/postProcess.yaml``); the ``postProcess`` model then steps an
``insitu`` operation after ``post_process`` that calls each entry's
:meth:`InSitu.execute` on that entry's write steps. NeoFOAM ships no entry
type: a renderer (Catalyst, say) registers one from its own package::

    @InSitu.register
    class Catalyst(InSitu):
        type: Literal["catalyst"] = "catalyst"

        def execute(self, ctx: Context) -> None: ...

Decomposed, every rank evaluates every entry at the same steps, so an
``execute`` may be collective.
"""

from typing import NamedTuple

from pydantic import ConfigDict

from neofoam.algorithms.field_writer.write_control import WriteControl
from neofoam.core.plugin_system import PluginSystem
from neofoam.framework.context import Context
from neofoam.io import BaseConfig


@PluginSystem.register(discriminator_variable="insitu", discriminator="type")
class InSitu(BaseConfig):
    """Plugin interface: one in-situ tool, called with the live Context.

    An entry is a spec until the run's first step: :class:`InSituRunner` runs a
    private copy, so the state an implementation keeps between calls (a
    session, an output counter) belongs to the run. Unknown keys are refused.

    Subclass it to drive an external tool from a case; for a table of reduced
    values use a :class:`~neofoam.postprocess.table.TableSet` table instead. Example::

        @InSitu.register
        class Snapshot(InSitu):
            type: Literal["snapshot"] = "snapshot"

            def execute(self, ctx: Context) -> None:
                render(ctx.fields["p"])
    """

    model_config = ConfigDict(extra="forbid")

    # A default only because the family's wrapper model inherits the base's fields;
    # a declared entry always has its name (InSituSpec requires it).
    name: str = ""

    def execute(self, ctx: Context) -> None:
        """Hand ``ctx`` to the tool: called on this entry's due steps, on every rank.

        Every rank calls it at the same steps, so it may be collective.
        """
        raise NotImplementedError

    def finalize(self) -> None:
        """Release what :meth:`execute` opened; called once, after the time loop."""


class InSituEntry(NamedTuple):
    """A declared entry: the plugin and when it runs.

    ``tables_for_case`` builds these from ``insitu:``; build one directly only to
    drive an :class:`InSituRunner` without a case file. It is a spec, shared by
    runs, so the runner copies it. Example::

        InSituEntry(plugin, write_control)
    """

    plugin: InSitu
    write_control: WriteControl


class InSituRunner:
    """The runtime state of the ``insitu`` operation: a private copy of every entry.

    The ``postProcess`` model builds one per run when a case declares entries, so a
    solver never constructs it. Its lifecycle is :meth:`execute` once per step, then
    :meth:`finalize` once; both run on every rank. Example::

        runner = InSituRunner(tables.insitu)
        runner.execute(ctx)
        runner.finalize()
    """

    def __init__(self, entries: list[InSituEntry]) -> None:
        self.entries = [
            InSituEntry(
                entry.plugin.model_copy(deep=True), entry.write_control.model_copy(deep=True)
            )
            for entry in entries
        ]

    def execute(self, ctx: Context) -> None:
        """Call every entry due at this step."""
        for plugin, write_control in self.entries:
            if write_control.should_write(ctx.time):
                plugin.execute(ctx)

    def finalize(self) -> None:
        """Finalize every entry (collective, like :meth:`execute`).

        A :class:`~neofoam.framework.solver.runtime.Finalizer`: the solver calls it
        after the time loop (``SolverRuntime.finalize``).
        """
        for plugin, _ in self.entries:
            plugin.finalize()
