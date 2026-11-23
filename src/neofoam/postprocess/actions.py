# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

# NB: no ``from __future__ import annotations`` — Context injection matches
# ``param.annotation is Context`` literally (see field_writer.py).

"""The Action plugin family: run something on the live simulation state.

Tables save reduced values to disk; an action runs on the live state at chosen
time steps, for example live visualization through Catalyst. A case declares
actions next to its tables (``actions:`` in ``system/postProcess.yaml``); the
``postProcess`` model then steps an ``actions`` operation after ``post_process``
that calls each action's :meth:`Action.execute` on its write steps. NeoFOAM
ships no action type: a renderer (Catalyst, say) registers one from its own
package::

    @Action.register
    class Catalyst(Action):
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


@PluginSystem.register(discriminator_variable="action", discriminator="type")
class Action(BaseConfig):
    """Plugin interface: one action on the live simulation state ("in situ" in HPC terms).

    An entry is a spec until the run's first step: :class:`ActionRunner` runs a
    private copy, so the state an implementation keeps between calls (a
    session, an output counter) belongs to the run. Unknown keys are refused.

    Subclass it to drive an external tool from a case; for a table of reduced
    values use a :class:`~neofoam.postprocess.table.TableSet` table instead. Example::

        @Action.register
        class Snapshot(Action):
            type: Literal["snapshot"] = "snapshot"

            def execute(self, ctx: Context) -> None:
                render(ctx.fields["p"])
    """

    model_config = ConfigDict(extra="forbid")

    # A default only because the family's wrapper model inherits the base's fields;
    # a declared entry always has its name (ActionSpec requires it).
    name: str = ""

    def execute(self, ctx: Context) -> None:
        """Hand ``ctx`` to the tool: called on this entry's due steps, on every rank.

        Every rank calls it at the same steps, so it may be collective.
        """
        raise NotImplementedError

    def finalize(self) -> None:
        """Release what :meth:`execute` opened; called once, after the time loop."""


class ActionEntry(NamedTuple):
    """A declared entry: the plugin and when it runs.

    ``tables_for_case`` builds these from ``actions:``; build one directly only to
    drive an :class:`ActionRunner` without a case file. It is a spec, shared by
    runs, so the runner copies it. Example::

        ActionEntry(plugin, write_control)
    """

    plugin: Action
    write_control: WriteControl


class ActionRunner:
    """The runtime state of the ``actions`` operation: a private copy of every entry.

    The ``postProcess`` model builds one per run when a case declares entries, so a
    solver never constructs it. Its lifecycle is :meth:`execute` once per step, then
    :meth:`finalize` once; both run on every rank. Example::

        runner = ActionRunner(tables.actions)
        runner.execute(ctx)
        runner.finalize()
    """

    def __init__(self, entries: list[ActionEntry]) -> None:
        self.entries = [
            ActionEntry(
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
