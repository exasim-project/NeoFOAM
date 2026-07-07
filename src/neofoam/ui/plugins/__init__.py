# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The wizard's **public extension interface**: a small step-plugin registry.

The built-in steps (models / geometry / bcs / initial / schemes / sweep / review)
are hard-wired in :mod:`neofoam.ui.app`. This package is the stable contract a
*separate* package plugs into to contribute an ADDITIONAL step — e.g. installing
``foamcadagent`` adds a CAD-geometry step. Nothing CAD-specific lives in neofoam;
only the interface below.

A plugin is any object that structurally satisfies :class:`StepPlugin` (it is
duck-typed — a contributor need not import or subclass anything from neofoam, it
just exposes the attributes + two methods). It is discovered via the
``neofoam.ui.steps`` :data:`entry-point group <ENTRY_POINT_GROUP>` (the entry point
names a zero-arg factory returning the plugin), ordered by :attr:`StepPlugin.after`
(use :data:`AT_START` to place a step *first*, before ``models``), then for each:

* :meth:`~StepPlugin.register` runs once at build time — it seeds the plugin's own
  trame state and controllers on ``ctx.server``;
* :meth:`~StepPlugin.render` runs inside the wizard's content column for that step.

Both receive a :class:`StepContext`. Discovery is tolerant: a plugin whose factory
raises (an optional dependency absent) is logged and skipped, never fatal, so a
plain neofoam install simply shows no extra step.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from importlib import metadata
from typing import Any, Callable, Protocol, runtime_checkable

__all__ = [
    "StepContext",
    "StepPlugin",
    "ENTRY_POINT_GROUP",
    "AT_START",
    "discover_step_plugins",
]

_LOG = logging.getLogger(__name__)

#: Entry-point group a contributor registers a ``StepPlugin`` factory under, e.g.
#: ``[project.entry-points."neofoam.ui.steps"] foamcad = "pkg.mod:StepClass"``.
ENTRY_POINT_GROUP = "neofoam.ui.steps"

#: Sentinel :attr:`StepPlugin.after` value placing a step FIRST — before every
#: built-in step. Multiple ``AT_START`` plugins keep discovery order.
AT_START = "@start"


@dataclass(frozen=True)
class StepContext:
    """The handles a plugin gets in ``register`` / ``render`` — the interface surface.

    A contributor codes against these fields (all stable):

    * ``server`` — the trame server; ``server.state`` / ``server.controller`` are
      where the plugin seeds its OWN state vars and controllers (namespace them,
      e.g. ``cad_*``, to avoid clashing with the built-ins).
    * ``solver`` / ``solver_name`` — the resolved solver and its name.
    * ``entries`` — the wizard's :class:`~neofoam.ui.forms.FormEntry` list.
    * ``json_forms`` — the bundled ``<json-forms>`` trame component class, to
      render a JSON-Schema form.
    * ``v3`` / ``html`` / ``client`` — the trame ``vuetify3`` / ``html`` / ``client``
      widget modules, so ``render`` draws into the current content container.
    * ``sweep`` — the live :class:`~neofoam.ui.sweep_panel.SweepPanel`; a plugin
      contributes a sweep dimension through its public methods (e.g.
      ``sweep.add_cad_dimension(model_path, params)``).
    * ``schema_key`` — maps a ``FormEntry`` to its state-var name holding the schema.
    * ``extras`` — a mutable scratch dict for future/host-specific handles.
    """

    server: Any
    solver: Any
    solver_name: str
    entries: list[Any]
    json_forms: type
    v3: Any
    html: Any
    client: Any
    sweep: Any
    schema_key: Callable[[Any], str]
    extras: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class StepPlugin(Protocol):
    """A contributed wizard step. Concrete plugins are plain classes.

    ``after`` places the step: the id of the built-in (or earlier plugin) step it
    should follow, :data:`AT_START` to be FIRST (before ``models``), or ``None`` /
    an unknown id to append at the end. ``register`` runs once at build time (its
    own state + controllers); ``render`` runs inside the content column for the
    plugin's step. Weaving into the fixed built-in order is done by
    :func:`neofoam.ui.steps.build_steps`, which resolves ``after`` against the full
    step list (built-in ids included).
    """

    id: str
    label: str
    icon: str
    caption: str
    after: str | None

    def register(self, ctx: StepContext) -> None: ...

    def render(self, ctx: StepContext) -> None: ...


def discover_step_plugins(
    plugins: list[StepPlugin] | None = None,
) -> list[StepPlugin]:
    """Discover contributed step plugins (final ordering happens in ``build_steps``).

    With ``plugins`` given, they are returned unchanged (the test/embedding seam).
    Otherwise the ``neofoam.ui.steps`` entry points are loaded in discovery order;
    any whose ``load()`` or construction raises (a missing optional dependency,
    most often) is logged and skipped so one absent integration never breaks the
    wizard. Discovery order only breaks ties between plugins sharing an ``after``
    anchor — the authoritative weave into the built-in step order is
    :func:`neofoam.ui.steps.build_steps`, which resolves ``after`` against the full
    step list (built-in ids included).
    """
    if plugins is not None:
        return list(plugins)

    found: list[StepPlugin] = []
    for ep in metadata.entry_points(group=ENTRY_POINT_GROUP):
        try:
            factory = ep.load()
            found.append(factory())
        except Exception as exc:  # noqa: BLE001 - a bad/absent plugin must not be fatal
            _LOG.info("skipping ui step plugin %r: %s", ep.name, exc)
    return found
