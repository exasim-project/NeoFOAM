# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""ToolSpec — a shared, per-solver-registered preprocessing capability.

A ``Tool`` is the slimmed parallel of :class:`~neofoam.framework.model.ModelSpec`:
it carries a single ``@tool.build`` that emits :class:`InitStep` objects to run
before the time loop, with its enable-file/step schema inferred from the
``@build`` parameter annotation (``step_config_type``). There is no
``@detect``/``@resolve``/``@operation`` — tools are enable-file activated and run only
at init time. Tools live in shared packages and self-register into the process-wide
registry (``neofoam.tools.registry``); the same instance is reusable across solvers.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from pydantic import BaseModel

from neofoam.framework.initialization import InitStep


class ToolSpec:
    """Immutable preprocessing-tool definition (read-only after import).

    ``consumes_mesh`` marks tools whose build reads ``ctx["_prev_mesh"]``
    (snappyHexMesh, checkMesh). When such a tool starts a pipeline (no
    ``depends_on``), the graph seeds ``_prev_mesh`` from a caller-provided
    mesh source (see :func:`~neofoam.framework.tools.graph.tool_graph_steps`).
    """

    def __init__(self, name: str, *, consumes_mesh: bool = False) -> None:
        self.name = name
        self.consumes_mesh = consumes_mesh
        self._build_func: Optional[Callable[[Any], list[InitStep]]] = None

    def build(
        self, func: Callable[[Any], list[InitStep]]
    ) -> Callable[[Any], list[InitStep]]:
        """Register the BUILD function: ``def build(cfg) -> list[InitStep]``."""
        self._build_func = func
        return func

    @property
    def step_config_type(self) -> Optional[type[BaseModel]]:
        """The ``@build`` first-parameter annotation when it is a ``BaseModel``.

        Returns ``None`` for an untyped/``Any`` build parameter — the open seam,
        whose pipeline entry is passed through as a raw mapping.
        """
        if self._build_func is None:
            return None
        params = list(inspect.signature(self._build_func).parameters.values())
        if not params:
            return None
        annotation = params[0].annotation
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            return annotation
        return None

    def instantiate(
        self, entry: dict[str, Any], *, instance_id: Optional[str] = None
    ) -> "ToolRuntime":
        """Resolve one pipeline entry to a runtime.

        A typed tool validates ``entry`` against its step-config (defaults
        applied; the ``depends_on`` envelope key is ignored by pydantic — it is
        not a tool option); the open seam receives the raw mapping. ``depends_on``
        (bare names of other listed tools this entry runs after) is carried onto
        the runtime. ``instance_id`` is accepted for ``ModelSpec`` parity but
        unused (one runtime per entry).
        """
        config_type = self.step_config_type
        config: Any = (
            config_type.model_validate(entry) if config_type is not None else entry
        )
        depends_on = entry.get("depends_on", [])
        return ToolRuntime(
            spec=self,
            name=f"preprocess.{self.name}",
            config=config,
            depends_on=list(depends_on),
        )


def Tool(name: str, *, consumes_mesh: bool = False) -> ToolSpec:
    """Factory: create a named :class:`ToolSpec`."""
    return ToolSpec(name, consumes_mesh=consumes_mesh)


@dataclass
class ToolRuntime:
    """One resolved pipeline entry — a ``ToolSpec`` + its validated step config."""

    spec: ToolSpec
    name: str  # "preprocess.<tool>"
    config: Any
    depends_on: list[str] = field(default_factory=list)  # bare tool names run-after

    def run_build(self) -> list[InitStep]:
        """Call the spec's build func with this entry's config."""
        if self.spec._build_func is None:
            return []
        return self.spec._build_func(self.config)
