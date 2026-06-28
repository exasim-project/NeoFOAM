# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Enable-file schema + generic pipeline resolver/chainer for preprocessing tools.

``system/preprocess.yaml`` holds an ordered ``pipeline`` of entries, each naming a
registered tool via its ``tool`` key (dict-file presence on disk is irrelevant — only
the list activates a tool). :func:`resolve_pipeline` matches each entry against a
solver's registered tools; :func:`tool_init_steps` chains the resulting runtimes into a
linear, mesh-passing :class:`InitStep` line with a terminal ``mesh`` alias.
"""

from typing import Any, Callable, Dict, List, Optional

from neofoam.framework.initialization import InitStep, lazy
from neofoam.framework.initialization.execution import MESH_STATS_CATEGORY
from neofoam.io import YAML, BaseConfig, IOStrategy

from .spec import ToolRuntime, ToolSpec


@IOStrategy(YAML("system/preprocess.yaml"))
class PreprocessConfig(BaseConfig):
    """The ordered enable list for the preprocessing pipeline.

    ``pipeline`` is an open list of raw mappings, each naming a registered tool via
    its ``tool`` key. Resolution turns each entry into that tool's typed step config,
    so adding a new tool needs no change to this schema.
    """

    pipeline: List[Dict[str, Any]] = []


def resolve_pipeline(tools: List[ToolSpec], cfg: PreprocessConfig) -> List[ToolRuntime]:
    """Resolve each ``pipeline`` entry's ``tool:`` against ``tools``, in order.

    An entry naming a tool not in ``tools`` raises ``ValueError``. An empty/absent
    pipeline yields ``[]``.
    """
    by_name = {tool.name: tool for tool in tools}
    runtimes: List[ToolRuntime] = []
    for entry in cfg.pipeline:
        tool_name = entry.get("tool")
        spec = by_name.get(tool_name) if isinstance(tool_name, str) else None
        if spec is None:
            raise ValueError(f"Unknown preprocess tool: '{tool_name}'")
        runtimes.append(spec.instantiate(entry))
    return runtimes


def tool_init_steps(runtimes: List[ToolRuntime]) -> List[InitStep]:
    """Chain each tool's build step into a linear, mesh-passing pipeline.

    Identical to the v1 family chaining: steps run in pipeline order via a linear
    ``depends_on`` chain; the mesh is threaded through ``ctx["_prev_mesh"]`` from the
    last **mesh-producing** step (``mesh_key``); a stats-only step (category
    ``mesh_stats``, e.g. checkMesh) consumes the mesh but does not advance it. A
    terminal ``lazy("mesh", …, replaces=["mesh"])`` alias republishes that mesh and
    supersedes the default disk-read step. Returns ``[]`` for an empty pipeline.
    """
    steps: List[InitStep] = []
    prev_step: Optional[str] = None  # last step in the chain (ordering)
    mesh_key: Optional[str] = None  # ctx key holding the current mesh
    for rt in runtimes:
        built = rt.run_build()
        base = built[0]
        deps = [prev_step] if prev_step is not None else ["_foam_time"]
        orig = base.initializer
        mk = mesh_key

        def chained(
            ctx: dict[str, Any],
            _orig: Callable[[dict[str, Any]], Any] = orig,
            _mk: Optional[str] = mk,
        ) -> Any:
            if _mk is not None:
                ctx = {**ctx, "_prev_mesh": ctx[_mk]}
            return _orig(ctx)

        steps.append(
            InitStep(
                name=base.name,
                depends_on=deps,
                initializer=chained,
                category=base.category,
            )
        )
        prev_step = base.name
        if base.category != MESH_STATS_CATEGORY:
            mesh_key = base.name

    if mesh_key is not None:
        published = mesh_key
        last = prev_step
        assert last is not None  # a mesh producer implies the chain is non-empty

        def publish_mesh(ctx: dict[str, Any], _k: str = published) -> Any:
            return ctx[_k]

        steps.append(lazy("mesh", publish_mesh, depends_on=[last], replaces=["mesh"]))
    return steps
