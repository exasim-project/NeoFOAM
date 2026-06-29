# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Enable-file schema + generic graph resolver/DAG-chainer for preprocessing tools.

``system/preprocess.yaml`` holds an open ``tools`` list of entries, each naming a
registered tool via its ``tool`` key plus an optional ``depends_on: list[str]`` (names
of OTHER listed tools it runs after) — dict-file presence on disk is irrelevant, only
the list activates a tool, and **order comes from the DAG, not list position**.
:func:`resolve_tools` matches each entry against a given set of registered tools (and
carries its ``depends_on``); :func:`tool_graph_steps` wires each runtime into the
existing init-DAG via its ``depends_on``, threading the mesh and publishing the sink
through a terminal ``mesh`` alias.
"""

from typing import Any, Callable, Dict, List, Optional

from neofoam.framework.initialization import InitStep, lazy
from neofoam.io import YAML, BaseConfig, IOStrategy

from .spec import ToolRuntime, ToolSpec


@IOStrategy(YAML("system/preprocess.yaml"))
class PreprocessConfig(BaseConfig):
    """The enable list for the preprocessing pipeline.

    ``tools`` is an open list of raw mappings; each names a registered tool via its
    ``tool`` key, may carry ``depends_on: list[str]`` (names of OTHER listed tools it
    runs after — order comes from the DAG, not list position), plus tool options.
    Resolution turns each entry into that tool's typed step config, so adding a new
    tool needs no change to this schema. Absent/empty file → no preprocessing.
    """

    tools: List[Dict[str, Any]] = []


def resolve_tools(tools: List[ToolSpec], cfg: PreprocessConfig) -> List[ToolRuntime]:
    """Resolve each ``tools`` entry's ``tool:`` against ``tools`` (the registered set).

    An entry naming a tool not in ``tools`` raises ``ValueError``. ``depends_on`` is
    carried onto each runtime; list position is NOT used for ordering. Empty → ``[]``.
    """
    by_name = {tool.name: tool for tool in tools}
    runtimes: List[ToolRuntime] = []
    for entry in cfg.tools:
        tool_name = entry.get("tool")
        spec = by_name.get(tool_name) if isinstance(tool_name, str) else None
        if spec is None:
            raise ValueError(f"Unknown preprocess tool: '{tool_name}'")
        runtimes.append(spec.instantiate(entry))
    return runtimes


def tool_graph_steps(runtimes: List[ToolRuntime]) -> List[InitStep]:
    """Wire each tool's build step into the init DAG from its ``depends_on``.

    Each runtime becomes an ``InitStep`` whose ``depends_on`` is ``_foam_time`` plus
    ``preprocess.<dep>`` for every declared dependency — the EXISTING topological sort
    orders them (file position is irrelevant). A tool with exactly one declared
    dependency receives ``ctx["_prev_mesh"]`` = that dependency's output mesh. The
    terminal ``lazy("mesh", …, replaces=["mesh"])`` alias publishes the pipeline SINK
    (the tool no other tool depends on) and supersedes the default disk-read step.
    A declared dependency naming a tool not in the file (or a cycle) is left for the
    existing DAG validation to reject with ``InitializationGraphError``. Returns ``[]``
    for an empty pipeline.
    """
    if not runtimes:
        return []

    depended: set[str] = set()
    for rt in runtimes:
        depended.update(rt.depends_on)

    steps: List[InitStep] = []
    for rt in runtimes:
        built = rt.run_build()
        if not built:
            raise ValueError(f"Tool '{rt.name}' produced no init steps from its @build")
        base = built[0]
        deps = ["_foam_time", *(f"preprocess.{d}" for d in rt.depends_on)]
        orig = base.initializer
        # Mesh threading is single-predecessor by design: a tool with >1 declared
        # dependency has no unambiguous _prev_mesh to thread, so reject it here with
        # a clear message rather than letting its @build hit a bare KeyError later.
        if len(rt.depends_on) > 1:
            dep_names = ", ".join(rt.depends_on)
            raise ValueError(
                f"preprocess tool '{rt.spec.name}' declares "
                f"{len(rt.depends_on)} dependencies [{dep_names}]; mesh threading "
                f"is single-predecessor by design — declare at most one depends_on"
            )
        # Exactly one declared dependency → thread its output mesh as _prev_mesh.
        mk = f"preprocess.{rt.depends_on[0]}" if len(rt.depends_on) == 1 else None

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

    # The sink is the tool no other entry depends on; it owns the published mesh.
    sinks = [rt for rt in runtimes if rt.spec.name not in depended]
    if len(sinks) == 1:
        published = sinks[0].name

        def publish_mesh(ctx: dict[str, Any], _k: str = published) -> Any:
            return ctx[_k]

        steps.append(
            lazy("mesh", publish_mesh, depends_on=[published], replaces=["mesh"])
        )
    elif len(sinks) > 1:
        names = ", ".join(rt.spec.name for rt in sinks)
        raise ValueError(
            f"preprocess pipeline must have exactly one sink tool; found "
            f"{len(sinks)}: [{names}]"
        )
    # len(sinks) == 0 → every tool is depended on → a cycle. Emit no alias and let the
    # existing DAG validation raise InitializationGraphError when the graph is sorted.
    return steps
