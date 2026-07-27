# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The predefined Snakemake rule library for NeoFOAM workflows.

The mesh sweep (``setup_mesh``/``blockMesh``/``snappyHexMesh``/``checkMesh``/
``setup``/``solve``/``post``) is reached the same way any packaged rule set
is — ``include:`` a path under :func:`rules_dir` — and is *selected*:
:class:`RuleRegistry` holds the specs and :meth:`RuleRegistry.plan` resolves an
enabled subset into a :class:`RulePlan` the codegen turns into a Snakefile.

Each :class:`RuleSpec` describes one packaged ``.smk`` file (shipped in this
directory) — its Snakemake rule name, the file patterns it consumes/produces
(``{case}`` = one sweep case, ``{mesh}`` = one mesh variant) and whether it is
*keyed* (runs once per mesh variant) or per-case. A generated Snakefile is a
thin header (build the :class:`~neofoam.tooling.workflow.paramspace.YamlParamSpace`,
materialize the configs, define the plan's globals) followed by ``include:``
lines referencing these files, so the rule bodies stay versioned together with
the ``python -m neofoam.tooling.workflow.sweep_runner`` CLI they shell out to.

The default mesh-sweep pipeline::

    setup_mesh ─ blockMesh ─ snappyHexMesh ─ checkMesh      (once per mesh variant)
                                        └─ setup ─ solve    (once per case)
                                                     └─ all

Each ``.smk`` file documents the header globals it consumes — defined by the
codegen's header (see :mod:`neofoam.tooling.workflow.sweep`). This module is
stdlib-only — generated Snakefiles import it at parse time.

Interface (``__all__`` — the rule model only):

* **Rule model** (the deep part — hides all Snakemake wiring): :class:`RuleKind`,
  :class:`RuleSpec`, :class:`RuleRegistry`, :class:`RulePlan`,
  :func:`default_registry`, :func:`rules_dir`.
* **Reserved dimension names**: :data:`MESH_DIM`.
* **Default selection**: :data:`DEFAULT_ENABLED`.

The ``{case}``/``{mesh}`` file-pattern constants (``MESH_CONFIG_PATTERN``,
``SETUP_STAMP_PATTERN``, ``RUN_DONE_PATTERN``, …) are defined below but are **not**
on the interface: they are shared internals the codegen and the packaged ``.smk``
bodies must agree on verbatim, not something a rule-graph consumer needs to learn.
They remain importable by name (``from …rules import RUN_DONE_PATTERN``) for the
codegen; a caller who only wants the rule graph uses the model above.
"""

from __future__ import annotations

import importlib.resources
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

__all__ = [
    "DEFAULT_ENABLED",
    "MESH_DIM",
    "RuleKind",
    "RulePlan",
    "RuleRegistry",
    "RuleSpec",
    "default_registry",
    "rules_dir",
]

#: The reserved keyed dimension: mesh variants (params.yaml key / sweep column).
MESH_DIM = "mesh"

# File patterns of the pipeline ({case} = sweep case, {mesh} = mesh variant).
MESH_CONFIG_PATTERN = "configs/mesh/{mesh}.json"
MESH_STAGE_STAMP = ".staged.json"  # meshes/{mesh}/.staged.json
SETUP_CONFIG_PATTERN = "configs/{case}/setup.json"
SETUP_STAMP_PATTERN = "cases/{case}/.applied.json"
RUN_DONE_PATTERN = "cases/{case}/done"
POST_DONE_PATTERN = "cases/{case}/.post.done"


def rules_dir() -> Path:
    """The installed directory holding the packaged ``.smk`` files."""
    return Path(str(importlib.resources.files("neofoam.tooling.workflow.rules")))


class RuleKind(Enum):
    """The wiring category of a rule — the single field that fixes how it is wired.

    One kind per rule replaces the earlier bag of mutually-constraining booleans
    (``creates_mesh``/``consumes_mesh_dim``/…), so an illegal combination (e.g.
    "creates a mesh AND stages the variant") is now unrepresentable and both
    :meth:`RuleRegistry.plan` and the canvas resolver branch on one value.

    Members:
        AGGREGATE: the ``all`` sink target (emitted inline, no ``.smk``).
        MESH_STAGE: stages a clean mesh mini-case per variant (``setup_mesh``).
        MESH_CREATE: creates the base mesh and starts the chain (``blockMesh``).
        MESH_TOOL: refines/validates an existing mesh (``snappyHexMesh``/``checkMesh``).
        CASE_SETUP: clones the base case + applies the case's configs (``setup``).
        CASE: a plain per-case rule (``solve``/``post``).
    """

    AGGREGATE = "aggregate"
    MESH_STAGE = "mesh_stage"
    MESH_CREATE = "mesh_create"
    MESH_TOOL = "mesh_tool"
    CASE_SETUP = "case_setup"
    CASE = "case"


#: Kinds that run once per mesh variant (the keyed rows).
_MESH_KEYED = frozenset({RuleKind.MESH_STAGE, RuleKind.MESH_CREATE, RuleKind.MESH_TOOL})
#: Keyed rules in the mesh *tool* chain — staging (``MESH_STAGE``) excluded.
_MESH_TOOLS = frozenset({RuleKind.MESH_CREATE, RuleKind.MESH_TOOL})


@dataclass(frozen=True)
class RuleSpec:
    """One predefined Snakemake rule backed by a packaged ``.smk`` file.

    Attributes:
        name: The Snakemake rule name (``blockMesh``).
        smk_file: The packaged file (``block_mesh.smk``).
        kind: The rule's wiring category (see :class:`RuleKind`) — the single
            source of truth the boolean views below derive from.
        inputs: Input file patterns (mesh-tool inputs are resolved per plan).
        outputs: Output file patterns.
        stamp_name: Override for the per-variant stamp basename (relative to
            ``meshes/{mesh}/``); defaults to ``.{name}.done``.
        title: Canvas display label.
    """

    name: str
    smk_file: str
    kind: RuleKind = RuleKind.CASE
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    stamp_name: str = ""
    title: str = ""

    @property
    def keyed_by(self) -> str | None:
        """``"mesh"`` when the rule runs once per mesh variant, else ``None``."""
        return MESH_DIM if self.kind in _MESH_KEYED else None

    @property
    def consumes_case_dims(self) -> bool:
        """``setup`` — gets every non-mesh dimension as a config port."""
        return self.kind is RuleKind.CASE_SETUP

    @property
    def consumes_mesh_dim(self) -> bool:
        """``setup_mesh`` — gets the mesh dimension's config port."""
        return self.kind is RuleKind.MESH_STAGE

    @property
    def creates_mesh(self) -> bool:
        """A valid start of the mesh tool chain (``blockMesh``)."""
        return self.kind is RuleKind.MESH_CREATE

    @property
    def is_mesh_tool(self) -> bool:
        """A keyed rule in the mesh tool chain (not the staging rule)."""
        return self.kind in _MESH_TOOLS

    @property
    def stamp(self) -> str:
        """The per-variant stamp a mesh tool writes (``.blockMesh.done``)."""
        return self.stamp_name or f".{self.name}.done"


@dataclass(frozen=True)
class RulePlan:
    """A validated rule selection resolved into concrete wiring.

    Attributes:
        rules: The enabled rules, in registry order.
        mesh_tool_input: Mesh-tool rule name -> its predecessor's stamp
            basename (the first tool chains off ``MESH_STAGE_STAMP``).
        mesh_done: The mesh chain's sink stamp basename (what ``setup`` waits on).
        final_pattern: What the ``all`` rule expands over.
    """

    rules: tuple[RuleSpec, ...]
    mesh_tool_input: dict[str, str]
    mesh_done: str
    final_pattern: str

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.rules)

    def get(self, name: str) -> RuleSpec:
        for spec in self.rules:
            if spec.name == name:
                return spec
        msg = f"rule '{name}' is not part of this plan (enabled: {list(self.names)})"
        raise KeyError(msg)


class RuleRegistry:
    """The predefined rules, in pipeline order."""

    def __init__(self, rules: Sequence[RuleSpec]) -> None:
        self._rules = tuple(rules)
        names = [spec.name for spec in self._rules]
        if len(set(names)) != len(names):
            msg = f"duplicate rule names in registry: {names}"
            raise ValueError(msg)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self._rules)

    def get(self, name: str) -> RuleSpec:
        for spec in self._rules:
            if spec.name == name:
                return spec
        msg = f"unknown rule '{name}' (registered: {list(self.names)})"
        raise KeyError(msg)

    def plan(self, enabled: Sequence[str] | None = None) -> RulePlan:
        """Resolve a rule selection into a validated :class:`RulePlan`.

        Selection order is irrelevant — rules resolve in registry (pipeline)
        order. Raises ``ValueError`` on unknown names, a missing required rule,
        a mesh chain that does not start with a mesh-creating tool, or ``post``
        without ``solve``.
        """
        chosen = set(DEFAULT_ENABLED if enabled is None else enabled)
        unknown = chosen - set(self.names)
        if unknown:
            msg = f"unknown rule(s) {sorted(unknown)} (registered: {list(self.names)})"
            raise ValueError(msg)
        for required in ("all", "setup", "solve", "setup_mesh"):
            if required not in chosen:
                msg = f"rule '{required}' is required in every sweep pipeline"
                raise ValueError(msg)

        rules = tuple(spec for spec in self._rules if spec.name in chosen)

        chain = [spec for spec in rules if spec.is_mesh_tool]
        if not chain:
            msg = "the mesh chain needs at least one mesh tool rule (e.g. blockMesh)"
            raise ValueError(msg)
        # setup_mesh stages a clean variant, so the first tool in the chain must
        # create the mesh.
        if chain[0].kind is not RuleKind.MESH_CREATE:
            msg = (
                f"the mesh chain starts at '{chain[0].name}', which does not create "
                f"a mesh — setup_mesh stages a clean variant, so the first tool must "
                f"be a mesh creator (e.g. blockMesh)"
            )
            raise ValueError(msg)

        mesh_tool_input: dict[str, str] = {}
        prev = MESH_STAGE_STAMP
        for spec in chain:
            mesh_tool_input[spec.name] = prev
            prev = spec.stamp

        if "post" in chosen:
            final = POST_DONE_PATTERN
        else:
            final = RUN_DONE_PATTERN

        return RulePlan(
            rules=rules,
            mesh_tool_input=mesh_tool_input,
            mesh_done=prev,
            final_pattern=final,
        )


#: The default pipeline selection (post is registered but opt-in).
DEFAULT_ENABLED: tuple[str, ...] = (
    "all",
    "setup_mesh",
    "blockMesh",
    "snappyHexMesh",
    "checkMesh",
    "setup",
    "solve",
)


def default_registry() -> RuleRegistry:
    """The NeoFOAM rule library, in pipeline order."""
    return RuleRegistry(
        [
            # `all` is emitted INLINE by the Snakefile generator: Snakemake's
            # implicit default target must be a rule of the MAIN Snakefile —
            # rules from include:d files are never picked as the default.
            RuleSpec(
                name="all",
                smk_file="",
                kind=RuleKind.AGGREGATE,
                title="all",
            ),
            RuleSpec(
                name="setup_mesh",
                smk_file="setup_mesh.smk",
                kind=RuleKind.MESH_STAGE,
                inputs=(MESH_CONFIG_PATTERN,),
                outputs=(f"meshes/{{mesh}}/{MESH_STAGE_STAMP}",),
                title="stage mesh variant",
            ),
            RuleSpec(
                name="blockMesh",
                smk_file="block_mesh.smk",
                kind=RuleKind.MESH_CREATE,
                inputs=(f"meshes/{{mesh}}/{MESH_STAGE_STAMP}",),
                outputs=("meshes/{mesh}/.blockMesh.done",),
                title="blockMesh",
            ),
            RuleSpec(
                name="snappyHexMesh",
                smk_file="snappy_hex_mesh.smk",
                kind=RuleKind.MESH_TOOL,
                inputs=("meshes/{mesh}/.blockMesh.done",),
                outputs=("meshes/{mesh}/.snappyHexMesh.done",),
                title="snappyHexMesh",
            ),
            RuleSpec(
                name="checkMesh",
                smk_file="check_mesh.smk",
                kind=RuleKind.MESH_TOOL,
                inputs=("meshes/{mesh}/.snappyHexMesh.done",),
                outputs=("meshes/{mesh}/.checkMesh.done",),
                title="checkMesh",
            ),
            RuleSpec(
                name="setup",
                smk_file="setup.smk",
                kind=RuleKind.CASE_SETUP,
                inputs=(SETUP_CONFIG_PATTERN, "meshes/{mesh}/<mesh_done>"),
                outputs=(SETUP_STAMP_PATTERN,),
                title="clone base + apply configs",
            ),
            RuleSpec(
                name="solve",
                smk_file="solve.smk",
                kind=RuleKind.CASE,
                inputs=(SETUP_STAMP_PATTERN,),
                outputs=(RUN_DONE_PATTERN,),
                title="solve",
            ),
            RuleSpec(
                name="post",
                smk_file="post.smk",
                kind=RuleKind.CASE,
                inputs=(RUN_DONE_PATTERN,),
                outputs=(POST_DONE_PATTERN,),
                title="post-process",
            ),
        ]
    )
