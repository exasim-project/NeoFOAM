# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""AI helper for the geometry stage: assign patch roles + refinement from prose.

A small pydantic-ai agent whose output is a list of ``(patch, role, refinement)``
assignments. Given the discovered patch names and a natural-language description
(e.g. *"the tubes are heated walls, refine them to level 3"*), it decides each
patch's CFD role and, for snappy surfaces, its refinement levels. It mirrors
:func:`neofoam.agent.case_fill.build_case_agent` (Anthropic ``claude-haiku-4-5`` by
default, ``await agent.run``); the caller degrades gracefully when
``ANTHROPIC_API_KEY`` is unset. Kept separate from the physics ``CaseSpec`` model so
meshing stays orthogonal to the solver configs.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from neofoam.ui.geometry import PatchRole

__all__ = [
    "RoleAssignment",
    "GeometryAssignments",
    "GEOMETRY_SYSTEM_PROMPT",
    "build_geometry_agent",
    "apply_assignments",
]

GEOMETRY_SYSTEM_PROMPT = (
    "You assign OpenFOAM boundary-patch roles for a CFD case. You are given the"
    " patch names discovered from the mesh geometry and a description of the case."
    " For each patch, choose a role from: inlet, outlet, wall, symmetry, empty."
    " For patches that are refinement surfaces (curved/interior geometry such as"
    " tubes or an object), you may also set (min, max) snappy refinement levels."
    " Only return assignments for patches you are confident about; leave the rest"
    " untouched. Never invent patch names — use exactly the names provided."
)


class RoleAssignment(BaseModel):
    """A single patch's assigned role and optional snappy refinement."""

    patch: str
    """The exact patch name (must match a discovered patch)."""
    role: PatchRole
    refinement: Optional[tuple[int, int]] = None
    """``(min, max)`` snappy surface refinement, when the patch is a surface."""


class GeometryAssignments(BaseModel):
    """The agent's output: role/refinement assignments to apply."""

    assignments: list[RoleAssignment] = Field(default_factory=list)


def build_geometry_agent(
    *,
    model: Any = None,
    model_name: str = "claude-haiku-4-5",
    system_prompt: str = GEOMETRY_SYSTEM_PROMPT,
    **agent_kwargs: Any,
) -> Any:
    """Build a pydantic-ai ``Agent`` returning :class:`GeometryAssignments`."""
    from pydantic_ai import Agent

    if model is None:
        from pydantic_ai.models.anthropic import AnthropicModel

        model = AnthropicModel(model_name)

    return Agent(
        model,
        output_type=GeometryAssignments,
        system_prompt=system_prompt,
        **agent_kwargs,
    )


def geometry_prompt(patch_names: list[str], description: str) -> str:
    """Render the patch list + user description into one prompt body."""
    listed = ", ".join(patch_names) if patch_names else "(none)"
    return (
        f"Patches: {listed}.\n\n"
        f"Case description: {description}\n\n"
        "Assign a role (and refinement for surfaces) to the patches you are sure of."
    )


def apply_assignments(
    patches: list[dict[str, Any]],
    assignments: GeometryAssignments,
) -> list[dict[str, Any]]:
    """Apply the agent's assignments onto the wizard's patch-state dicts (pure).

    ``patches`` are the ``state.geometry_patches`` rows (``{"name", "role",
    "refinement", ...}``). Each assignment updates the matching row's ``role`` (and
    ``refinement`` when given and the patch is a snappy surface). Unknown patch
    names are ignored. A new list is returned (the input is not mutated).
    """
    by_name = {a.patch: a for a in assignments.assignments}
    out: list[dict[str, Any]] = []
    for row in patches:
        assignment = by_name.get(row["name"])
        if assignment is None:
            out.append(dict(row))
            continue
        updated = dict(row)
        updated["role"] = assignment.role.value
        if assignment.refinement is not None and not row.get("box_faces"):
            updated["refinement"] = list(assignment.refinement)
        out.append(updated)
    return out
