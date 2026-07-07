# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""AI chat: a multi-turn assistant that fills the wizard forms (headless logic).

Mirrors the marimo wizard's ``mo.ui.chat`` behaviour: each user message runs the
pydantic-ai case agent with the running ``message_history`` (so follow-ups refine the
same case), pushes the produced values into the form state objects, auto-selects the
optional models it filled, auto-saves the AI-produced configs to the target dir, and
replies with a summary (**Filled / Selected models / Wrote to**). ``await agent.run``
(never ``run_sync``) since trame owns the loop; a missing ``ANTHROPIC_API_KEY`` degrades
to a chat message. The chat *widgets* are rendered by ``app.py``; this module owns the
logic so it stays unit-testable without trame or a browser.
"""

from __future__ import annotations

import os
from typing import Any, Callable

from neofoam.agent.case_fill import build_case_agent, case_spec_to_configs
from neofoam.agent.case_fill import save_case as save_case_spec
from neofoam.ui.case_spec import configs_to_form_state, models_filled_by
from neofoam.ui.forms import FormEntry
from neofoam.ui.geometry_agent import (
    apply_assignments,
    build_geometry_agent,
    geometry_prompt,
)

__all__ = ["build_agent_panel", "SUGGESTED_PROMPTS"]

#: Default case-fill model. Overridable per session via ``NEOFOAM_CASE_MODEL`` —
#: a stronger model (e.g. Sonnet) produces far fewer OpenFOAM-invalid-but-schema-
#: valid dictionaries (malformed grad/div schemes, missing Final solvers), which
#: the save-time validation cannot catch. Read at agent-build time so the env var
#: can be set after import.
_MODEL_NAME = "claude-haiku-4-5"


def _case_model_name() -> str:
    return os.environ.get("NEOFOAM_CASE_MODEL", _MODEL_NAME)


SUGGESTED_PROMPTS = [
    "Lid-driven cavity, laminar; top patch movingWall, other patches fixedWalls.",
    "Buoyant hot room with Boussinesq and kEpsilon; patches top bottom left right front back.",
    "Turbulent pipe flow, kOmegaSST, inlet fixedValue velocity, outlet zeroGradient.",
]


def build_agent_panel(
    server: Any,
    entries: list[FormEntry],
    solver: Any,
    *,
    agent_factory: Callable[..., Any] = build_case_agent,
    geometry_agent_factory: Callable[..., Any] = build_geometry_agent,
) -> Callable[..., Any]:
    """Register the chat state (``chat_log``/``chat_input``/``ai_busy``) + the async
    ``send_message`` controller on ``server``. Returns the coroutine function.

    The same prompt also drives the geometry stage: when patches have been scanned
    (``state.geometry_patches``), it is run through the geometry agent to assign
    patch roles / refinement, keeping the physics fill unchanged.
    """
    state, ctrl = server.state, server.controller
    state.chat_input = ""
    state.chat_log = []  # [{"role": "user"|"assistant", "content": str}]
    state.ai_busy = False
    state.suggested_prompts = list(SUGGESTED_PROMPTS)
    state.geometry_patches = []  # populated by the geometry step's scan (app.py)

    by_key = {e.key: e for e in entries}
    agent_cache: dict[str, Any] = {}
    geo_cache: dict[str, Any] = {}
    history: list[Any] = []  # pydantic-ai message history → multi-turn refinement
    # Per-step chat handlers (a plugin can own the prompt while its step is active).
    chat_handlers: dict[str, Callable[[str], Any]] = {}

    def register_chat_handler(step_id: str, handler: Callable[[str], Any]) -> None:
        """Route the chat prompt to ``handler`` while ``step_id`` is the active step.

        A step plugin (e.g. the CAD step) registers here so that typing in the shared
        AI assistant drives *its* agent instead of the physics fill; the handler runs
        for ``prompt`` and returns an optional assistant summary string.
        """
        chat_handlers[step_id] = handler

    ctrl.register_chat_handler = register_chat_handler

    def _say(role: str, content: str) -> None:
        state.chat_log = [*state.chat_log, {"role": role, "content": content}]

    async def _fill_geometry(prompt: str) -> None:
        """Assign patch roles / refinement from ``prompt`` when patches are loaded."""
        if not state.geometry_patches:
            return
        try:
            agent = geo_cache.get("agent")
            if agent is None:
                agent = geometry_agent_factory()
                geo_cache["agent"] = agent
        except Exception:  # noqa: BLE001 - geometry AI unavailable; physics reported it
            return
        try:
            names = [p["name"] for p in state.geometry_patches]
            result = await agent.run(geometry_prompt(names, prompt))
            state.geometry_patches = apply_assignments(
                state.geometry_patches, result.output
            )
            changed = sorted({a.patch for a in result.output.assignments})
            if changed:
                _say("assistant", "**Mesh roles set:** " + ", ".join(changed))
        except Exception as exc:  # noqa: BLE001 - report, don't crash the chat
            _say("assistant", f"_(mesh role fill skipped: {exc})_")

    async def send_message(text: str | None = None) -> None:
        prompt = (text if text is not None else state.chat_input).strip()
        if not prompt:
            return
        _say("user", prompt)
        state.chat_input = ""
        state.ai_busy = True
        try:
            # A step plugin can own the prompt while its step is active (e.g. CAD).
            handler = chat_handlers.get(state.current_step)
            if handler is not None:
                try:
                    reply = await handler(prompt)
                except Exception as exc:  # noqa: BLE001 - report, don't crash the chat
                    reply = f"**Failed:** {exc}"
                if reply:
                    _say("assistant", str(reply))
                return

            try:
                agent = agent_cache.get("agent")
                if agent is None:
                    agent = agent_factory(solver=solver, model_name=_case_model_name())
                    agent_cache["agent"] = agent
            except Exception as exc:  # noqa: BLE001
                _say(
                    "assistant",
                    f"**AI unavailable** — set `ANTHROPIC_API_KEY`.\n\n{exc}",
                )
                return

            try:
                result = await agent.run(prompt, message_history=history)
                history[:] = result.all_messages()
                configs = case_spec_to_configs(result.output)

                for key, data in configs_to_form_state(entries, configs).items():
                    state[by_key[key].state_key] = data
                filled_models = models_filled_by(entries, configs)
                for model in filled_models:
                    state[f"sel_{model}"] = True

                _say(
                    "assistant",
                    _summary(configs, filled_models, result, state.target_dir),
                )
            except Exception as exc:  # noqa: BLE001
                _say("assistant", f"**Fill failed:** {exc}")

            # Same prompt also assigns mesh patch roles / refinement (if scanned).
            await _fill_geometry(prompt)
        finally:
            state.ai_busy = False

    def _summary(
        configs: list[Any], filled_models: set[str], result: Any, target: str
    ) -> str:
        names = sorted(type(c).__name__ for c in configs)
        lines = ["**Filled:** " + (", ".join(names) if names else "_nothing_")]
        if filled_models:
            lines.append("**Selected models:** " + ", ".join(sorted(filled_models)))
        if target and configs:
            try:
                written = save_case_spec(result.output, target)
                lines += [f"**Wrote to** `{target}`:"] + [
                    f"- {p.name}" for p in written
                ]
            except Exception as exc:  # noqa: BLE001 - report, don't crash the chat
                lines.append(f"_(auto-save skipped: {exc})_")
        lines.append("Review the forms; click **Save case** when ready.")
        return "\n\n".join(lines)

    ctrl.send_message = send_message
    return send_message
