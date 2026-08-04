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

The agent also carries a ``load_case`` tool, so "open the case at <path>" reads an
existing OpenFOAM case straight off disk into the forms — deterministically, via
:func:`neofoam.agent.case_fill.load_case_from_disk`, never through the model's own
transcription of the dictionaries.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

from neofoam.agent.case_fill import (
    build_case_agent,
    case_spec_to_configs,
    load_case_from_disk,
)
from neofoam.io import write_configs
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
    "Load the case at /path/to/case, then raise endTime to 10.",
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
    # Seeded here, not in app.py: the panel is usable standalone, and this runs
    # after app.py's state block anyway. The geometry step's scan overwrites it.
    state.geometry_patches = []

    by_key = {e.key: e for e in entries}
    agent_cache: dict[str, Any] = {}
    geo_cache: dict[str, Any] = {}
    history: list[Any] = []  # pydantic-ai message history → multi-turn refinement
    # Handoff from the load_case tool back to send_message: the tool runs inside
    # agent.run, so it parks what it read here and the turn applies it afterwards.
    loaded: dict[str, Any] = {}
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

    def load_case(case_dir: str) -> str:
        """Load an existing OpenFOAM case from disk into the wizard.

        Call this whenever the user names a case directory to open, continue from or
        start out from. Every config file present in the case is read and applied to
        the wizard forms directly, so do NOT transcribe the values into your own
        output — leave those fields null unless the user asked you to change them.

        Args:
            case_dir: Path to the case directory (the one holding ``system/``).
        """
        path = Path(case_dir).expanduser()
        if not path.is_dir():
            return f"No case directory at {path}."
        warnings: list[dict[str, str]] = []
        try:
            spec = load_case_from_disk(path, solver=solver, warnings=warnings)
        except Exception as exc:  # noqa: BLE001 - report to the model, don't kill the run
            return f"Could not read {path}: {exc}"
        configs = case_spec_to_configs(spec)
        loaded["dir"] = str(path)
        loaded["configs"] = configs

        names = sorted(type(c).__name__ for c in configs)
        reply = f"Loaded {len(names)} configs from {path}: {', '.join(names) or 'none'}."
        if warnings:
            reply += " Present but invalid: " + ", ".join(w["config"] for w in warnings) + "."
        return reply

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
            state.geometry_patches = apply_assignments(state.geometry_patches, result.output)
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
                    agent = agent_factory(
                        solver=solver,
                        model_name=_case_model_name(),
                        tools=[load_case],
                    )
                    agent_cache["agent"] = agent
            except Exception as exc:  # noqa: BLE001
                _say(
                    "assistant",
                    f"**AI unavailable** — set `ANTHROPIC_API_KEY`.\n\n{exc}",
                )
                return

            loaded.clear()  # only this turn's load_case call may push into the forms
            try:
                result = await agent.run(prompt, message_history=history)
                history[:] = result.all_messages()
                # A tool-loaded case is the baseline; the agent's own output refines
                # it, so the agent's configs are applied last and win per entry.
                configs = [*loaded.get("configs", []), *case_spec_to_configs(result.output)]

                for key, data in configs_to_form_state(entries, configs).items():
                    state[by_key[key].state_key] = data
                filled_models = models_filled_by(entries, configs)
                for model in filled_models:
                    state[f"sel_{model}"] = True

                _say(
                    "assistant",
                    _summary(configs, filled_models, state.target_dir, loaded.get("dir")),
                )
            except Exception as exc:  # noqa: BLE001
                _say("assistant", f"**Fill failed:** {exc}")

            # Same prompt also assigns mesh patch roles / refinement (if scanned).
            await _fill_geometry(prompt)
        finally:
            state.ai_busy = False

    def _summary(
        configs: list[Any], filled_models: set[str], target: str, source: str | None
    ) -> str:
        names = sorted(type(c).__name__ for c in configs)
        lines = [f"**Loaded** `{source}`"] if source else []
        lines.append("**Filled:** " + (", ".join(names) if names else "_nothing_"))
        if filled_models:
            lines.append("**Selected models:** " + ", ".join(sorted(filled_models)))
        if target and configs:
            try:
                # Not save_case(result.output): a loaded case lives in `configs`, not
                # in the agent's own output, and must be written out too.
                written = write_configs(configs, target)
                lines += [f"**Wrote to** `{target}`:"] + [f"- {Path(f).name}" for f in written]
            except Exception as exc:  # noqa: BLE001 - report, don't crash the chat
                lines.append(f"_(auto-save skipped: {exc})_")
        lines.append("Review the forms; click **Save case** when ready.")
        return "\n\n".join(lines)

    ctrl.send_message = send_message
    return send_message
