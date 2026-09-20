# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""AI chat: a multi-turn assistant that fills the wizard forms (headless logic).

Mirrors the marimo wizard's ``mo.ui.chat`` behaviour: each user message runs the
pydantic-ai case agent with the running ``message_history`` (so follow-ups refine the
same case), pushes the produced values into the form state objects, auto-selects the
models it filled, auto-saves the AI-produced configs to the target dir, and
replies with a summary (**Filled / Selected models / Wrote to**). ``await agent.run``
(never ``run_sync``) since trame owns the loop; a missing ``ANTHROPIC_API_KEY`` degrades
to a chat message. :meth:`AgentPanel.render` draws the chat with the widget modules
it is handed, so the logic stays unit-testable without trame or a browser.

The agent also carries a ``load_case`` tool, so "open the case at <path>" reads an
existing OpenFOAM case straight off disk into the forms — deterministically, via
:func:`neofoam.ui.case_load.read_case_configs`, never through the model's own
transcription of the dictionaries.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

from neofoam.agent.case_fill import build_case_agent, case_spec_to_configs
from neofoam.io import write_configs
from neofoam.ui._responsive import _MOBILE, _responsive_open
from neofoam.ui.case_load import apply_configs_to_forms, read_case_configs
from neofoam.ui.forms import FormEntry
from neofoam.ui.geometry_agent import (
    apply_assignments,
    build_geometry_agent,
    geometry_prompt,
)
from neofoam.ui.steps import build_model_families

__all__ = ["AgentPanel", "build_agent_panel", "SUGGESTED_PROMPTS"]

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


def _summary(
    configs: list[Any], filled_models: set[str], source: str | None, saved: list[str]
) -> str:
    """The assistant's reply for one fill; ``saved`` is the auto-save's own report."""
    names = sorted(type(c).__name__ for c in configs)
    lines = [f"**Loaded** `{source}`"] if source else []
    lines.append("**Filled:** " + (", ".join(names) if names else "_nothing_"))
    if filled_models:
        lines.append("**Selected models:** " + ", ".join(sorted(filled_models)))
    lines += saved
    lines.append("Review the forms; click **Save case** when ready.")
    return "\n\n".join(lines)


class AgentPanel:
    """Chat state, the ``send_message`` controller and the assistant drawer."""

    def __init__(
        self,
        server: Any,
        entries: list[FormEntry],
        solver: Any,
        *,
        agent_factory: Callable[..., Any] = build_case_agent,
        geometry_agent_factory: Callable[..., Any] = build_geometry_agent,
    ):
        self._state = server.state
        self._ctrl = server.controller
        self._entries = entries
        self._solver = solver
        self._families = build_model_families(solver)
        self._agent_factory = agent_factory
        self._geometry_agent_factory = geometry_agent_factory
        # Both agents are built on first use: building needs the API key.
        self._agent: Any = None
        self._geometry_agent: Any = None
        self._history: list[Any] = []  # pydantic-ai message history → multi-turn refinement
        # Handoff from the load_case tool back to send_message: the tool runs inside
        # agent.run, so it parks what it read here and the turn applies it afterwards.
        self._loaded: dict[str, Any] = {}
        # Per-step chat handlers (a plugin can own the prompt while its step is active).
        self._chat_handlers: dict[str, Callable[[str], Any]] = {}

        state = self._state
        state.chat_input = ""
        state.chat_log = []  # [{"role": "user"|"assistant", "content": str}]
        state.ai_busy = False
        state.suggested_prompts = list(SUGGESTED_PROMPTS)
        # Owned by the geometry panel; defaulted here so the chat is usable standalone.
        state.setdefault("geometry_patches", [])
        self._ctrl.register_chat_handler = self.register_chat_handler
        self._ctrl.send_message = self.send_message

    def register_chat_handler(self, step_id: str, handler: Callable[[str], Any]) -> None:
        """Route the chat prompt to ``handler`` while ``step_id`` is the active step.

        A step plugin (e.g. the CAD step) registers here so that typing in the shared
        AI assistant drives *its* agent instead of the physics fill; the handler runs
        for ``prompt`` and returns an optional assistant summary string.
        """
        self._chat_handlers[step_id] = handler

    def _say(self, role: str, content: str) -> None:
        self._state.chat_log = [*self._state.chat_log, {"role": role, "content": content}]

    def load_case(self, case_dir: str) -> str:
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
            configs = read_case_configs(path, self._solver, warnings)
        except Exception as exc:  # noqa: BLE001 - report to the model, don't kill the run
            return f"Could not read {path}: {exc}"
        self._loaded["dir"] = str(path)
        self._loaded["configs"] = configs

        names = sorted(type(c).__name__ for c in configs)
        reply = f"Loaded {len(names)} configs from {path}: {', '.join(names) or 'none'}."
        if warnings:
            reply += " Present but invalid: " + ", ".join(w["config"] for w in warnings) + "."
        return reply

    async def send_message(self, text: str | None = None) -> None:
        """Run one chat turn: the active step's handler if it has one, else the case fill."""
        state = self._state
        prompt = (text if text is not None else state.chat_input).strip()
        if not prompt:
            return
        # The widgets are disabled while busy, but a queued click still lands here:
        # a second concurrent turn would start from an empty `history` and then
        # overwrite the first turn's transcript.
        if state.ai_busy:
            return
        self._say("user", prompt)
        state.chat_input = ""
        state.ai_busy = True
        try:
            # A step plugin can own the prompt while its step is active (e.g. CAD).
            handler = self._chat_handlers.get(state.current_step)
            if handler is not None:
                await self._run_step_handler(handler, prompt)
            else:
                await self._fill_case(prompt)
        finally:
            state.ai_busy = False

    async def _run_step_handler(self, handler: Callable[[str], Any], prompt: str) -> None:
        try:
            reply = await handler(prompt)
        except Exception as exc:  # noqa: BLE001 - report, don't crash the chat
            reply = f"**Failed:** {exc}"
        if reply:
            self._say("assistant", str(reply))

    async def _fill_case(self, prompt: str) -> None:
        """Fill the forms from ``prompt``, then assign the mesh patch roles from it too."""
        try:
            if self._agent is None:
                self._agent = self._agent_factory(
                    solver=self._solver,
                    model_name=_case_model_name(),
                    tools=[self.load_case],
                )
        except Exception as exc:  # noqa: BLE001
            self._say(
                "assistant",
                f"**AI unavailable** — set `ANTHROPIC_API_KEY`.\n\n{exc}",
            )
            return

        self._loaded.clear()  # only this turn's load_case call may push into the forms
        try:
            result = await self._agent.run(prompt, message_history=self._history)
            self._history[:] = result.all_messages()
            # A tool-loaded case is the baseline; the agent's own output refines
            # it, so the agent's configs are applied last and win per entry.
            configs = [*self._loaded.get("configs", []), *case_spec_to_configs(result.output)]

            loaded_dir = self._loaded.get("dir")
            filled_models = apply_configs_to_forms(
                self._state,
                self._entries,
                self._families,
                configs,
                Path(loaded_dir) if loaded_dir else None,
            )
            saved = self._autosave(configs)
            self._say("assistant", _summary(configs, filled_models, self._loaded.get("dir"), saved))
        except Exception as exc:  # noqa: BLE001
            self._say("assistant", f"**Fill failed:** {exc}")

        # Same prompt also assigns mesh patch roles / refinement (if scanned).
        await self._fill_geometry(prompt)

    def _autosave(self, configs: list[Any]) -> list[str]:
        """Write ``configs`` to the target dir; returns the reply lines reporting it."""
        target = self._state.target_dir
        if not (target and configs):
            return []
        try:
            # Not save_case(result.output): a loaded case lives in `configs`, not
            # in the agent's own output, and must be written out too.
            written = write_configs(configs, target)
        except Exception as exc:  # noqa: BLE001 - report, don't crash the chat
            return [f"_(auto-save skipped: {exc})_"]
        return [f"**Wrote to** `{target}`:"] + [f"- {Path(f).name}" for f in written]

    async def _fill_geometry(self, prompt: str) -> None:
        """Assign patch roles / refinement from ``prompt`` when patches are loaded."""
        state = self._state
        if not state.geometry_patches:
            return
        try:
            if self._geometry_agent is None:
                self._geometry_agent = self._geometry_agent_factory()
        except Exception:  # noqa: BLE001 - geometry AI unavailable; physics reported it
            return
        try:
            names = [p["name"] for p in state.geometry_patches]
            result = await self._geometry_agent.run(geometry_prompt(names, prompt))
            state.geometry_patches = apply_assignments(state.geometry_patches, result.output)
            changed = sorted({a.patch for a in result.output.assignments})
            if changed:
                self._say("assistant", "**Mesh roles set:** " + ", ".join(changed))
        except Exception as exc:  # noqa: BLE001 - report, don't crash the chat
            self._say("assistant", f"_(mesh role fill skipped: {exc})_")

    def render(self, v3: Any, html: Any) -> None:
        """Right-hand foldable AI chat drawer (multi-turn, fills the forms)."""
        ctrl = self._ctrl
        with v3.VNavigationDrawer(
            location="right",
            width=400,
            **_responsive_open("ai_panel", "ai_panel_mobile"),
        ):
            with html.Div(classes="d-flex flex-column", style="height: 100%;"):
                with v3.VToolbar(title="AI assistant", density="compact", flat=True):
                    # The overlay leaves only a sliver of scrim to tap on a phone.
                    v3.VBtn(
                        icon="mdi-close",
                        click="ai_panel_mobile = false",
                        v_if=_MOBILE,
                    )
                # Scrolling transcript.
                with html.Div(classes="flex-grow-1 pa-3", style="overflow-y: auto;"):
                    # Empty-state hint + suggested prompts.
                    with html.Div(v_show="!chat_log.length"):
                        v3.VCardText(
                            "Describe your case and I'll fill the forms. Try:",
                            classes="text-medium-emphasis px-0",
                        )
                        with v3.VChip(
                            v_for="(p, i) in suggested_prompts",
                            key="i",
                            click=(ctrl.send_message, "[p]"),
                            disabled=("ai_busy",),
                            size="small",
                            variant="tonal",
                            color="secondary",
                            classes="mb-2",
                            style="height: auto; white-space: normal;",
                        ):
                            html.Span("{{ p }}", classes="py-1")
                    # Messages.
                    with v3.VSheet(
                        v_for="(m, i) in chat_log",
                        key="i",
                        rounded="lg",
                        classes="pa-3 mb-2",
                        color=("m.role === 'user' ? 'primary' : 'surface-variant'",),
                    ):
                        html.Div(
                            "{{ m.content }}",
                            style="white-space: pre-wrap; font-size: 0.9rem;",
                        )
                    v3.VProgressLinear(indeterminate=True, v_show="ai_busy", color="secondary")
                # Composer pinned to the bottom.
                with html.Div(classes="pa-3"):
                    v3.VTextField(
                        v_model=("chat_input",),
                        placeholder="Message the assistant…",
                        hide_details=True,
                        keydown_enter=(ctrl.send_message, "[]"),
                        disabled=("ai_busy",),
                    )
                    v3.VBtn(
                        "Send",
                        click=(ctrl.send_message, "[]"),
                        loading=("ai_busy",),
                        disabled=("!chat_input || ai_busy",),
                        color="secondary",
                        prepend_icon="mdi-send",
                        block=True,
                        classes="mt-2",
                    )


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
    panel = AgentPanel(
        server,
        entries,
        solver,
        agent_factory=agent_factory,
        geometry_agent_factory=geometry_agent_factory,
    )
    return panel.send_message
