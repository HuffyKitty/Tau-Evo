"""Read-only Evo skills for Tau's ordinary frontends; opt in with tau -e DIRECTORY."""

from __future__ import annotations

import os
from pathlib import Path

from tau_coding.evo.selection import render_selection, select_skills
from tau_coding.evo.store import SkillStore
from tau_coding.extensions import ExtensionAPI, ExtensionContext, InputEvent, InputHookResult
from tau_coding.paths import TauPaths


def setup(tau: ExtensionAPI) -> None:
    # One frozen selection per extension generation. Use a fresh session for a new task.
    consumed = False
    selected_version: str | None = None
    settled = False

    async def on_input(event: object, context: ExtensionContext) -> InputHookResult | None:
        nonlocal consumed, selected_version
        if not isinstance(event, InputEvent):
            return None
        if (
            consumed
            or context.is_running
            or event.source != "interactive"
            or event.streaming_behavior is not None
        ):
            return None
        # Leave explicit skills and templates to Tau's native expansion.
        if event.text.lstrip().startswith("/"):
            consumed = True
            return None
        try:
            root = Path(os.environ.get("TAU_EVO_HOME", str(TauPaths().home / "evo")))
            selection = select_skills(SkillStore(root).snapshot(), event.text)
            await tau.append_entry("tau.evo.selection", selection.model_dump(mode="json"))
        except (OSError, ValueError, RuntimeError):
            return InputHookResult(action="handled", message="Evo skill state is unreadable.")
        consumed = True
        selected_version = selection.snapshot
        guidance = render_selection(selection)
        if not guidance:
            return None
        return InputHookResult(
            action="transform",
            text=f"{event.text}\n\n<evo_guidance>\n{guidance}</evo_guidance>",
        )

    async def on_settled(event: object, context: ExtensionContext) -> None:
        nonlocal settled
        if selected_version is not None and not settled:
            settled = True
            await tau.append_entry(
                "tau.evo.settled",
                {
                    "snapshot": selected_version,
                    "verification": "unknown",
                },
            )

    tau.on("input", on_input)
    tau.on("agent_settled", on_settled)
