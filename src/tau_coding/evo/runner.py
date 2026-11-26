"""One task, one frozen CodingSession, then a host-owned verifier."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from uuid import uuid4

from tau_agent.events import ToolExecutionStartEvent, TurnStartEvent
from tau_agent.messages import AssistantMessage, sum_usage
from tau_agent.provider import ModelProvider
from tau_agent.session import JsonlSessionStorage
from tau_agent.tools import AgentTool
from tau_coding.evo.models import TaskRun, VerifierFeedback, digest, now
from tau_coding.evo.selection import render_selection, select_skills
from tau_coding.evo.store import SkillStore, atomic_write
from tau_coding.paths import TauPaths
from tau_coding.resources import TauResourcePaths
from tau_coding.session import CodingSessionConfig
from tau_coding.session_preparation import prepare_coding_session
from tau_coding.system_prompt import BuildSystemPromptOptions, build_system_prompt
from tau_coding.version import current_version

Verifier = Callable[[Path], Awaitable[VerifierFeedback]]


async def run_task(
    *,
    store: SkillStore,
    provider: ModelProvider,
    model: str,
    provider_name: str,
    task_id: str,
    prompt: str,
    cwd: Path,
    tools: Sequence[AgentTool],
    verifier: Verifier | None = None,
    max_bytes: int = 8000,
    max_skills: int = 3,
    max_turns: int = 12,
    timeout_seconds: float = 180,
) -> TaskRun:
    """Caller owns the provider and workspace; neither is a security sandbox."""
    if max_turns < 1 or timeout_seconds <= 0 or not prompt.strip():
        raise ValueError("Provide a prompt, positive max_turns and timeout_seconds")
    cwd = cwd.resolve(strict=True)
    selection = select_skills(
        store.snapshot(),
        prompt,
        max_bytes=max_bytes,
        max_skills=max_skills,
    )
    system = build_system_prompt(
        BuildSystemPromptOptions(
            cwd=cwd,
            tools=tools,
            append_system_prompt=render_selection(selection),
        )
    )
    run_id = uuid4().hex
    run_dir = store.root / "runs" / run_id
    run_dir.mkdir(parents=True)
    record = TaskRun(
        run_id=run_id,
        task_id=task_id,
        prompt_sha256=digest(prompt),
        system_sha256=digest(system),
        tools_sha256=digest(
            json.dumps(
                [(tool.name, tool.description, tool.parameters) for tool in tools],
                sort_keys=True,
            )
        ),
        model=model,
        provider=provider_name,
        solver_version=current_version(),
        selection=selection,
        started_at=now(),
        max_turns=max_turns,
        timeout_seconds=timeout_seconds,
    )
    record_path = run_dir / "run.json"
    atomic_write(record_path, record.model_dump_json(indent=2))
    atomic_write(run_dir / "system.txt", system)
    paths = TauPaths(home=run_dir / "tau", agents_home=run_dir / "agents")
    session = None
    try:
        async with asyncio.timeout(timeout_seconds):
            prepared = await prepare_coding_session(
                CodingSessionConfig(
                    provider=provider,
                    model=model,
                    provider_name=provider_name,
                    storage=JsonlSessionStorage(run_dir / "session.jsonl"),
                    cwd=cwd,
                    system=system,
                    tools=list(tools),
                    allowed_tool_names=frozenset(tool.name for tool in tools),
                    resource_paths=TauResourcePaths(
                        root=paths.home,
                        agents_root=None,
                        paths=paths,
                    ),
                    skills_enabled=False,
                    extensions_enabled=False,
                    trust_override="decline",
                    auto_compact_enabled=False,
                )
            )
            session = await prepared.adopt()
            await session.emit_pending_session_start()
            await session.append_custom_entry(
                "tau.evo",
                {
                    "run_id": run_id,
                    "snapshot": selection.snapshot,
                    "system_sha256": record.system_sha256,
                },
            )
            async for event in session.prompt(prompt):
                if isinstance(event, TurnStartEvent):
                    if record.turns >= max_turns:
                        session.cancel()
                        record = record.model_copy(update={"error_type": "TurnLimit"})
                    else:
                        record = record.model_copy(update={"turns": record.turns + 1})
                elif isinstance(event, ToolExecutionStartEvent):
                    record = record.model_copy(update={"tool_calls": record.tool_calls + 1})
            assistants = [m for m in session.messages if isinstance(m, AssistantMessage)]
            completed = (
                bool(assistants)
                and assistants[-1].stop_reason not in {"error", "aborted", "length"}
                and record.error_type is None
            )
            record = record.model_copy(
                update={
                    "status": "completed" if completed else "error",
                    "usage": sum_usage(m.usage for m in assistants),
                }
            )
            if completed and verifier is not None:
                feedback = await verifier(cwd)
                record = record.model_copy(update={"feedback": feedback})
    except TimeoutError:
        record = record.model_copy(update={"status": "timeout", "error_type": "TimeoutError"})
    except asyncio.CancelledError:
        record = record.model_copy(update={"status": "cancelled", "error_type": "CancelledError"})
        raise
    except Exception as exc:
        # Exception messages and provider error bodies can contain secrets.
        record = record.model_copy(update={"status": "error", "error_type": type(exc).__name__})
    finally:
        try:
            if session is not None:
                session.cancel()
                # Completed responses still incur usage if a later tool/request is interrupted.
                record = record.model_copy(
                    update={
                        "usage": sum_usage(
                            message.usage
                            for message in session.messages
                            if isinstance(message, AssistantMessage)
                        )
                    }
                )
                await session.aclose()
        finally:
            record = record.model_copy(update={"finished_at": now()})
            atomic_write(record_path, record.model_dump_json(indent=2))
    return record
