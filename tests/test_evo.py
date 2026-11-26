"""Evo contracts and real CodingSession boundaries, without network access."""

import asyncio
import json
import sys
from pathlib import Path

import pytest

from pi_event_helpers import assistant_done, assistant_start
from tau_agent.messages import AssistantMessage, ToolCall
from tau_agent.session import InMemorySessionStorage
from tau_ai.fake import FakeProvider
from tau_coding.evo.cli import read_api_key
from tau_coding.evo.models import SkillRevision, SkillSnapshot, VerifierFeedback
from tau_coding.evo.runner import run_task
from tau_coding.evo.selection import render_selection, select_skills
from tau_coding.evo.store import SkillStore
from tau_coding.evo.verifier import verify_command
from tau_coding.resources import TauResourcePaths
from tau_coding.session import CodingSession, CodingSessionConfig
from tau_coding.tools import create_write_tool


def snapshot(body: str = "Check boundary cases.") -> SkillSnapshot:
    return SkillSnapshot(
        skills=(
            SkillRevision(
                id="python",
                revision=1,
                triggers=("python",),
                body=body,
            ),
        )
    )


def fake(message: AssistantMessage | None = None) -> FakeProvider:
    return FakeProvider(
        [
            [
                assistant_start(),
                assistant_done(
                    message or AssistantMessage(content="Done"),
                ),
            ]
        ]
    )


def test_store_compare_and_swap_rollback_and_corruption(tmp_path: Path) -> None:
    store = SkillStore(tmp_path)
    first = snapshot()
    with pytest.raises(ValueError, match="Review"):
        store.publish(first, expected="empty", reviewed=False)
    version = store.publish(first, expected="empty", reviewed=True)
    frozen = store.snapshot()
    second = snapshot("Check a different case.")
    newer = store.publish(second, expected=version, reviewed=True)
    assert frozen == first
    with pytest.raises(ValueError, match="changed"):
        store.publish(first, expected=version, reviewed=True)
    assert store.snapshot().version == newer
    store.rollback(version, expected=newer)
    assert store.snapshot() == first
    (tmp_path / "snapshots" / f"{version}.json").write_text(
        second.model_dump_json(),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="checksum"):
        store.snapshot()
    with pytest.raises(ValueError, match="Invalid"):
        store.snapshot("../../credentials")


def test_writer_lock_and_failed_replace_preserve_current(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SkillStore(tmp_path)
    store.publish(snapshot(), expected="empty", reviewed=True)
    current = store.snapshot().version
    (tmp_path / "writer.lock").touch()
    with pytest.raises(RuntimeError, match="busy"):
        store.rollback("empty", expected=current)
    (tmp_path / "writer.lock").unlink()

    def fail_replace(*args: object) -> None:
        raise OSError("disk failure")

    monkeypatch.setattr("tau_coding.evo.store.os.replace", fail_replace)
    with pytest.raises(OSError):
        store.rollback("empty", expected=current)
    assert store.snapshot().version == current
    assert not (tmp_path / "writer.lock").exists()


def test_selection_budget_and_unicode_are_deterministic() -> None:
    skills = SkillSnapshot(
        skills=(
            SkillRevision(id="general", revision=1, scope="general", body="通用经验"),
            SkillRevision(id="python", revision=1, triggers=("python",), body="检查边界"),
            SkillRevision(id="unrelated", revision=1, triggers=("rust",), body="Rust"),
        )
    )
    selected = select_skills(skills, "PYTHON", max_skills=1)
    assert [s.id for s in selected.skills] == ["python"]
    assert select_skills(skills, "python", max_bytes=0).skills == ()
    assert select_skills(skills, "python", max_bytes=selected.bytes_used, max_skills=1) == (
        selected.model_copy(update={"max_bytes": selected.bytes_used})
    )
    assert "检查边界" in render_selection(selected)
    assert render_selection(select_skills(SkillSnapshot(), "x")) == ""
    with pytest.raises(ValueError):
        select_skills(skills, "python", max_bytes=-1)


@pytest.mark.anyio
async def test_runner_real_tool_then_external_verifier(tmp_path: Path) -> None:
    workspace = tmp_path / "work"
    workspace.mkdir()
    store = SkillStore(tmp_path / "evo")
    version = store.publish(snapshot(), expected="empty", reviewed=True)
    provider = FakeProvider(
        [
            [
                assistant_start(),
                assistant_done(
                    AssistantMessage(
                        content=[
                            ToolCall(
                                id="write1",
                                name="write",
                                arguments={"path": "answer.txt", "content": "42"},
                            )
                        ],
                        stop_reason="toolUse",
                    )
                ),
            ],
            [assistant_start(), assistant_done(AssistantMessage(content="Done"))],
        ]
    )

    async def verify(cwd: Path) -> VerifierFeedback:
        assert (cwd / "answer.txt").read_text() == "42"
        return VerifierFeedback(verifier="test", status="passed", exit_code=0)

    record = await run_task(
        store=store,
        provider=provider,
        model="fake",
        provider_name="fake",
        task_id="write",
        prompt="Use python to write the answer",
        cwd=workspace,
        tools=[create_write_tool(cwd=workspace)],
        verifier=verify,
    )
    assert record.status == "completed"
    assert record.feedback and record.feedback.status == "passed"
    assert record.tool_calls == 1
    assert record.selection.snapshot == version
    assert "Check boundary cases." in provider.calls[0][1]
    assert provider.calls[0][2][0].text == "Use python to write the answer"
    run_dir = store.root / "runs" / record.run_id
    assert json.loads((run_dir / "run.json").read_text())["finished_at"]
    assert "tau.evo" in (run_dir / "session.jsonl").read_text()


@pytest.mark.anyio
async def test_self_report_is_not_success_and_error_skips_verifier(tmp_path: Path) -> None:
    for message in (
        AssistantMessage(content="All tests passed!"),
        AssistantMessage(content="bad", stop_reason="error"),
    ):
        called = False

        async def fail_check(cwd: Path) -> VerifierFeedback:
            nonlocal called
            called = True
            return VerifierFeedback(verifier="test", status="failed", exit_code=1)

        record = await run_task(
            store=SkillStore(tmp_path / "evo"),
            provider=fake(message),
            model="fake",
            provider_name="fake",
            task_id="failure",
            prompt="task",
            cwd=tmp_path,
            tools=[],
            verifier=fail_check,
        )
        if message.stop_reason == "error":
            assert not called and record.feedback is None and record.status == "error"
        else:
            assert record.feedback and record.feedback.status == "failed"


@pytest.mark.anyio
async def test_cancel_and_timeout_leave_durable_run(tmp_path: Path) -> None:
    started = asyncio.Event()

    async def slow_verifier(cwd: Path) -> VerifierFeedback:
        started.set()
        await asyncio.sleep(30)
        return VerifierFeedback(verifier="slow", status="passed")

    store = SkillStore(tmp_path / "evo")
    task = asyncio.create_task(
        run_task(
            store=store,
            provider=fake(),
            model="fake",
            provider_name="fake",
            task_id="cancel",
            prompt="task",
            cwd=tmp_path,
            tools=[],
            verifier=slow_verifier,
        )
    )
    await asyncio.wait_for(started.wait(), 10)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    records = [json.loads(p.read_text()) for p in (store.root / "runs").glob("*/run.json")]
    assert records[0]["status"] == "cancelled"
    assert records[0]["finished_at"]
    result = await run_task(
        store=store,
        provider=fake(),
        model="fake",
        provider_name="fake",
        task_id="timeout",
        prompt="task",
        cwd=tmp_path,
        tools=[],
        verifier=slow_verifier,
        timeout_seconds=0.2,
    )
    assert result.status == "timeout" and result.feedback is None


@pytest.mark.anyio
async def test_verifier_success_failure_timeout_and_secret_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-secret")
    result = await verify_command(
        [
            sys.executable,
            "-c",
            "import os; assert 'DEEPSEEK_API_KEY' not in os.environ; print('ok')",
        ],
        tmp_path,
    )
    assert result.status == "passed" and result.output_sha256
    result = await verify_command([sys.executable, "-c", "raise SystemExit(7)"], tmp_path)
    assert result.status == "failed" and result.exit_code == 7
    result = await verify_command(
        [sys.executable, "-c", "import time; time.sleep(10)"],
        tmp_path,
        timeout_seconds=0.1,
    )
    assert result.status == "unknown" and result.reason == "timeout"


def test_credential_input_and_ambiguous_key_rejection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "key.txt"
    path.write_text('DEEPSEEK_API_KEY="sk-example123"', encoding="utf-8-sig")
    assert read_api_key(path) == "sk-example123"
    path.write_text("sk-one sk-two", encoding="utf-8")
    with pytest.raises(ValueError):
        read_api_key(path)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-example456")
    assert read_api_key(None) == "sk-example456"


@pytest.mark.anyio
async def test_extension_freezes_first_prompt_and_preserves_followup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SkillStore(tmp_path / "evo")
    store.publish(snapshot(), expected="empty", reviewed=True)
    monkeypatch.setenv("TAU_EVO_HOME", str(store.root))
    provider = FakeProvider(
        [
            [assistant_start(), assistant_done(AssistantMessage(content="One"))],
            [assistant_start(), assistant_done(AssistantMessage(content="Two"))],
        ]
    )
    extension = Path(__file__).parents[1] / "src/tau_coding/extensions/evo_harness"
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=provider,
            model="fake",
            storage=InMemorySessionStorage(),
            cwd=tmp_path,
            tools=[],
            system="Test",
            extensions_enabled=False,
            extension_paths=(extension,),
            trust_override="decline",
            resource_paths=TauResourcePaths(root=tmp_path / "tau", agents_root=None),
        )
    )
    try:
        _ = [event async for event in session.prompt("python task")]
        assert "<evo_guidance>" in provider.calls[0][2][0].text
        _ = [event async for event in session.prompt("followup")]
        assert provider.calls[1][2][-1].text == "followup"
    finally:
        await session.aclose()


@pytest.mark.anyio
async def test_turn_budget_prevents_another_provider_call(tmp_path: Path) -> None:
    provider = FakeProvider(
        [
            [
                assistant_start(),
                assistant_done(
                    AssistantMessage(
                        content=[
                            ToolCall(
                                id="write",
                                name="write",
                                arguments={"path": "x.txt", "content": "x"},
                            )
                        ],
                        stop_reason="toolUse",
                    )
                ),
            ],
            [assistant_start(), assistant_done(AssistantMessage(content="Should not run"))],
        ]
    )
    result = await run_task(
        store=SkillStore(tmp_path / "evo"),
        provider=provider,
        model="fake",
        provider_name="fake",
        task_id="limit",
        prompt="task",
        cwd=tmp_path,
        tools=[create_write_tool(cwd=tmp_path)],
        max_turns=1,
    )
    assert result.status == "error" and result.error_type == "TurnLimit"
    assert result.feedback is None
    assert len(provider.calls) == 1


@pytest.mark.anyio
async def test_publishing_during_execution_cannot_change_run_snapshot(tmp_path: Path) -> None:
    store = SkillStore(tmp_path / "evo")
    original = store.publish(snapshot(), expected="empty", reviewed=True)
    changed = snapshot("NEW guidance")

    async def verifier(cwd: Path) -> VerifierFeedback:
        store.publish(changed, expected=original, reviewed=True)
        return VerifierFeedback(verifier="test", status="passed")

    provider = fake()
    record = await run_task(
        store=store,
        provider=provider,
        model="fake",
        provider_name="fake",
        task_id="frozen",
        prompt="python task",
        cwd=tmp_path,
        tools=[],
        verifier=verifier,
    )
    assert record.selection.snapshot == original
    assert record.selection.skills[0].body == "Check boundary cases."
    assert store.snapshot().version == changed.version
    assert "NEW guidance" not in provider.calls[0][1]


@pytest.mark.anyio
@pytest.mark.parametrize(
    "prompt,source", [("/skill:missing", "interactive"), ("python", "extension")]
)
async def test_extension_leaves_explicit_and_extension_input_untouched(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prompt: str,
    source: str,
) -> None:
    # Exercise the actual registered handler while asserting it never touches storage.
    from unittest.mock import MagicMock

    from tau_coding.extensions import ExtensionContext, InputEvent
    from tau_coding.extensions.evo_harness.extension import setup

    api = MagicMock()
    setup(api)
    handler = next(call.args[1] for call in api.on.call_args_list if call.args[0] == "input")
    context = MagicMock(spec=ExtensionContext)
    context.is_running = False
    outcome = await handler(InputEvent(text=prompt, source=source), context)
    assert outcome is None
    api.append_entry.assert_not_called()


@pytest.mark.anyio
async def test_unknown_verifier_and_spawn_error_are_not_success(tmp_path: Path) -> None:
    result = await verify_command([str(tmp_path / "does-not-exist")], tmp_path)
    assert result.status == "unknown" and result.reason == "spawn_error"
    record = await run_task(
        store=SkillStore(tmp_path / "evo"),
        provider=fake(),
        model="fake",
        provider_name="fake",
        task_id="unverified",
        prompt="task",
        cwd=tmp_path,
        tools=[],
    )
    assert record.status == "completed" and record.feedback is None


@pytest.mark.anyio
async def test_interrupted_tool_retains_completed_response_usage(tmp_path: Path) -> None:
    from tau_agent.messages import Usage
    from tau_agent.tools import AgentTool, AgentToolResult

    started = asyncio.Event()

    async def wait_tool(*args: object, **kwargs: object) -> AgentToolResult:
        started.set()
        await asyncio.sleep(30)
        return AgentToolResult(content="done")

    provider = fake(
        AssistantMessage(
            content=[ToolCall(id="wait", name="wait", arguments={})],
            stop_reason="toolUse",
            usage=Usage(total_tokens=17),
        )
    )
    store = SkillStore(tmp_path / "evo")
    task = asyncio.create_task(
        run_task(
            store=store,
            provider=provider,
            model="fake",
            provider_name="fake",
            task_id="partial-usage",
            prompt="task",
            cwd=tmp_path,
            tools=[
                AgentTool(
                    name="wait",
                    label="Wait",
                    description="Wait",
                    parameters={},
                    execute_fn=wait_tool,
                )
            ],
        )
    )
    await asyncio.wait_for(started.wait(), 10)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    record = json.loads(next((store.root / "runs").glob("*/run.json")).read_text())
    assert record["status"] == "cancelled"
    assert record["usage"]["totalTokens"] == 17
