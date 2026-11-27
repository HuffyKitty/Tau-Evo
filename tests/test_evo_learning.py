"""Deterministic tests for official-score interpretation and online skill learning."""

import json
from pathlib import Path
from typing import Any

import pytest

from pi_event_helpers import assistant_done, assistant_start
from tau_agent.messages import AssistantMessage
from tau_ai.fake import FakeProvider
from tau_coding.evo.evidence import redact
from tau_coding.evo.experiment import read_experiment, rescore, run_stream
from tau_coding.evo.learning import Compilation, Decision, apply_compilation, ask_json, reflect
from tau_coding.evo.models import (
    CandidateLesson,
    SkillSnapshot,
    TaskRun,
    VerifierFeedback,
    digest,
)
from tau_coding.evo.selection import select_skills
from tau_coding.evo.store import SkillStore, atomic_write
from tau_coding.evo.swe import interpret_report, official_score


def lesson() -> CandidateLesson:
    return CandidateLesson(
        run_id="failed-run",
        lesson="Inspect symbolic assumptions before simplifying.",
        trigger="sympy",
        scope_hint="topic",
        evidence=("official:test_failed",),
    )


def test_compiler_grounding_merge_budget_and_production_boundary(tmp_path: Path) -> None:
    add = Decision(
        action="ADD",
        id="assumptions",
        triggers=("sympy",),
        body=lesson().lesson,
        run_ids=("failed-run",),
    )
    snapshot = apply_compilation(SkillSnapshot(), Compilation(decisions=(add,)), (lesson(),))
    assert snapshot.skills[0].evidence == ("official:test_failed",)
    assert select_skills(snapshot, "Repository: sympy/sympy").skills
    merge = add.model_copy(update={"action": "MERGE", "body": "Updated procedure"})
    updated = apply_compilation(snapshot, Compilation(decisions=(merge,)), (lesson(),))
    assert updated.skills[0].revision == 2
    with pytest.raises(ValueError, match="unknown"):
        apply_compilation(
            snapshot,
            Compilation(decisions=(merge.model_copy(update={"run_ids": ("future",)}),)),
            (lesson(),),
        )
    with pytest.raises(ValueError, match="budget"):
        apply_compilation(SkillSnapshot(), Compilation(decisions=(add,)), (lesson(),), max_skills=0)
    with pytest.raises(ValueError, match="experimental"):
        SkillStore(tmp_path / "production").activate_experimental(
            snapshot,
            expected="empty",
            experiment_id="test",
        )


def test_official_report_requires_authoritative_evidence() -> None:
    report = {"task": {"resolved": True, "tests_status": {"FAIL_TO_PASS": {"success": ["x"]}}}}
    feedback = interpret_report("task", {}, report, "hash")
    assert feedback.status == "passed"
    assert "FAIL_TO_PASS" in feedback.summary
    assert feedback.evidence
    assert interpret_report("task", {"error_ids": ["task"]}, report, "hash").status == "unknown"
    assert interpret_report("task", {}, None, "hash").status == "unknown"
    assert interpret_report("task", {"empty_patch_ids": ["task"]}, None, "hash").status == "failed"


def make_run(task_id: str, snapshot: SkillSnapshot, status: str = "failed") -> TaskRun:
    return TaskRun(
        run_id=task_id,
        task_id=task_id,
        prompt_sha256="p",
        system_sha256="s",
        tools_sha256="t",
        model="fake",
        provider="fake",
        solver_version="test",
        selection=select_skills(snapshot, "sympy"),
        started_at="2026-01-01T00:00:00+00:00",
        finished_at="2026-01-01T00:00:01+00:00",
        status="completed",
        max_turns=2,
        timeout_seconds=10,
        feedback=VerifierFeedback(
            verifier="official",
            status=status,
            summary="assertion failed",
            evidence=("official:test",),
        ),
    )


@pytest.mark.anyio
async def test_success_and_unknown_never_trigger_reflection(tmp_path: Path) -> None:
    provider = FakeProvider([])
    for verdict in ("passed", "unknown"):
        result = await reflect(
            make_run("task", SkillSnapshot(), verdict),
            provider=provider,
            model="fake",
            store_root=tmp_path,
            problem="problem",
            repo="sympy",
            output=tmp_path / "reflection.json",
        )
        assert result is None
    assert provider.calls == []


@pytest.mark.anyio
async def test_failed_task_generates_skill_used_only_in_next_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tasks = [
        {"instance_id": f"task-{i}", "repo": "sympy", "problem_statement": f"issue-{i}"}
        for i in range(4)
    ]
    text = json.dumps(tasks)
    atomic_write(tmp_path / "tasks.json", text)
    atomic_write(
        tmp_path / "experiment.json",
        json.dumps(
            {
                "experiment_id": "test",
                "model": "fake",
                "batch_size": 2,
                "task_ids": [t["instance_id"] for t in tasks],
                "tasks_sha256": digest(text),
            }
        ),
    )
    seen = []

    async def solve(task: dict[str, Any], **kwargs: Any) -> TaskRun:
        snapshot = kwargs["snapshot"]
        seen.append(snapshot)
        record = make_run(
            task["instance_id"], snapshot, "failed" if task["instance_id"] == "task-0" else "passed"
        )
        store = kwargs["store"]
        atomic_write(store.root / "runs" / record.run_id / "session.jsonl", "bounded trajectory")
        atomic_write(kwargs["destination"] / "result.json", record.model_dump_json())
        return record

    monkeypatch.setattr("tau_coding.evo.experiment.solve_task", solve)
    responses = [
        {
            "lesson": "Check assumptions before simplification",
            "trigger": "sympy",
            "scope_hint": "topic",
        },
        {
            "decisions": [
                {
                    "action": "ADD",
                    "id": "assumptions",
                    "scope": "topic",
                    "triggers": ["sympy"],
                    "body": "Check assumptions before simplification",
                    "run_ids": ["task-0"],
                }
            ]
        },
    ]
    provider = FakeProvider(
        [
            [assistant_start(), assistant_done(AssistantMessage(content=json.dumps(value)))]
            for value in responses
        ]
    )
    summary = await run_stream(tmp_path, "evolving", provider)
    assert seen[0].version == seen[1].version == "empty"
    assert seen[2].version == seen[3].version != "empty"
    assert select_skills(seen[2], "sympy").skills[0].id == "assumptions"
    assert summary["arms"]["evolving"]["passed"] == 3
    assert "issue-2" not in provider.calls[0][2][0].text
    assert "issue-3" not in provider.calls[1][2][0].text
    # A completed stream resumes without paying for a model call or rerunning a task.
    await run_stream(tmp_path, "evolving", FakeProvider([]))
    assert len(seen) == 4


def test_redaction_and_manifest_integrity(tmp_path: Path) -> None:
    assert "sk-secret123" not in redact("x sk-secret123 y")
    atomic_write(tmp_path / "experiment.json", json.dumps({"tasks_sha256": "wrong"}))
    atomic_write(tmp_path / "tasks.json", "[]")
    with pytest.raises(ValueError, match="modified"):
        read_experiment(tmp_path)


@pytest.mark.anyio
async def test_official_grader_uses_linux_cli_and_reads_release_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    async def docker(argv: list[str], **kwargs: Any) -> tuple[int, str]:
        calls.append(argv)
        if argv[1] == "run":
            assert "swebench.harness.run_evaluation" in argv
            assert "tau-evo-grader:5.0.2" in argv
            run_id = argv[argv.index("--run_id") + 1]
            atomic_write(
                tmp_path / f"tau-evo.{run_id}.json",
                json.dumps({"resolved_ids": ["task"]}),
            )
            atomic_write(
                tmp_path / "logs" / "run_evaluation" / run_id / "tau-evo/task/report.json",
                json.dumps({"task": {"resolved": True, "tests_status": {}}}),
            )
        return 0, ""

    monkeypatch.setattr("tau_coding.evo.swe.command", docker)
    feedback = await official_score({"instance_id": "task"}, "patch\n", tmp_path)
    assert feedback.status == "passed"
    assert (tmp_path / "feedback.json").exists()
    assert calls[-1][:3] == ["docker", "rm", "-f"]
    assert "\r\n" not in (tmp_path / "prediction.jsonl").read_bytes().decode()


@pytest.mark.anyio
async def test_rescore_reuses_patch_and_never_reexecutes_solver(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tasks = [{"instance_id": "task", "image": "tag"}]
    data = json.dumps(tasks)
    atomic_write(tmp_path / "tasks.json", data)
    atomic_write(
        tmp_path / "experiment.json",
        json.dumps(
            {
                "experiment_id": "test",
                "tasks_sha256": digest(data),
                "task_ids": ["task"],
                "model": "fake",
                "eval_timeout": 1,
            }
        ),
    )
    destination = tmp_path / "frozen/tasks/task"
    run = make_run("task", SkillSnapshot(), "unknown")
    atomic_write(destination / "result.json", run.model_dump_json())
    atomic_write(destination / "model.patch", "original patch")
    atomic_write(destination / "image.json", json.dumps({"image": "pinned-image"}))

    async def image(value: str) -> tuple[str, bool]:
        assert value == "pinned-image"
        return value, False

    async def score(task: dict[str, Any], patch: str, *_: Any, **kwargs: Any) -> VerifierFeedback:
        assert patch == "original patch" and task["image"] == "pinned-image"
        return VerifierFeedback(verifier="official", status="passed")

    monkeypatch.setattr("tau_coding.evo.experiment.ensure_image", image)
    monkeypatch.setattr("tau_coding.evo.experiment.official_score", score)
    await rescore(tmp_path, "frozen", "task")
    updated = TaskRun.model_validate_json((destination / "result.json").read_text("utf-8"))
    assert updated.run_id == run.run_id
    assert updated.feedback.status == "passed"
    with pytest.raises(ValueError, match="Only unknown"):
        await rescore(tmp_path, "frozen", "task")
    assert not (tmp_path / "stream.lock").exists()


@pytest.mark.anyio
async def test_protocol_refuses_changed_solver_or_grader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tau_coding.evo.provenance import freeze_protocol

    async def docker(*_: Any, **kwargs: Any) -> tuple[int, str]:
        return 0, "image-v1\n"

    atomic_write(tmp_path / "experiment.json", "{}")
    monkeypatch.setattr("tau_coding.evo.provenance.command", docker)
    monkeypatch.setattr("tau_coding.evo.provenance.solver_digest", lambda: "solver-v1")
    await freeze_protocol(tmp_path)
    await freeze_protocol(tmp_path)
    monkeypatch.setattr("tau_coding.evo.provenance.solver_digest", lambda: "solver-v2")
    with pytest.raises(ValueError, match="changed"):
        await freeze_protocol(tmp_path)


@pytest.mark.anyio
async def test_learning_checkpoint_prevents_unaccounted_model_retries(tmp_path: Path) -> None:
    provider = FakeProvider(
        [
            [
                assistant_start(),
                assistant_done(AssistantMessage(content='{"lesson": "procedure"}')),
            ],
            [assistant_start(), assistant_done(AssistantMessage(content="malformed JSON"))],
        ]
    )
    path = tmp_path / "reflection.json"
    expected = await ask_json(provider, "fake", "system", "prompt", path)
    assert await ask_json(provider, "fake", "system", "prompt", path) == expected
    assert len(provider.calls) == 1
    failed = tmp_path / "failed.json"
    with pytest.raises(ValueError):
        await ask_json(provider, "fake", "system", "prompt", failed)
    with pytest.raises(ValueError, match="Previous learning call failed"):
        await ask_json(provider, "fake", "system", "prompt", failed)
    assert len(provider.calls) == 2
    assert json.loads(failed.read_text("utf-8"))["response"] == "malformed JSON"
