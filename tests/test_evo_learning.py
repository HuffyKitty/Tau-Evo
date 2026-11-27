"""Deterministic tests for official-score interpretation and online skill learning."""

import json
from pathlib import Path
from typing import Any

import pytest

from pi_event_helpers import assistant_done, assistant_start
from tau_agent.messages import AssistantMessage
from tau_ai.fake import FakeProvider
from tau_coding.evo.evidence import redact
from tau_coding.evo.experiment import (
    adjudicate_timeout,
    read_experiment,
    regrade,
    rescore,
    run_stream,
    verify_timeout_evidence,
)
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


def test_merge_accepts_existing_lineage_but_requires_new_failed_evidence() -> None:
    old = lesson().model_copy(update={"evidence": ("run:failed-run", "official:old")})
    add = Decision(
        action="ADD",
        id="procedure",
        scope="general",
        body="Old procedure",
        run_ids=(old.run_id,),
    )
    current = apply_compilation(SkillSnapshot(), Compilation(decisions=(add,)), (old,))
    new = old.model_copy(update={"run_id": "new-run", "evidence": ("run:new-run", "official:new")})
    merge = add.model_copy(
        update={
            "action": "MERGE",
            "body": "Updated procedure",
            "run_ids": (old.run_id, new.run_id),
        }
    )
    updated = apply_compilation(current, Compilation(decisions=(merge,)), (new,))
    assert updated.skills[0].revision == 2
    assert updated.skills[0].evidence == (*old.evidence, *new.evidence)
    for refs in (("future-run", new.run_id), (old.run_id,)):
        with pytest.raises(ValueError, match="unknown failed run or no current candidate"):
            apply_compilation(
                current,
                Compilation(decisions=(merge.model_copy(update={"run_ids": refs}),)),
                (new,),
            )


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
@pytest.mark.parametrize("driver_timeout", [False, True])
async def test_official_grader_uses_linux_cli_and_reads_release_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    driver_timeout: bool,
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
            if driver_timeout:
                raise TimeoutError("Driver stuck after writing official results")
        return 0, ""

    monkeypatch.setattr("tau_coding.evo.swe.command", docker)
    feedback = await official_score({"instance_id": "task"}, "patch\n", tmp_path)
    assert feedback.status == "passed"
    assert (tmp_path / "feedback.json").exists()
    assert calls[-1][:3] == ["docker", "rm", "-f"]
    assert "\r\n" not in (tmp_path / "prediction.jsonl").read_bytes().decode()


def test_timeout_diagnosis_requires_official_evidence_and_passing_control(tmp_path: Path) -> None:
    text = json.dumps(
        {
            "failure_reasons": {"task": "tests_timed_out"},
            "ambiguous_failure_ids": ["task"],
        }
    )
    atomic_write(tmp_path / "tau-evo.evo-test.json", text)
    candidate = VerifierFeedback(
        verifier="official", status="unknown", evidence=("official:" + digest(text),)
    )
    verify_timeout_evidence(tmp_path, "task", candidate)
    for status in ("failed", "unknown"):
        control = VerifierFeedback(verifier="official", status=status)
        assert adjudicate_timeout(candidate, control, 180) == candidate
    control = VerifierFeedback(verifier="official", status="passed", evidence=("control:hash",))
    feedback = adjudicate_timeout(candidate, control, 180)
    assert feedback.status == "failed" and feedback.reason == "candidate_tests_timeout"
    assert feedback.evidence == (*candidate.evidence, "control:hash")
    atomic_write(tmp_path / "tau-evo.evo-test.json", "{}")
    with pytest.raises(ValueError, match="official ambiguous"):
        verify_timeout_evidence(tmp_path, "task", candidate)


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


@pytest.mark.anyio
async def test_grading_calibration_is_uniform_and_preserves_solver_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tasks = [{"instance_id": "task", "image": "image"}]
    data = json.dumps(tasks)
    manifest = json.dumps(
        {
            "experiment_id": "test",
            "model": "fake",
            "task_ids": ["task"],
            "tasks_sha256": digest(data),
            "eval_timeout": 180,
        }
    )
    atomic_write(tmp_path / "tasks.json", data)
    atomic_write(tmp_path / "experiment.json", manifest)
    for arm in ("frozen", "evolving"):
        destination = tmp_path / arm / "tasks/task"
        atomic_write(
            destination / "result.json", make_run("task", SkillSnapshot()).model_dump_json()
        )
        atomic_write(destination / "image.json", json.dumps({"image": "pinned"}))
        atomic_write(destination / "model.patch", "original patch")
    calls = []

    async def image(value: str) -> tuple[str, bool]:
        return value, False

    async def score(
        task: dict[str, Any], patch: str, *args: Any, **kwargs: Any
    ) -> VerifierFeedback:
        assert patch == "original patch" and kwargs["timeout"] == 1800
        calls.append(task["instance_id"])
        return VerifierFeedback(verifier="official", status="passed")

    monkeypatch.setattr("tau_coding.evo.experiment.ensure_image", image)
    monkeypatch.setattr("tau_coding.evo.experiment.official_score", score)
    await regrade(tmp_path, 1800)
    assert len(calls) == 2
    await regrade(tmp_path, 1800)
    assert len(calls) == 2
    assert (tmp_path / "experiment.json").read_text("utf-8") == manifest
    config, _ = read_experiment(tmp_path)
    assert config["eval_timeout"] == 1800 and config["original_eval_timeout"] == 180
    (tmp_path / "grading-complete.json").unlink()
    with pytest.raises(ValueError, match="uniform regrading"):
        await run_stream(tmp_path, "evolving", FakeProvider([]))
