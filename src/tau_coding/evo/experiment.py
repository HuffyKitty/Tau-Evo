"""Sequential, checkpointed frozen-vs-evolving task streams with batch isolation."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from tau_agent.provider import ModelProvider
from tau_coding.evo.learning import compile_lessons, reflect
from tau_coding.evo.models import CandidateLesson, SkillSnapshot, TaskRun, VerifierFeedback, digest
from tau_coding.evo.runner import run_task
from tau_coding.evo.store import SkillStore, atomic_write
from tau_coding.evo.swe import TaskContainer, command, ensure_image, official_score


def read_experiment(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = json.loads((root / "experiment.json").read_text(encoding="utf-8"))
    text = (root / "tasks.json").read_text(encoding="utf-8")
    if digest(text) != config["tasks_sha256"]:
        raise ValueError("Frozen dataset was modified")
    tasks: list[dict[str, Any]] = json.loads(text)
    if [task["instance_id"] for task in tasks] != config["task_ids"]:
        raise ValueError("Task order differs from the experiment manifest")
    policy_path = root / "grading-policy.json"
    if policy_path.exists():
        policy = json.loads(policy_path.read_text("utf-8"))
        if policy["manifest_sha256"] != digest((root / "experiment.json").read_text("utf-8")):
            raise ValueError("Grading policy belongs to a different experiment")
        config = dict(
            config,
            original_eval_timeout=config["eval_timeout"],
            eval_timeout=policy["eval_timeout"],
            grading_policy_sha256=digest(policy_path.read_text("utf-8")),
        )
    return config, tasks


async def solve_task(
    task: dict[str, Any],
    *,
    provider: ModelProvider,
    config: dict[str, Any],
    store: SkillStore,
    snapshot: SkillSnapshot,
    destination: Path,
) -> TaskRun:
    task_id = task["instance_id"]
    pinned, created = await ensure_image(task["image"])
    container = TaskContainer(pinned)
    destination.mkdir(parents=True, exist_ok=True)
    atomic_write(destination / "image.json", json.dumps({"image": pinned}))
    try:
        await container.start()
        print(f"  SOLVE {task_id}", flush=True)
        record = await run_task(
            store=store,
            provider=provider,
            provider_name="deepseek",
            model=config["model"],
            task_id=task_id,
            prompt=f"Repository: {task['repo']}\n\nFix this issue:\n{task['problem_statement']}",
            cwd=destination,
            tools=[container.tool()],
            max_turns=config["max_turns"],
            timeout_seconds=config["solver_timeout"],
            skill_snapshot=snapshot,
            system_context=(
                "All repository files are INSIDE the bash tool's isolated container at /testbed. "
                "Python uses the activated testbed environment. Use the bash tool to inspect and "
                "edit files and run focused tests. Fix the issue with a minimal production-code "
                "change. Do not modify tests to make them pass. Finish when your patch is ready. "
                "Solve from the issue and current repository code. Do not search for official "
                "solutions, gold patches, hidden tests, later upstream versions, git objects "
                "containing future fixes, or benchmark artifacts outside /testbed. "
                "No internet is available inside the task container. Reproduce the problem, "
                "implement a focused fix early, then test it. Avoid repeated speculative searches. "
                f"You have at most {config['max_turns']} model turns; leave time to edit and test."
            ),
        )
        # Score partial patches after solver timeout/turn-limit too; solver state is separate.
        patch = await container.patch()
        atomic_write(destination / "model.patch", patch)
        print(f"  SCORE {task_id} patch_bytes={len(patch.encode())}", flush=True)
        score_task = dict(task, image=pinned)
        feedback = await official_score(
            score_task,
            patch,
            destination / "official",
            timeout=config["eval_timeout"],
        )
        record = record.model_copy(update={"feedback": feedback})
        atomic_write(
            store.root / "runs" / record.run_id / "run.json", record.model_dump_json(indent=2)
        )
        atomic_write(destination / "result.json", record.model_dump_json(indent=2))
        print(f"  RESULT {task_id}: {feedback.status} ({feedback.reason})", flush=True)
        return record
    finally:
        await container.close()
        if created and config.get("release_images", True):
            # Only remove this task's newly pulled tag. Never prune shared/user images.
            await command(["docker", "image", "rm", task["image"]], timeout=60)


async def run_stream(
    root: Path,
    arm: Literal["frozen", "evolving"],
    provider: ModelProvider,
) -> dict[str, Any]:
    root = root.resolve()
    config, tasks = read_experiment(root)
    if (root / "grading-policy.json").exists():
        complete = root / "grading-complete.json"
        if not complete.exists() or json.loads(complete.read_text("utf-8")) != json.loads(
            (root / "grading-policy.json").read_text("utf-8")
        ):
            raise ValueError("Complete uniform regrading before resuming either solver arm")
    arm_dir = root / arm
    store = SkillStore(arm_dir / "experimental-skills")
    arm_dir.mkdir(parents=True, exist_ok=True)
    # One stream writer. Do not silently replay an interrupted task with unknown side effects.
    lock = root / "stream.lock"
    with lock.open("x", encoding="utf-8") as handle:
        handle.write(config["experiment_id"])
    try:
        for start in range(0, len(tasks), config["batch_size"]):
            batch_index = start // config["batch_size"]
            batch_dir = arm_dir / "batches" / str(batch_index)
            batch_dir.mkdir(parents=True, exist_ok=True)
            snapshot_path = batch_dir / "snapshot.json"
            if snapshot_path.exists():
                snapshot = SkillSnapshot.model_validate_json(snapshot_path.read_text("utf-8"))
            else:
                snapshot = SkillSnapshot() if arm == "frozen" else store.snapshot()
                atomic_write(snapshot_path, snapshot.model_dump_json(indent=2))
            if (batch_dir / "complete.json").exists():
                continue
            lessons: list[CandidateLesson] = []
            for task in tasks[start : start + config["batch_size"]]:
                destination = arm_dir / "tasks" / task["instance_id"]
                result_path = destination / "result.json"
                if result_path.exists():
                    run = TaskRun.model_validate_json(result_path.read_text("utf-8"))
                    if run.selection.snapshot != snapshot.version:
                        raise ValueError("Checkpoint has the wrong skill snapshot")
                else:
                    if destination.exists():
                        raise RuntimeError(
                            f"Incomplete task {task['instance_id']}; inspect artifacts before retry"
                        )
                    # Reuse the other arm's exact image digest, not a mutable registry tag.
                    other_arm = "frozen" if arm == "evolving" else "evolving"
                    pin = root / other_arm / "tasks" / task["instance_id"] / "image.json"
                    solver_task = task
                    if pin.exists():
                        solver_task = dict(task, image=json.loads(pin.read_text("utf-8"))["image"])
                    run = await solve_task(
                        solver_task,
                        provider=provider,
                        config=config,
                        store=store,
                        snapshot=snapshot,
                        destination=destination,
                    )
                if run.feedback is None or run.feedback.status == "unknown":
                    raise RuntimeError(f"Scoring infrastructure failed for {task['instance_id']}")
                if arm == "evolving" and start + config["batch_size"] < len(tasks):
                    lesson_path = destination / "lesson.json"
                    lesson: CandidateLesson | None
                    if lesson_path.exists():
                        lesson = CandidateLesson.model_validate_json(lesson_path.read_text("utf-8"))
                    else:
                        lesson = await reflect(
                            run,
                            provider=provider,
                            model=config["model"],
                            store_root=store.root,
                            problem=task["problem_statement"],
                            repo=task["repo"],
                            output=destination / "reflection.json",
                        )
                        if lesson is not None:
                            atomic_write(lesson_path, lesson.model_dump_json(indent=2))
                    if lesson is not None:
                        lessons.append(lesson)
            next_snapshot = snapshot
            if lessons:
                candidate = batch_dir / "candidate.json"
                if candidate.exists():
                    next_snapshot = SkillSnapshot.model_validate_json(candidate.read_text("utf-8"))
                else:
                    next_snapshot = await compile_lessons(
                        snapshot,
                        tuple(lessons),
                        provider=provider,
                        model=config["model"],
                        output=batch_dir / "compilation.json",
                    )
                    atomic_write(candidate, next_snapshot.model_dump_json(indent=2))
                active = store.snapshot().version
                # Restart after activation but before checkpoint write is idempotent.
                if active != next_snapshot.version:
                    store.activate_experimental(
                        next_snapshot,
                        expected=snapshot.version,
                        experiment_id=config["experiment_id"],
                    )
            atomic_write(
                batch_dir / "complete.json",
                json.dumps(
                    {
                        "input_version": snapshot.version,
                        "output_version": next_snapshot.version,
                        "source_runs": [lesson.run_id for lesson in lessons],
                    },
                    indent=2,
                ),
            )
            print(
                f"BATCH {batch_index}: {snapshot.version[:12]} -> {next_snapshot.version[:12]}",
                flush=True,
            )
            summarize(root)
    finally:
        lock.unlink(missing_ok=True)
    return summarize(root)


async def rescore(root: Path, arm: str, task_id: str, *, diagnose_timeout: bool = False) -> None:
    """Retry infrastructure grading only; never rerun the solver or spend model tokens."""
    root = root.resolve()
    config, tasks = read_experiment(root)
    task = next((item for item in tasks if item["instance_id"] == task_id), None)
    if task is None or arm not in {"frozen", "evolving"}:
        raise ValueError("Unknown task or arm")
    lock = root / "stream.lock"
    with lock.open("x", encoding="utf-8") as handle:
        handle.write("rescore")
    try:
        destination = root / arm / "tasks" / task_id
        result_path = destination / "result.json"
        run = TaskRun.model_validate_json(result_path.read_text("utf-8"))
        if run.feedback is not None and run.feedback.status != "unknown":
            raise ValueError("Only unknown infrastructure results can be rescored")
        if diagnose_timeout:
            verify_timeout_evidence(destination, task_id, run.feedback)
        image = json.loads((destination / "image.json").read_text("utf-8"))["image"]
        pinned, created = await ensure_image(image)
        try:
            feedback = await official_score(
                dict(task, image=pinned),
                task["patch"]
                if diagnose_timeout
                else (destination / "model.patch").read_text("utf-8"),
                destination
                / (("timeout-control-" if diagnose_timeout else "rescore-") + uuid4().hex[:8]),
                timeout=config["eval_timeout"],
            )
            if diagnose_timeout:
                assert run.feedback is not None
                feedback = adjudicate_timeout(run.feedback, feedback, config["eval_timeout"])
            run = run.model_copy(update={"feedback": feedback})
            atomic_write(result_path, run.model_dump_json(indent=2))
            atomic_write(
                root / arm / "experimental-skills" / "runs" / run.run_id / "run.json",
                run.model_dump_json(indent=2),
            )
            print(f"RESCORE {task_id}: {feedback.status}", flush=True)
        finally:
            if created and config.get("release_images", True):
                await command(["docker", "image", "rm", image], timeout=60)
    finally:
        lock.unlink(missing_ok=True)
    summarize(root)


def verify_timeout_evidence(
    destination: Path, task_id: str, feedback: VerifierFeedback | None
) -> None:
    if feedback is None or not feedback.evidence:
        raise ValueError("No official timeout evidence")
    expected = feedback.evidence[0].rsplit(":", 1)[-1]
    for path in destination.rglob("tau-evo.evo-*.json"):
        text = path.read_text("utf-8")
        if digest(text) == expected:
            report = json.loads(text)
            if (
                report.get("failure_reasons", {}).get(task_id) == "tests_timed_out"
                and task_id in report.get("ambiguous_failure_ids", [])
                and task_id not in report.get("infra_failure_ids", [])
            ):
                return
    raise ValueError("Only an official ambiguous test timeout can use this diagnosis")


def adjudicate_timeout(
    candidate: VerifierFeedback, control: VerifierFeedback, timeout: int
) -> VerifierFeedback:
    if control.status != "passed":
        return candidate
    return VerifierFeedback(
        verifier="swebench-5.0.2",
        status="failed",
        reason="candidate_tests_timeout",
        summary=(
            f"Candidate tests exceeded {timeout} seconds in the official harness. "
            "The official reference patch passed in the same pinned image and time limit. "
            "This is a diagnosed timeout, not an assertion-test verdict; both reports are retained."
        ),
        evidence=(*candidate.evidence, *control.evidence),
    )


async def regrade(root: Path, timeout: int) -> None:
    """Calibrate scoring uniformly before any learned batch has been activated."""
    root = root.resolve()
    config, tasks = read_experiment(root)
    if timeout < config["eval_timeout"]:
        raise ValueError("Calibration may only increase the grading time limit")
    if any((root / "evolving" / "batches").glob("*/complete.json")):
        raise ValueError("Start a new experiment to change grading after skill compilation")
    lock = root / "stream.lock"
    with lock.open("x", encoding="utf-8") as handle:
        handle.write("regrade")
    try:
        policy_path = root / "grading-policy.json"
        policy = {
            "manifest_sha256": digest((root / "experiment.json").read_text("utf-8")),
            "eval_timeout": timeout,
            "scope": "all existing saved patches and future tasks in both arms; solver unchanged",
        }
        if policy_path.exists() and json.loads(policy_path.read_text("utf-8")) != policy:
            raise ValueError("Grading policy is already frozen at another limit")
        atomic_write(policy_path, json.dumps(policy, indent=2))
        for arm in ("frozen", "evolving"):
            for task in tasks:
                destination = root / arm / "tasks" / task["instance_id"]
                result_path = destination / "result.json"
                if not result_path.exists():
                    continue
                checkpoint = destination / f"graded-{timeout}.json"
                if checkpoint.exists():
                    continue
                run = TaskRun.model_validate_json(result_path.read_text("utf-8"))
                image = json.loads((destination / "image.json").read_text("utf-8"))["image"]
                await ensure_image(image)
                feedback = await official_score(
                    dict(task, image=image),
                    (destination / "model.patch").read_text("utf-8"),
                    destination / f"regrade-{timeout}",
                    timeout=timeout,
                )
                if run.feedback is None or run.feedback.status != feedback.status:
                    lesson_path = destination / "lesson.json"
                    if lesson_path.exists():
                        lesson_path.rename(destination / "lesson-before-regrade.json")
                atomic_write(destination / "before-regrade.json", run.model_dump_json(indent=2))
                run = run.model_copy(update={"feedback": feedback})
                atomic_write(result_path, run.model_dump_json(indent=2))
                atomic_write(
                    root / arm / "experimental-skills" / "runs" / run.run_id / "run.json",
                    run.model_dump_json(indent=2),
                )
                print(f"REGRADE {arm} {task['instance_id']}: {feedback.status}", flush=True)
                if feedback.status == "unknown":
                    raise RuntimeError("Regrading is still unknown; inspect the official logs")
                atomic_write(checkpoint, json.dumps({"run_id": run.run_id, "timeout": timeout}))
        atomic_write(root / "grading-complete.json", json.dumps(policy))
    finally:
        lock.unlink(missing_ok=True)
    summarize(root)


def elapsed_seconds(run: TaskRun) -> float:
    if run.finished_at is None:
        return 0
    return (
        datetime.fromisoformat(run.finished_at) - datetime.fromisoformat(run.started_at)
    ).total_seconds()


def summarize(root: Path) -> dict[str, Any]:
    config, tasks = read_experiment(root)
    summary: dict[str, Any] = {
        "experiment_id": config["experiment_id"],
        "model": config["model"],
        "task_ids": config["task_ids"],
        "planned_per_arm": len(tasks),
        "scope": "single-repository small-sample pipeline validation; not RSI efficacy evidence",
        "arms": {},
    }
    for arm in ("frozen", "evolving"):
        records = []
        for task in tasks:
            path = root / arm / "tasks" / task["instance_id"] / "result.json"
            if path.exists():
                records.append(TaskRun.model_validate_json(path.read_text("utf-8")))
        passed = sum(r.feedback is not None and r.feedback.status == "passed" for r in records)
        learning_tokens = 0
        for filename in ("reflection.json", "compilation.json"):
            for path in (root / arm).rglob(filename):
                data = json.loads(path.read_text("utf-8"))
                learning_tokens += data.get("usage", {}).get("totalTokens", 0)
        summary["arms"][arm] = {
            "completed": len(records),
            "passed": passed,
            "pass_rate_of_planned": passed / len(tasks),
            "solver_tokens": sum(r.usage.total_tokens for r in records),
            "solver_seconds": round(sum(elapsed_seconds(r) for r in records), 2),
            "learning_tokens": learning_tokens,
            "tasks": [
                {
                    "id": r.task_id,
                    "status": r.status,
                    "stop_reason": r.error_type,
                    "verdict": r.feedback.status if r.feedback else "unknown",
                    "verdict_reason": r.feedback.reason if r.feedback else "missing",
                    "turns": r.turns,
                    "tool_calls": r.tool_calls,
                    "solver_seconds": round(elapsed_seconds(r), 2),
                    "tokens": r.usage.total_tokens,
                    "snapshot": r.selection.snapshot,
                    "selected_skills": [s.id for s in r.selection.skills],
                }
                for r in records
            ],
        }
    frozen = {task["id"]: task for task in summary["arms"]["frozen"]["tasks"]}
    evolving = {task["id"]: task for task in summary["arms"]["evolving"]["tasks"]}
    pairs = [
        task_id
        for task_id in config["task_ids"]
        if task_id in frozen
        and task_id in evolving
        and frozen[task_id]["verdict"] != "unknown"
        and evolving[task_id]["verdict"] != "unknown"
    ]
    summary["paired"] = {
        "count": len(pairs),
        "improvements": [
            t
            for t in pairs
            if frozen[t]["verdict"] == "failed" and evolving[t]["verdict"] == "passed"
        ],
        "regressions": [
            t
            for t in pairs
            if frozen[t]["verdict"] == "passed" and evolving[t]["verdict"] == "failed"
        ],
        "budget_note": "Equal solver limits; learning tokens are additional, reported separately",
    }
    atomic_write(root / "summary.json", json.dumps(summary, ensure_ascii=False, indent=2))
    return summary
