"""Validate a completed paired experiment and export a reviewed-summary candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tau_coding.evo.experiment import read_experiment, summarize
from tau_coding.evo.models import SkillSnapshot, TaskRun, digest
from tau_coding.evo.store import atomic_write


def export(root: Path, output: Path) -> dict[str, Any]:
    config, tasks = read_experiment(root)
    records: dict[str, list[TaskRun]] = {}
    images: dict[str, list[str]] = {}
    reports: list[dict[str, str]] = []
    for arm in ("frozen", "evolving"):
        records[arm], images[arm] = [], []
        for task in tasks:
            destination = root / arm / "tasks" / task["instance_id"]
            run = TaskRun.model_validate_json((destination / "result.json").read_text("utf-8"))
            if run.feedback is None or run.feedback.status == "unknown":
                raise ValueError("All tasks must have authoritative pass/fail feedback")
            if run.model != config["model"] or (run.max_turns, run.timeout_seconds) != (
                config["max_turns"],
                config["solver_timeout"],
            ):
                raise ValueError("Solver model/budget differs from the manifest")
            batch_index = len(records[arm]) // config["batch_size"]
            batch_dir = root / arm / "batches" / str(batch_index)
            snapshot = SkillSnapshot.model_validate_json(
                (batch_dir / "snapshot.json").read_text("utf-8")
            )
            if not (batch_dir / "complete.json").exists():
                raise ValueError("Batch has not completed")
            if run.selection.snapshot != snapshot.version:
                raise ValueError("Task did not use the batch's frozen snapshot")
            if any(skill not in snapshot.skills for skill in run.selection.skills):
                raise ValueError("Selected skill contents differ from the frozen library")
            system = (
                root / arm / "experimental-skills" / "runs" / run.run_id / "system.txt"
            ).read_text("utf-8")
            if digest(system) != run.system_sha256:
                raise ValueError("Stored system prompt does not match the actual run hash")
            if any(skill.body.strip() not in system for skill in run.selection.skills):
                raise ValueError("A selected skill was not injected into the system prompt")
            if arm == "frozen" and run.selection.snapshot != "empty":
                raise ValueError("Frozen arm received skills")
            records[arm].append(run)
            images[arm].append(json.loads((destination / "image.json").read_text("utf-8"))["image"])
            # Hash the actual official report, or its aggregate for an empty-patch verdict.
            candidates = list(destination.rglob("report.json")) + list(
                destination.rglob("tau-evo.evo-*.json")
            )
            hashes = {digest(path.read_text("utf-8")) for path in candidates}
            drivers = [
                path
                for path in destination.rglob("driver.json")
                if json.loads(path.read_text("utf-8")).get("eval_timeout") == config["eval_timeout"]
            ]
            if not drivers:
                raise ValueError("No scorer record using the common evaluation time limit")
            for evidence in run.feedback.evidence:
                expected = evidence.rsplit(":", 1)[-1]
                if expected not in hashes:
                    raise ValueError("Official report hash does not match feedback evidence")
                if not any(
                    driver.parent in path.parents and digest(path.read_text("utf-8")) == expected
                    for driver in drivers
                    for path in candidates
                ):
                    raise ValueError("Verdict did not come from the common grading policy")
                reports.append({"arm": arm, "task_id": run.task_id, "report_sha256": expected})
        if [r.started_at for r in records[arm]] != sorted(r.started_at for r in records[arm]):
            raise ValueError("Task execution order differs from the frozen order")
    if images["frozen"] != images["evolving"]:
        raise ValueError("Task images differed across arms")
    for frozen, evolving in zip(records["frozen"], records["evolving"], strict=True):
        if (frozen.prompt_sha256, frozen.tools_sha256, frozen.provider) != (
            evolving.prompt_sha256,
            evolving.tools_sha256,
            evolving.provider,
        ):
            raise ValueError("Problem, tool schema or provider differed across arms")
    positions = {r.run_id: i for i, r in enumerate(records["evolving"])}
    learning_links = []
    for index, run in enumerate(records["evolving"]):
        for skill in run.selection.skills:
            source_ids = [ref[4:] for ref in skill.evidence if ref.startswith("run:")]
            if not source_ids:
                raise ValueError("Selected skill has no source run")
            sources = []
            for source in source_ids:
                position = positions[source]
                source_run = records["evolving"][position]
                if position >= (index // config["batch_size"]) * config["batch_size"]:
                    raise ValueError("Future/same-batch evidence leaked into skill selection")
                if source_run.feedback is None or source_run.feedback.status != "failed":
                    raise ValueError("Skill source was not an externally failed task")
                sources.append(source_run.task_id)
            learning_links.append(
                {
                    "task_id": run.task_id,
                    "skill_id": skill.id,
                    "revision": skill.revision,
                    "source_failed_tasks": sources,
                    "body_sha256": digest(skill.body),
                }
            )
    if not learning_links:
        raise ValueError("No evidence that a learned skill reached a subsequent batch")
    result = {
        "manifest": json.loads((root / "experiment.json").read_text("utf-8")),
        "effective_eval_timeout": config["eval_timeout"],
        "grading_policy": (
            json.loads((root / "grading-policy.json").read_text("utf-8"))
            if (root / "grading-policy.json").exists()
            else None
        ),
        "protocol": json.loads((root / "protocol.json").read_text("utf-8")),
        "summary": summarize(root),
        "verified": [
            "same_model_provider_problem_order_and_solver_limits",
            "same_tool_schema_and_task_image_digests",
            "frozen_empty_skills_and_per_batch_snapshot_isolation",
            "selected_skill_contents_present_in_hashed_system_prompts",
            "official_feedback_matches_retained_report_hashes",
            "selected_skills_cite_only_failed_runs_from_earlier_batches",
        ],
        "images": dict(zip(config["task_ids"], images["frozen"], strict=True)),
        "official_reports": reports,
        "learning_links": learning_links,
    }
    atomic_write(output, json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    exported = export(args.experiment, args.output)
    print(f"Verified {exported['summary']['paired']['count']} pairs; summary exported for review")
