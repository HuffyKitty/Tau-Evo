"""SWE-bench 5.x dataset/container adapter; grading remains the official CLI."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4

from tau_agent.messages import TextContent
from tau_agent.tools import AgentTool, AgentToolResult, ToolCancellationToken, ToolUpdateCallback
from tau_agent.types import JSONValue
from tau_coding.evo.evidence import redact
from tau_coding.evo.models import VerifierFeedback, digest
from tau_coding.evo.store import atomic_write


async def command(
    argv: list[str],
    *,
    cwd: Path | None = None,
    timeout: float = 120,
    output_path: Path | None = None,
) -> tuple[int, str]:
    """Bound host subprocess lifetime and retained output, without shell interpolation."""
    with tempfile.TemporaryFile() as output:
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=cwd,
            stdout=output,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            await asyncio.wait_for(process.wait(), timeout)
        except (TimeoutError, asyncio.CancelledError):
            if process.returncode is None:
                process.kill()
                await process.wait()
            raise
        finally:
            output.seek(0)
            if output_path is not None:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with output_path.open("wb") as target:
                    while chunk := output.read(65536):
                        target.write(chunk)
        output.seek(0, os.SEEK_END)
        output.seek(max(0, output.tell() - 20000))
        text = output.read().decode("utf-8", errors="replace")
        return process.returncode or 0, text


class TaskContainer:
    """Solver has only an exec tool inside its container, no host tools or mounted secrets."""

    def __init__(self, image: str) -> None:
        self.image = image
        self.name = "tau-evo-solver-" + uuid4().hex[:16]

    async def start(self) -> None:
        code, output = await command(
            [
                "docker",
                "run",
                "-d",
                "--name",
                self.name,
                "--network",
                "none",
                "--cpus",
                "2",
                "--memory",
                "3g",
                "--pids-limit",
                "512",
                self.image,
                "sleep",
                "infinity",
            ]
        )
        if code:
            raise RuntimeError("Solver container failed: " + redact(output, 1000))

    async def close(self) -> None:
        await command(["docker", "rm", "-f", self.name], timeout=30)

    async def shell(self, script: str, seconds: int = 60) -> tuple[int, str]:
        return await command(
            [
                "docker",
                "exec",
                "-w",
                "/testbed",
                self.name,
                "timeout",
                "-k",
                "5",
                str(seconds),
                "/bin/bash",
                "-lc",
                "source /opt/miniconda3/bin/activate testbed && " + script,
            ],
            timeout=seconds + 15,
        )

    def tool(self) -> AgentTool:
        async def execute(
            tool_call_id: str,
            arguments: Mapping[str, JSONValue],
            signal: ToolCancellationToken | None = None,
            on_update: ToolUpdateCallback | None = None,
        ) -> AgentToolResult:
            if signal is not None and signal.is_cancelled():
                raise asyncio.CancelledError
            script = arguments.get("command")
            if not isinstance(script, str):
                raise ValueError("command must be text")
            try:
                code, output = await self.shell(script)
                return AgentToolResult(
                    content=[TextContent(text=f"Exit code: {code}\n{redact(output, 16000)}")]
                )
            except TimeoutError:
                return AgentToolResult(content=[TextContent(text="Command exceeded 60 seconds.")])

        return AgentTool(
            name="bash",
            label="Container shell",
            description="Run a shell command in the isolated task container at /testbed.",
            parameters={
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
                "additionalProperties": False,
            },
            execute_fn=execute,
            execution_mode="sequential",
            prompt_snippet="Inspect, edit and test repository files inside /testbed.",
        )

    async def patch(self) -> str:
        code, output = await self.shell("git add -N . && git diff --binary HEAD")
        if code:
            raise RuntimeError("Could not extract solver patch")
        # Large patches must not be silently truncated by the bounded tool-output path.
        destination = Path(tempfile.gettempdir()) / f"{self.name}.patch"
        try:
            code, _ = await self.shell("git diff --binary HEAD > /tmp/tau-model.patch")
            if code:
                raise RuntimeError("Patch export failed")
            code, _ = await command(
                [
                    "docker",
                    "cp",
                    self.name + ":/tmp/tau-model.patch",
                    str(destination),
                ]
            )
            if code:
                raise RuntimeError("Patch copy failed")
            if destination.stat().st_size > 2_000_000:
                raise ValueError("Patch exceeds 2 MB experiment limit")
            return destination.read_text(encoding="utf-8")
        finally:
            destination.unlink(missing_ok=True)


async def ensure_image(image: str) -> tuple[str, bool]:
    code, _ = await command(["docker", "image", "inspect", image])
    created = code != 0
    if created:
        code, output = await command(["docker", "pull", image], timeout=1200)
        if code:
            raise RuntimeError("Image pull failed: " + redact(output, 1000))
    code, output = await command(
        [
            "docker",
            "image",
            "inspect",
            "--format",
            "{{json .RepoDigests}}",
            image,
        ]
    )
    if code:
        raise RuntimeError("Cannot pin image digest")
    digests = json.loads(output)
    if not digests:
        raise ValueError("Benchmark image has no registry digest")
    return image if "@sha256:" in image else str(digests[0]), created


def interpret_report(
    task_id: str,
    summary: dict[str, Any],
    report: dict[str, Any] | None,
    evidence_hash: str,
) -> VerifierFeedback:
    evidence = (f"swebench:5.0.2:{task_id}:{evidence_hash}",)
    if task_id in summary.get("infra_failure_ids", []) or task_id in summary.get("error_ids", []):
        return VerifierFeedback(
            verifier="swebench-5.0.2",
            status="unknown",
            reason="harness_error",
            summary=redact(str(summary.get("failure_reasons", {}))),
            evidence=evidence,
        )
    if task_id in summary.get("empty_patch_ids", []):
        return VerifierFeedback(
            verifier="swebench-5.0.2",
            status="failed",
            reason="empty_patch",
            summary="Official harness recorded an empty model patch.",
            evidence=evidence,
        )
    if report is None or task_id not in report:
        return VerifierFeedback(
            verifier="swebench-5.0.2", status="unknown", reason="missing_report"
        )
    result = report[task_id]
    tests = {
        group: {"passed": len(status.get("success", [])), "failed": status.get("failure", [])}
        for group, status in result.get("tests_status", {}).items()
    }
    return VerifierFeedback(
        verifier="swebench-5.0.2",
        status="passed" if result["resolved"] else "failed",
        reason="resolved" if result["resolved"] else "tests_failed",
        summary=redact(json.dumps(tests, ensure_ascii=False), 6000),
        output_sha256=evidence_hash,
        evidence=evidence,
    )


async def official_score(
    task: dict[str, Any],
    patch: str,
    destination: Path,
    *,
    timeout: int = 180,
) -> VerifierFeedback:
    destination.mkdir(parents=True, exist_ok=True)
    dataset = destination / "task.json"
    prediction = destination / "prediction.jsonl"
    atomic_write(dataset, json.dumps([task], ensure_ascii=False))
    atomic_write(
        prediction,
        json.dumps(
            {
                "instance_id": task["instance_id"],
                "model_name_or_path": "tau-evo",
                "model_patch": patch,
            }
        ),
    )
    run_id = "evo-" + uuid4().hex[:16]
    grader_name = "tau-evo-grader-" + run_id
    try:
        code, _ = await command(
            [
                "docker",
                "run",
                "--rm",
                "--name",
                grader_name,
                "--mount",
                f"type=bind,source={destination.resolve()},target=/workspace",
                "--mount",
                "type=bind,source=/var/run/docker.sock,target=/var/run/docker.sock",
                "tau-evo-grader:5.0.2",
                "python",
                "-m",
                "swebench.harness.run_evaluation",
                "--dataset_name",
                "/workspace/task.json",
                "--predictions_path",
                "/workspace/prediction.jsonl",
                "--max_workers",
                "1",
                "--run_id",
                run_id,
                "--timeout",
                str(timeout),
            ],
            cwd=destination,
            timeout=timeout + 180,
            output_path=destination / "harness.log",
        )
        root = destination / "logs" / "run_evaluation" / run_id
        summary_path = destination / f"tau-evo.{run_id}.json"
        report_path = root / "tau-evo" / task["instance_id"] / "report.json"
        if code or not summary_path.exists():
            return VerifierFeedback(
                verifier="swebench-5.0.2",
                status="unknown",
                reason="harness_process_error",
                summary="Inspect host-only harness.log; infrastructure errors are excluded.",
            )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        report_text = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
        report = json.loads(report_text) if report_text else None
        feedback = interpret_report(
            task["instance_id"],
            summary,
            report,
            digest(report_text or summary_path.read_text(encoding="utf-8")),
        )
        atomic_write(destination / "feedback.json", feedback.model_dump_json(indent=2))
        return feedback
    except TimeoutError:
        return VerifierFeedback(
            verifier="swebench-5.0.2",
            status="unknown",
            reason="harness_timeout",
        )
    finally:
        # Only our uniquely named scorer container; never touch unrelated Docker resources.
        await command(
            [
                "docker",
                "rm",
                "-f",
                grader_name,
                f"sweb.eval.{task['instance_id'].lower()}.{run_id}",
            ],
            timeout=30,
        )
