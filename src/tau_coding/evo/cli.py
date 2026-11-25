"""Small noninteractive CLI for reproducible Evo foundation experiments."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import tempfile
from pathlib import Path

from tau_ai.env import OpenAICompatibleConfig
from tau_ai.openai_compatible import OpenAICompatibleProvider
from tau_coding.evo.models import SkillRevision, SkillSnapshot, TaskRun
from tau_coding.evo.runner import run_task
from tau_coding.evo.store import SkillStore
from tau_coding.evo.verifier import verify_command
from tau_coding.tools import create_coding_tools

DEEPSEEK_URL = "https://api.deepseek.com"


def read_api_key(path: Path | None) -> str:
    raw = path.read_text(encoding="utf-8-sig") if path else os.environ.pop("DEEPSEEK_API_KEY", "")
    keys: set[str] = set(re.findall(r"sk-[A-Za-z0-9_-]+", raw))
    if len(keys) != 1:
        raise ValueError("Provide exactly one DeepSeek key via --api-key-file or DEEPSEEK_API_KEY")
    return keys.pop()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tau-Evo: versioned skills and verified tasks")
    parser.add_argument("--store", type=Path, default=Path(".tau-evo"))
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status")
    publish = commands.add_parser("publish")
    publish.add_argument("snapshot", type=Path)
    publish.add_argument("--expected", required=True)
    publish.add_argument("--reviewed", action="store_true")
    rollback = commands.add_parser("rollback")
    rollback.add_argument("version")
    rollback.add_argument("--expected", required=True)
    for name in ("run", "smoke"):
        run = commands.add_parser(name)
        run.add_argument("--model", required=True)
        run.add_argument("--api-key-file", type=Path)
        run.add_argument("--max-turns", type=int, default=12)
        run.add_argument("--timeout", type=float, default=180)
        if name == "run":
            run.add_argument("--cwd", type=Path, required=True)
            run.add_argument("--task-id", required=True)
            run.add_argument("--prompt-file", type=Path, required=True)
            run.add_argument("--allow-shell", action="store_true")
            run.add_argument("--max-bytes", type=int, default=8000)
            run.add_argument("--max-skills", type=int, default=3)
            run.add_argument("--verify", nargs=argparse.REMAINDER, required=True)
    return parser


async def _run(args: argparse.Namespace) -> TaskRun:
    key = read_api_key(args.api_key_file)
    provider = OpenAICompatibleProvider(
        OpenAICompatibleConfig(
            api_key=key,
            base_url=DEEPSEEK_URL,
            provider_name="deepseek",
            max_tokens=4096,
            max_retries=0,
            timeout_seconds=60,
            infer_api_from_model=False,
        )
    )
    try:
        if args.command == "smoke":
            return await _smoke(args, provider)
        if not args.verify:
            raise ValueError("--verify needs a command and must be the last option")
        store = SkillStore(args.store)
        cwd = args.cwd.resolve(strict=True)
        # Keep local credentials and audit storage outside the working task directory.
        if args.api_key_file and args.api_key_file.resolve().is_relative_to(cwd):
            raise ValueError("Move the task workspace outside the API key's directory")
        if store.root.is_relative_to(cwd):
            raise ValueError("Choose --store outside the task workspace")
        return await run_task(
            store=store,
            provider=provider,
            provider_name="deepseek",
            model=args.model,
            task_id=args.task_id,
            prompt=args.prompt_file.read_text(encoding="utf-8"),
            cwd=cwd,
            tools=[t for t in create_coding_tools(cwd=cwd) if args.allow_shell or t.name != "bash"],
            verifier=lambda root: verify_command(args.verify, root),
            max_turns=args.max_turns,
            timeout_seconds=args.timeout,
            max_bytes=args.max_bytes,
            max_skills=args.max_skills,
        )
    finally:
        await provider.aclose()


async def _smoke(
    args: argparse.Namespace,
    provider: OpenAICompatibleProvider,
) -> TaskRun:
    """Real tool-use smoke; fixture skills are human-authored, not learned."""
    with tempfile.TemporaryDirectory(prefix="tau-evo-smoke-") as directory:
        root = Path(directory)
        workspace = root / "task"
        workspace.mkdir()
        (workspace / "calculator.py").write_text(
            "def mean(values):\n    return sum(values) // len(values)\n",
            encoding="utf-8",
        )
        check = root / "verify.py"
        check.write_text(
            "import runpy\n"
            "mean = runpy.run_path('calculator.py')['mean']\n"
            "assert mean([1, 2]) == 1.5\n"
            "assert mean([-3, 0]) == -1.5\n"
            "assert mean([5]) == 5\n"
            "try:\n    mean([])\n"
            "except ValueError:\n    pass\n"
            "else:\n    raise AssertionError('empty input must raise ValueError')\n",
            encoding="utf-8",
        )
        # A separate smoke namespace never changes the user's active skill version.
        store = SkillStore(args.store / "smoke")
        snapshot = SkillSnapshot(
            skills=(
                SkillRevision(
                    id="boundary-checks",
                    revision=1,
                    scope="topic",
                    triggers=("mean",),
                    body=(
                        "Inspect the implementation first. "
                        "Handle fractional, negative, and empty inputs."
                    ),
                    author="smoke-fixture",
                ),
            )
        )
        store.publish(snapshot, expected=store.snapshot().version, reviewed=True)
        return await run_task(
            store=store,
            provider=provider,
            provider_name="deepseek",
            model=args.model,
            task_id="mean-boundary-smoke",
            prompt=(
                "Fix calculator.py so mean(values) returns the arithmetic mean, including "
                "fractional and negative results, and raises ValueError for an empty list. "
                "Read the file with tools, edit it, then give a short explanation."
            ),
            cwd=workspace,
            tools=[t for t in create_coding_tools(cwd=workspace) if t.name != "bash"],
            verifier=lambda cwd: verify_command([sys.executable, "-I", str(check)], cwd),
            max_turns=args.max_turns,
            timeout_seconds=args.timeout,
        )


def main() -> None:
    args = _parser().parse_args()
    try:
        store = SkillStore(args.store)
        if args.command == "status":
            snapshot = store.snapshot()
            print(json.dumps({"version": snapshot.version, "skills": len(snapshot.skills)}))
        elif args.command == "publish":
            snapshot = SkillSnapshot.model_validate_json(args.snapshot.read_text(encoding="utf-8"))
            print(store.publish(snapshot, expected=args.expected, reviewed=args.reviewed))
        elif args.command == "rollback":
            print(store.rollback(args.version, expected=args.expected))
        else:
            record = asyncio.run(_run(args))
            print(record.model_dump_json(indent=2))
            if (
                record.status != "completed"
                or record.feedback is None
                or record.feedback.status != "passed"
                or (args.command == "smoke" and record.tool_calls == 0)
            ):
                raise SystemExit(1)
    except (OSError, ValueError, RuntimeError) as exc:
        # Never print provider bodies, key-file contents, or Pydantic input values.
        print(
            f"Evo operation failed ({type(exc).__name__}); check arguments and local state.",
            file=sys.stderr,
        )
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
