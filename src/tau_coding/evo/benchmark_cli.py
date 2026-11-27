"""Opt-in SWE-bench experiments. Install with uv sync --extra benchmark."""

from __future__ import annotations

import argparse
import asyncio
import io
import json
from importlib import import_module
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx

from tau_ai.env import OpenAICompatibleConfig
from tau_ai.openai_compatible import OpenAICompatibleProvider
from tau_coding.evo.cli import DEEPSEEK_URL, read_api_key
from tau_coding.evo.experiment import read_experiment, regrade, rescore, run_stream, summarize
from tau_coding.evo.models import digest
from tau_coding.evo.provenance import freeze_protocol
from tau_coding.evo.store import atomic_write
from tau_coding.evo.swe import ensure_image, official_score

DATASET = "SWE-bench/SWE-bench_Lite"


def prepare(args: argparse.Namespace) -> None:
    root: Path = args.experiment.resolve()
    if (root / "experiment.json").exists():
        raise ValueError("Experiment already exists; its task order and budgets are frozen")
    with httpx.Client(timeout=120, follow_redirects=True) as client:
        revision = args.revision
        if revision is None:
            info = client.get(f"https://huggingface.co/api/datasets/{DATASET}")
            info.raise_for_status()
            revision = info.json()["sha"]
        if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
            raise ValueError("Dataset revision must be a full commit SHA")
        response = client.get(
            f"https://huggingface.co/datasets/{DATASET}/resolve/{revision}/"
            "data/test-00000-of-00001.parquet"
        )
        response.raise_for_status()
    parquet = import_module("pyarrow.parquet")
    rows: list[dict[str, Any]] = parquet.read_table(io.BytesIO(response.content)).to_pylist()
    tasks = sorted(
        [row for row in rows if row["repo"] == args.repo],
        key=lambda row: row["instance_id"],
    )[: args.limit]
    if len(tasks) != args.limit or args.limit < 2:
        raise ValueError("Requested sample size is not available in the selected repository")
    text = json.dumps(tasks, ensure_ascii=False, sort_keys=True)
    atomic_write(root / "tasks.json", text)
    atomic_write(
        root / "experiment.json",
        json.dumps(
            {
                "experiment_id": uuid4().hex,
                "dataset": DATASET,
                "dataset_revision": revision,
                "tasks_sha256": digest(text),
                "task_ids": [r["instance_id"] for r in tasks],
                "selection": "lexicographic instance_id within one repo, before outcomes",
                "repo": args.repo,
                "model": args.model,
                "batch_size": args.batch_size,
                "max_turns": args.max_turns,
                "solver_timeout": args.solver_timeout,
                "eval_timeout": args.eval_timeout,
                "max_tokens": 4096,
                "sampling": "provider defaults; same settings for both arms",
                "harness_version": "5.0.2",
                "release_images": True,
                "solver_policy": "current-code-only-v1; no upstream solutions or hidden artifacts",
            },
            indent=2,
        ),
    )
    print(f"Prepared {len(tasks)} tasks at dataset revision {revision}")


async def run(args: argparse.Namespace) -> None:
    config, _ = read_experiment(args.experiment)
    await freeze_protocol(args.experiment)
    provider = OpenAICompatibleProvider(
        OpenAICompatibleConfig(
            api_key=read_api_key(args.api_key_file),
            base_url=DEEPSEEK_URL,
            provider_name="deepseek",
            max_tokens=config["max_tokens"],
            max_retries=0,
            timeout_seconds=90,
            infer_api_from_model=False,
        )
    )
    try:
        await run_stream(args.experiment, args.arm, provider)
    finally:
        await provider.aclose()


async def preflight(root: Path) -> None:
    config, tasks = read_experiment(root)
    await freeze_protocol(root)
    task = tasks[0]
    pinned, _ = await ensure_image(task["image"])
    feedback = await official_score(
        dict(task, image=pinned), task["patch"], root / "preflight", timeout=config["eval_timeout"]
    )
    if feedback.status != "passed":
        raise RuntimeError("Reference-patch preflight failed; inspect preflight/harness.log")
    print("Official reference-patch preflight passed (not a model result).")


async def retry_score(args: argparse.Namespace) -> None:
    await freeze_protocol(args.experiment)
    await rescore(args.experiment, args.arm, args.task_id, diagnose_timeout=args.diagnose_timeout)


async def calibrate(args: argparse.Namespace) -> None:
    await freeze_protocol(args.experiment)
    await regrade(args.experiment, args.eval_timeout)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    sub = parser.add_subparsers(dest="operation", required=True)
    init = sub.add_parser("prepare")
    init.add_argument("--repo", default="sympy/sympy")
    init.add_argument("--limit", type=int, default=10)
    init.add_argument("--batch-size", type=int, default=2)
    init.add_argument("--model", default="deepseek-flash")
    init.add_argument("--revision", help="Exact dataset commit SHA; otherwise pin the current SHA")
    init.add_argument("--max-turns", type=int, default=20)
    init.add_argument("--solver-timeout", type=int, default=300)
    init.add_argument("--eval-timeout", type=int, default=1800)
    execute = sub.add_parser("run")
    execute.add_argument("--arm", choices=("frozen", "evolving"), required=True)
    execute.add_argument("--api-key-file", type=Path)
    retry = sub.add_parser("rescore")
    retry.add_argument("--arm", choices=("frozen", "evolving"), required=True)
    retry.add_argument("--task-id", required=True)
    retry.add_argument(
        "--diagnose-timeout",
        action="store_true",
        help="For official ambiguous test timeouts, run a same-budget reference-patch control",
    )
    sub.add_parser("preflight")
    calibration = sub.add_parser("regrade")
    calibration.add_argument("--eval-timeout", type=int, required=True)
    sub.add_parser("summary")
    args = parser.parse_args()
    if args.operation == "prepare":
        if min(args.batch_size, args.max_turns, args.solver_timeout, args.eval_timeout) < 1:
            parser.error("Budgets and batch size must be positive")
        prepare(args)
    elif args.operation == "run":
        asyncio.run(run(args))
    elif args.operation == "preflight":
        asyncio.run(preflight(args.experiment))
    elif args.operation == "rescore":
        asyncio.run(retry_score(args))
    elif args.operation == "regrade":
        asyncio.run(calibrate(args))
    else:
        print(json.dumps(summarize(args.experiment), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
