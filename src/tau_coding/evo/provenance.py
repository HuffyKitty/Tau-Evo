"""Freeze solver implementation, manifest and scorer image across experiment arms."""

from __future__ import annotations

import ast
import json
from pathlib import Path

from tau_coding.evo.models import digest
from tau_coding.evo.store import atomic_write
from tau_coding.evo.swe import command


def solver_digest() -> str:
    source = Path(__file__).resolve().parents[2]
    parts: dict[str, str] = {}
    for package in ("tau_ai", "tau_agent", "tau_coding"):
        for path in sorted((source / package).rglob("*.py")):
            if path.parent.name == "evo" and path.name not in {
                "runner.py",
                "selection.py",
                "models.py",
                "evidence.py",
            }:
                continue
            parts[path.relative_to(source).as_posix()] = ast.dump(
                ast.parse(path.read_text("utf-8"))
            )
    # Controller/reporting changes do not alter the solver. Freeze its actual adapter too.
    for filename, name in (("experiment.py", "solve_task"), ("swe.py", "TaskContainer")):
        tree = ast.parse((Path(__file__).parent / filename).read_text("utf-8"))
        node = next(node for node in tree.body if getattr(node, "name", None) == name)
        parts[filename + ":" + name] = ast.dump(node)
    return digest(json.dumps(parts, sort_keys=True))


async def freeze_protocol(root: Path) -> None:
    code, output = await command(
        ["docker", "image", "inspect", "tau-evo-grader:5.0.2", "--format", "{{.Id}}"]
    )
    if code:
        raise RuntimeError("Build the Linux grader image first; see the SWE-bench guide")
    value = {
        "grader_image_id": output.strip(),
        "swebench": "5.0.2",
        "solver_sha256": solver_digest(),
        "manifest_sha256": digest((root / "experiment.json").read_text("utf-8")),
    }
    pin = root / "protocol.json"
    if pin.exists():
        if json.loads(pin.read_text("utf-8")) != value:
            raise ValueError("Solver, grader or manifest changed during the frozen experiment")
    else:
        atomic_write(pin, json.dumps(value, indent=2))
