"""Content-addressed skill snapshots with serialized, atomic activation."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from tau_coding.evo.models import SkillSnapshot


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".evo-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class SkillStore:
    """Trusted local state. A crashed writer's lock requires manual inspection."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def snapshot(self, version: str | None = None) -> SkillSnapshot:
        if version is None:
            pointer = self.root / "current.json"
            if not pointer.exists():
                return SkillSnapshot()
            value = json.loads(pointer.read_text(encoding="utf-8"))
            if (
                not isinstance(value, dict)
                or set(value) != {"version"}
                or not isinstance(value["version"], str)
            ):
                raise ValueError("Invalid current skill pointer")
            version = value["version"]
        if version == "empty":
            return SkillSnapshot()
        if len(version) != 64 or any(c not in "0123456789abcdef" for c in version):
            raise ValueError("Invalid skill snapshot ID")
        snapshot = SkillSnapshot.model_validate_json(
            (self.root / "snapshots" / f"{version}.json").read_text(encoding="utf-8")
        )
        if snapshot.version != version:
            raise ValueError("Skill snapshot checksum mismatch")
        return snapshot

    @contextmanager
    def _lock(self) -> Iterator[None]:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / "writer.lock"
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            raise RuntimeError(
                "Evo store is busy; inspect writer.lock if a writer crashed"
            ) from None
        try:
            os.close(fd)
            yield
        finally:
            path.unlink()

    def publish(self, snapshot: SkillSnapshot, *, expected: str, reviewed: bool) -> str:
        if not reviewed:
            raise ValueError("Review the complete skill snapshot before activation")
        with self._lock():
            if self.snapshot().version != expected:
                raise ValueError("Active snapshot changed; review against the current version")
            version = snapshot.version
            if version != "empty":
                atomic_write(
                    self.root / "snapshots" / f"{version}.json",
                    snapshot.model_dump_json(indent=2),
                )
            atomic_write(self.root / "current.json", json.dumps({"version": version}))
        return version

    def rollback(self, version: str, *, expected: str) -> str:
        return self.publish(self.snapshot(version), expected=expected, reviewed=True)
