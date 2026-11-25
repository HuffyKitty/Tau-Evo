"""Trusted host verification, separate from the solver's conversation."""

from __future__ import annotations

import asyncio
import hashlib
import os
import signal
import subprocess
import tempfile
from contextlib import suppress
from pathlib import Path

from tau_coding.evo.models import VerifierFeedback


async def verify_command(
    argv: list[str],
    cwd: Path,
    *,
    timeout_seconds: float = 30,
) -> VerifierFeedback:
    """Run argv without a shell. Retain a hash, never raw command output."""
    if not argv or timeout_seconds <= 0:
        raise ValueError("Verifier requires argv and a positive timeout")
    environment = {
        key: value
        for key, value in os.environ.items()
        if not any(marker in key.upper() for marker in ("KEY", "TOKEN", "SECRET", "PASSWORD"))
    }
    with tempfile.TemporaryFile() as output:
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                cwd=cwd,
                env=environment,
                stdout=output,
                stderr=output,
                start_new_session=os.name == "posix",
                creationflags=(
                    getattr(subprocess, "CREATE_NEW_PROCESS_GROUP")  # noqa: B009
                    if os.name == "nt"
                    else 0
                ),
            )
        except OSError:
            return VerifierFeedback(verifier="command", status="unknown", reason="spawn_error")
        try:
            await asyncio.wait_for(process.wait(), timeout_seconds)
        except (TimeoutError, asyncio.CancelledError) as exc:
            if process.returncode is None:
                if os.name == "posix":
                    with suppress(ProcessLookupError):
                        getattr(os, "killpg")(process.pid, getattr(signal, "SIGKILL"))  # noqa: B009
                else:
                    killer = await asyncio.create_subprocess_exec(
                        "taskkill",
                        "/PID",
                        str(process.pid),
                        "/T",
                        "/F",
                        stdout=asyncio.subprocess.DEVNULL,
                        stderr=asyncio.subprocess.DEVNULL,
                    )
                    await killer.wait()
                await process.wait()
            if isinstance(exc, asyncio.CancelledError):
                raise
            return VerifierFeedback(verifier="command", status="unknown", reason="timeout")
        output.seek(0)
        fingerprint = hashlib.file_digest(output, "sha256").hexdigest()
    return VerifierFeedback(
        verifier="command",
        status="passed" if process.returncode == 0 else "failed",
        exit_code=process.returncode,
        output_sha256=fingerprint,
    )
