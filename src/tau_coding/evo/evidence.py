"""Bounded, credential-redacted text for externally grounded reflection."""

import re


def redact(text: str, limit: int = 6000) -> str:
    text = re.sub(r"sk-[A-Za-z0-9_-]+", "[REDACTED]", text)
    text = re.sub(
        r"(?i)(authorization|api[_-]?key|access[_-]?token|password)([\s:=]+)[^\s,;]+",
        r"\1\2[REDACTED]",
        text,
    )
    return text[-limit:]
