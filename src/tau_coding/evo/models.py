"""Versioned, provider-neutral contracts for the Evo task boundary."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tau_agent.messages import Usage


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def now() -> str:
    return datetime.now(UTC).isoformat()


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1] = 1


class SkillRevision(Record):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$")
    revision: int = Field(ge=1)
    scope: Literal["general", "topic"] = "topic"
    triggers: tuple[str, ...] = ()
    body: str = Field(min_length=1, max_length=32000)
    evidence: tuple[str, ...] = ()
    author: str = "human"

    @model_validator(mode="after")
    def check_triggers(self) -> SkillRevision:
        if self.scope == "topic" and not any(word.strip() for word in self.triggers):
            raise ValueError("Topic skills need at least one nonempty trigger")
        if not self.body.strip():
            raise ValueError("Skill body must not be blank")
        return self


class SkillSnapshot(Record):
    skills: tuple[SkillRevision, ...] = ()

    @model_validator(mode="after")
    def unique_ids(self) -> SkillSnapshot:
        if len({skill.id for skill in self.skills}) != len(self.skills):
            raise ValueError("Duplicate skill IDs")
        return self

    @property
    def version(self) -> str:
        if not self.skills:
            return "empty"
        return digest(json.dumps(self.model_dump(mode="json"), sort_keys=True))


class Selection(Record):
    snapshot: str
    skills: tuple[SkillRevision, ...]
    bytes_used: int
    max_bytes: int
    max_skills: int


class VerifierFeedback(Record):
    verifier: str
    status: Literal["passed", "failed", "unknown"]
    exit_code: int | None = None
    output_sha256: str | None = None
    reason: str = ""
    summary: str = ""
    evidence: tuple[str, ...] = ()


class TaskRun(Record):
    run_id: str
    task_id: str
    prompt_sha256: str
    system_sha256: str
    tools_sha256: str
    model: str
    provider: str
    solver_version: str
    selection: Selection
    started_at: str
    finished_at: str | None = None
    status: Literal["running", "completed", "error", "cancelled", "timeout"] = "running"
    max_turns: int
    timeout_seconds: float
    turns: int = 0
    tool_calls: int = 0
    usage: Usage = Field(default_factory=Usage)
    feedback: VerifierFeedback | None = None
    error_type: str | None = None


class CandidateLesson(Record):
    """Future compiler input; only construct from externally failed task runs."""

    run_id: str
    lesson: str = Field(min_length=1)
    trigger: str = Field(min_length=1)
    evidence: tuple[str, ...] = Field(min_length=1)
    scope_hint: Literal["general", "topic"]
