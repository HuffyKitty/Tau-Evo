"""Deterministic selection with an exact UTF-8 byte budget (not a token estimate)."""

from __future__ import annotations

from tau_coding.evo.models import Selection, SkillRevision, SkillSnapshot


def render_skill(skill: SkillRevision) -> str:
    return f"### {skill.id}@{skill.revision}\n{skill.body.strip()}\n"


def select_skills(
    snapshot: SkillSnapshot, prompt: str, *, max_bytes: int = 8000, max_skills: int = 3
) -> Selection:
    if max_bytes < 0 or max_skills < 0:
        raise ValueError("Selection budgets must be nonnegative")
    query = prompt.casefold()
    ranked: list[tuple[int, SkillRevision]] = []
    for skill in snapshot.skills:
        score = sum(bool(word.strip()) and word.casefold() in query for word in skill.triggers)
        if score or skill.scope == "general":
            ranked.append((score, skill))
    ranked.sort(key=lambda item: (-item[0], item[1].id))
    chosen: list[SkillRevision] = []
    used = 0
    for _, skill in ranked:
        size = len(render_skill(skill).encode("utf-8"))
        if len(chosen) < max_skills and used + size <= max_bytes:
            chosen.append(skill)
            used += size
    return Selection(
        snapshot=snapshot.version,
        skills=tuple(chosen),
        bytes_used=used,
        max_bytes=max_bytes,
        max_skills=max_skills,
    )


def render_selection(selection: Selection) -> str:
    if not selection.skills:
        return ""
    return (
        "## Reviewed Evo guidance\n"
        "Apply only when relevant; the user's explicit task takes precedence.\n"
        + "".join(render_skill(skill) for skill in selection.skills)
    )
