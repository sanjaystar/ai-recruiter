"""Deterministic factual 1-2 sentence reasoning per candidate (no LLM)."""

import hashlib
from datetime import datetime

from .schema_analyzer import get_profile, get_skills, get_redrob_signals, safe_float, REFERENCE_DATE
from .skill_taxonomy import get_canonical_skill

_PROF_RANK = {"expert": 3, "advanced": 2, "intermediate": 1, "beginner": 0}
# Generic software/data skills aren't central to a retrieval/ranking role.
_GENERIC_DOMAINS = {"SOFTWARE_ENGINEERING", "DATA_ENGINEERING"}


def _days_since(date_str: str):
    try:
        return (REFERENCE_DATE - datetime.fromisoformat(date_str[:10])).days
    except (ValueError, TypeError):
        return None


def _relevant_skills(candidate: dict, jd_data: dict) -> list[str]:
    """Candidate skills that map to a JD required/preferred domain, strongest first."""
    wanted = (set(jd_data.get("required_skills", [])) | set(jd_data.get("preferred_skills", []))) - _GENERIC_DOMAINS
    scored = []
    for s in get_skills(candidate):
        name = s.get("name", "")
        if name and get_canonical_skill(name) in wanted:
            scored.append((
                _PROF_RANK.get(s.get("proficiency", "").lower(), 0),
                safe_float(s.get("duration_months", 0)),
                name,
            ))
    scored.sort(reverse=True)
    return [name for _, _, name in scored]


def _honest_concern(signals: dict):
    """The single most salient red flag for hireability, or None if the data is clean."""
    resp = safe_float(signals.get("recruiter_response_rate", -1.0))
    offer = safe_float(signals.get("offer_acceptance_rate", -1.0))
    interview = safe_float(signals.get("interview_completion_rate", -1.0))
    notice = safe_float(signals.get("notice_period_days", 0.0))
    last = _days_since(signals.get("last_active_date", ""))

    if 0.0 <= resp < 0.3:
        return f"low recruiter response rate ({resp:.0%})"
    if last is not None and last > 120:
        return f"inactive for ~{last // 30} months"
    if notice > 60:
        return f"{int(notice)}-day notice period"
    if not signals.get("open_to_work_flag", False):
        return "not marked open-to-work"
    if 0.0 <= offer < 0.3:
        return f"low past offer-acceptance ({offer:.0%})"
    if 0.0 <= interview < 0.5:
        return f"spotty interview completion ({interview:.0%})"
    return None


def generate_reasoning(candidate: dict, rank: int, final_score: float, jd_data: dict = None) -> str:
    """Constructs a varied, factual 1-2 sentence justification for this rank."""
    jd_data = jd_data or {}
    profile = get_profile(candidate)
    signals = get_redrob_signals(candidate)

    yoe = safe_float(profile.get("years_of_experience", 0))
    title = profile.get("current_title") or "Professional"
    company = profile.get("current_company") or ""
    facts = f"{yoe:.1f}y as {title}" + (f" at {company}" if company else "")

    rel = _relevant_skills(candidate, jd_data)
    if rel:
        strength = f"skills central to this retrieval/ranking role ({' and '.join(rel[:2])})"
    elif rank <= 50:
        strength = "fits on demonstrated experience rather than listed buzzwords"
    else:
        strength = "only adjacent skills — included as lower-confidence filler"

    if rank <= 10:
        lead = "Top pick"
    elif rank <= 50:
        lead = "Solid fit"
    else:
        lead = "Borderline"

    # Deterministic structural variety so rows don't read as a template
    cid = candidate.get("candidate_id", "")
    variant = (int(hashlib.md5(cid.encode()).hexdigest(), 16) % 4) if cid else (rank % 4)
    if variant == 0:
        text = f"{lead}: {facts}, with {strength}."
    elif variant == 1:
        text = f"{facts}; {strength}. {lead} for this role."
    elif variant == 2:
        text = f"{lead} — ranked on {strength} ({facts})."
    else:
        text = f"{facts}. Brings {strength}; rated a {lead.lower()}."

    concern = _honest_concern(signals)
    if concern:
        text += f" Concern: {concern}."
    elif rank <= 10:
        text += " Clean behavioral signals."
    return text
