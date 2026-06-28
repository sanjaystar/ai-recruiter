"""Career score: relevant YoE, title progression, and tenure stability vs the JD."""
import numpy as np

from datetime import datetime
from .schema_analyzer import get_career_history, safe_str, REFERENCE_DATE
from .skill_score import DOMAIN_PHRASES

# Generic domains never count as applied-ML/retrieval work on their own.
_GENERIC_DOMAINS = {"SOFTWARE_ENGINEERING", "DATA_ENGINEERING"}
# Min cosine to a core JD domain for a role to count; tuned on the pool (real ~0.28-0.56, stuffers <=0.21).
ROLE_RELEVANCE_THRESHOLD = 0.30

TITLE_HIERARCHY = {
    "intern": 1, "trainee": 1,
    "junior": 2, "associate": 2, "analyst": 2,
    "engineer": 3, "developer": 3, "scientist": 3, "specialist": 3,
    "senior": 4, "lead": 4, "manager": 4,
    "staff": 5, "principal": 5, "director": 5,
    "vp": 6, "vice president": 6, "head": 6,
    "c-level": 7, "cto": 7, "ceo": 7, "founder": 7
}

def _get_title_level(title: str) -> int:
    title_lower = title.lower()
    best_level = 0
    for key, level in TITLE_HIERARCHY.items():
        if key in title_lower and level > best_level:
            best_level = level
    return best_level if best_level > 0 else 3  # default mid-level

def _parse_date(date_str: str) -> datetime:
    if not date_str:
        return REFERENCE_DATE
    try:
        return datetime.fromisoformat(date_str[:10])
    except (ValueError, TypeError):
        return REFERENCE_DATE

def build_career_anchors(model, jd_data: dict) -> np.ndarray:
    """Embedding anchors from the JD's core domains (generic excluded), or None if none."""
    if not jd_data:
        return None
    core = {
        d for d in (set(jd_data.get("required_skills", [])) | set(jd_data.get("preferred_skills", [])))
        if d in DOMAIN_PHRASES and d not in _GENERIC_DOMAINS
    }
    if not core:
        return None
    return model.encode([DOMAIN_PHRASES[d] for d in sorted(core)], normalize_embeddings=True)


def role_relevance_from_embeddings(role_embeddings: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """Per-role relevance: True where a role embeds within ROLE_RELEVANCE_THRESHOLD of any anchor."""
    if anchors is None or len(role_embeddings) == 0:
        return np.zeros(len(role_embeddings), dtype=bool)
    return (role_embeddings @ anchors.T).max(axis=1) >= ROLE_RELEVANCE_THRESHOLD


def calculate_career_score(candidate: dict, jd_data: dict = None, role_relevance=None) -> float:
    """Score (max 100): relevant YoE (50) + title progression (30) + stability (20)."""
    career = get_career_history(candidate)
    if not career:
        return 0.0

    relevant_months = 0
    total_months = 0
    job_count = len(career)

    # Per-role relevance precomputed from embeddings; fallback treats every role as relevant.
    if role_relevance is None or len(role_relevance) != len(career):
        role_relevance = [True] * len(career)

    # History is often newest-first; sort chronological, keeping each relevance flag attached.
    paired = sorted(zip(career, role_relevance),
                    key=lambda cr: _parse_date(cr[0].get("start_date", "")))

    start_level = 0
    end_level = 0

    for i, (role, is_relevant) in enumerate(paired):
        title = safe_str(role.get("title", ""))
        start_d = _parse_date(role.get("start_date", ""))
        end_d = _parse_date(role.get("end_date", ""))

        months = max(1, (end_d.year - start_d.year) * 12 + end_d.month - start_d.month)
        total_months += months
        if is_relevant:
            relevant_months += months

        level = _get_title_level(title)
        if i == 0:
            start_level = level
        end_level = level
        
    relevant_yoe = relevant_months / 12.0
    total_yoe = total_months / 12.0

    # 1. Relevant experience (max 50), favoring the 5-10 year sweet spot
    yoe_score = 0.0
    if relevant_yoe >= 10:
        yoe_score = 50.0
    elif relevant_yoe >= 5:
        yoe_score = 45.0 + (relevant_yoe - 5) * 1.0
    elif relevant_yoe >= 2:
        yoe_score = 25.0 + (relevant_yoe - 2) * 6.6
    else:
        yoe_score = relevant_yoe * 12.5

    # 2. Progression (max 30)
    prog_score = 0.0
    level_growth = end_level - start_level

    if level_growth > 0:
        prog_score = min(30.0, level_growth * 10.0)
    elif level_growth == 0:
        prog_score = 15.0
    else:
        prog_score = 5.0

    # Title inflation: senior title with little total experience
    if end_level >= 6 and total_yoe < 5:
        prog_score *= 0.1

    # 3. Stability (max 20): average tenure per job
    stab_score = 0.0
    if job_count > 0:
        avg_tenure_years = total_yoe / job_count
        if avg_tenure_years >= 3.0:
            stab_score = 20.0
        elif avg_tenure_years >= 1.5:
            stab_score = 10.0 + ((avg_tenure_years - 1.5) / 1.5) * 10.0
        else:
            stab_score = avg_tenure_years * 6.6

    final_score = yoe_score + prog_score + stab_score
    return min(100.0, max(0.0, final_score))
