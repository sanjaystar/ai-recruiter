"""Safe accessors for nested candidate fields."""

from datetime import datetime

# Dataset reference "today" (from max last_active_date 2026-05-27), not wall-clock.
REFERENCE_DATE = datetime(2026, 6, 1)

def get_candidate_id(candidate: dict) -> str:
    return candidate.get("candidate_id", "")

def get_profile(candidate: dict) -> dict:
    return candidate.get("profile", {})

def get_career_history(candidate: dict) -> list[dict]:
    return candidate.get("career_history", [])

def get_skills(candidate: dict) -> list[dict]:
    return candidate.get("skills", [])

def get_redrob_signals(candidate: dict) -> dict:
    return candidate.get("redrob_signals", {})

def safe_float(val, default=0.0) -> float:
    try:
        if val is None:
            return default
        return float(val)
    except (ValueError, TypeError):
        return default

def safe_int(val, default=0) -> int:
    try:
        if val is None:
            return default
        return int(val)
    except (ValueError, TypeError):
        return default

def safe_str(val, default="") -> str:
    return str(val) if val is not None else default
