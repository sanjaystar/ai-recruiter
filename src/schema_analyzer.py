"""
schema_analyzer.py
Provides robust, fast, and safe access to nested candidate data structures to prevent key errors.
"""

from datetime import datetime

# Reference "today" for the dataset. Derived from the dataset's own activity range
# (max last_active_date = 2026-05-27), NOT the real wall-clock date. All recency/notice
# scoring is measured against this so the signals actually differentiate candidates.
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
    """Safely converts a value to float, catching None or invalid strings."""
    try:
        if val is None:
            return default
        return float(val)
    except (ValueError, TypeError):
        return default

def safe_int(val, default=0) -> int:
    """Safely converts a value to int, catching None or invalid strings."""
    try:
        if val is None:
            return default
        return int(val)
    except (ValueError, TypeError):
        return default

def safe_str(val, default="") -> str:
    """Safely converts a value to string, catching None."""
    return str(val) if val is not None else default
