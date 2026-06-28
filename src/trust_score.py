"""Trust score (0-100): deduct for resume anomalies; flag impossible honeypots."""

from datetime import datetime
from .schema_analyzer import get_profile, get_career_history, get_skills, safe_float, REFERENCE_DATE

def calculate_trust(candidate: dict) -> tuple[float, bool]:
    """Start at 100, deduct for inconsistencies; mathematically impossible ones flag a honeypot."""
    profile = get_profile(candidate)
    career = get_career_history(candidate)
    skills = get_skills(candidate)
    
    score = 100.0
    is_honeypot = False
    
    yoe = safe_float(profile.get("years_of_experience", 0.0))
    yoe_months = yoe * 12.0

    # 1. Skill anomalies
    for skill in skills:
        prof = skill.get("proficiency", "").lower()
        dur = safe_float(skill.get("duration_months", 0))

        # Expert proficiency with 0 months of use is impossible
        if prof == "expert" and dur == 0:
            score -= 50.0
            is_honeypot = True

        if dur > (yoe_months + 60.0):  # 5+ year skill/career discrepancy
            score -= 40.0
            is_honeypot = True
        elif dur > (yoe_months + 24.0):
            score -= 20.0

    # 2. Career anomalies & timelines
    intervals = []
    for role in career:
        dur = safe_float(role.get("duration_months", 0))
        if dur < 0:
            score -= 50.0
            is_honeypot = True
            
        start_str = role.get("start_date")
        end_str = role.get("end_date")
        is_current = role.get("is_current", False)
        
        try:
            if start_str:
                start_dt = datetime.fromisoformat(start_str[:10])
                if is_current or not end_str:
                    # Current roles run up to the dataset's reference "now".
                    end_dt = REFERENCE_DATE
                else:
                    end_dt = datetime.fromisoformat(end_str[:10])
                    
                if start_dt > end_dt:
                    score -= 50.0  # Negative date range
                    is_honeypot = True
                else:
                    intervals.append((start_dt, end_dt))
        except (ValueError, TypeError):
            continue

    # Overlapping jobs: histories are short (<10), so O(N^2) is fine
    overlap_count = 0
    for i in range(len(intervals)):
        for j in range(i + 1, len(intervals)):
            s1, e1 = intervals[i]
            s2, e2 = intervals[j]

            overlap_start = max(s1, s2)
            overlap_end = min(e1, e2)

            if overlap_start < overlap_end:
                overlap_days = (overlap_end - overlap_start).days
                if overlap_days > 90:  # ignore short transition overlaps
                    overlap_count += 1

    if overlap_count > 0:
        score -= (overlap_count * 15.0)
        if overlap_count >= 3:
            is_honeypot = True

    # 3. Experience vs. timeline: legit profiles have yoe ~= career span
    # (99th-percentile gap 0.4y), so a multi-year excess is manufactured.
    if intervals:
        earliest = min(s for s, _ in intervals)
        latest = max(e for _, e in intervals)
        span_years = max(0.0, (latest - earliest).days / 365.0)
        if yoe - span_years >= 5.0:
            score -= 45.0
            is_honeypot = True

    return max(0.0, score), is_honeypot
