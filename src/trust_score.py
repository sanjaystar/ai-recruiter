"""
trust_score.py
Calculates a baseline trust score (0-100) by detecting logical anomalies in the resume.
Returns the score, a boolean honeypot flag, and a list of human-readable concern strings.
"""

from datetime import datetime
from .schema_analyzer import get_profile, get_career_history, get_skills, safe_float

def calculate_trust(candidate: dict) -> tuple[float, bool, list[str]]:
    """
    Base score is 100. Deductions are made for inconsistencies.
    If anomalies are mathematically impossible, is_honeypot is set to True.
    Returns (score, is_honeypot, concerns).
    """
    profile = get_profile(candidate)
    career = get_career_history(candidate)
    skills = get_skills(candidate)
    
    score = 100.0
    is_honeypot = False
    concerns = []
    
    yoe = safe_float(profile.get("years_of_experience", 0.0))
    yoe_months = yoe * 12.0
    
    # ==========================================
    # 1. Skill Anomalies
    # ==========================================
    expert_zero_count = 0
    duration_anomaly_count = 0
    
    for skill in skills:
        prof = skill.get("proficiency", "").lower()
        dur = safe_float(skill.get("duration_months", 0))
        
        # Impossible: Expert proficiency but 0 months of use
        if prof == "expert" and dur == 0:
            score -= 50.0
            is_honeypot = True
            expert_zero_count += 1
            
        # Impossible: Claiming way more skill experience than total career
        # (We allow a 24-month buffer for pre-professional/university coding)
        if dur > (yoe_months + 60.0): # 5+ years discrepancy is a honeypot
            score -= 40.0
            is_honeypot = True
            duration_anomaly_count += 1
        elif dur > (yoe_months + 24.0):
            score -= 20.0
            duration_anomaly_count += 1
    
    if expert_zero_count > 0:
        concerns.append(f"{expert_zero_count} skill(s) listed as expert with 0 months usage")
    if duration_anomaly_count > 0:
        concerns.append(f"skill duration exceeds total career experience in {duration_anomaly_count} skill(s)")
            
    # ==========================================
    # 2. Career Anomalies & Timelines
    # ==========================================
    intervals = []
    negative_dates = False
    
    for role in career:
        dur = safe_float(role.get("duration_months", 0))
        if dur < 0:
            score -= 50.0
            is_honeypot = True
            negative_dates = True
            
        start_str = role.get("start_date")
        end_str = role.get("end_date")
        is_current = role.get("is_current", False)
        
        try:
            if start_str:
                start_dt = datetime.fromisoformat(start_str[:10])
                if is_current or not end_str:
                    # Treat current roles as extending into the future for overlap checking
                    end_dt = datetime(2026, 12, 31) 
                else:
                    end_dt = datetime.fromisoformat(end_str[:10])
                    
                if start_dt > end_dt:
                    score -= 50.0 # Negative date range
                    is_honeypot = True
                    negative_dates = True
                else:
                    intervals.append((start_dt, end_dt))
        except (ValueError, TypeError):
            continue
    
    if negative_dates:
        concerns.append("impossible date ranges in career timeline")
            
    # Check Overlapping Jobs
    # Since career histories are short (<10 items), O(N^2) is extremely fast
    overlap_count = 0
    for i in range(len(intervals)):
        for j in range(i + 1, len(intervals)):
            s1, e1 = intervals[i]
            s2, e2 = intervals[j]
            
            # Calculate intersection
            overlap_start = max(s1, s2)
            overlap_end = min(e1, e2)
            
            if overlap_start < overlap_end:
                overlap_days = (overlap_end - overlap_start).days
                # Only flag significant overlaps (>90 days) to avoid penalizing standard job transitions
                if overlap_days > 90:
                    overlap_count += 1
                    
    if overlap_count > 0:
        # Deduct points for concurrent roles (e.g., "Overemployed" or fake profiles)
        score -= (overlap_count * 15.0)
        concerns.append(f"{overlap_count} significantly overlapping concurrent roles")
        # 3+ overlapping full-time long-term roles is statistically absurd
        if overlap_count >= 3: 
            is_honeypot = True

    return max(0.0, score), is_honeypot, concerns
