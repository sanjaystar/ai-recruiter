"""
behavior_score.py
Evaluates candidate behavior by fully utilizing all 23 available redrob_signals.
Strictly splitting between Platform Engagement/Market Demand,
Recruitability/Conversion Probability, and Location/Logistics fit
to maximize methodology coherence.
"""

from datetime import datetime
from .schema_analyzer import get_profile, get_redrob_signals, safe_float, safe_str

# Assume a static reference date for deterministic scoring in the hackathon dataset
REFERENCE_DATE = datetime(2026, 6, 20)

# JD specifies Pune/Noida with openness to Tier-1 Indian cities
PREFERRED_LOCATIONS = {"pune", "noida"}
GOOD_LOCATIONS = {"hyderabad", "mumbai", "bangalore", "bengaluru", "delhi", "gurgaon", "gurugram", "chennai", "ncr", "new delhi"}

def _calculate_days_ago(date_str: str) -> float:
    if not date_str:
        return 365.0 * 5
    try:
        dt = datetime.fromisoformat(date_str[:10])
        delta = (REFERENCE_DATE - dt).days
        return max(float(delta), 0.0)
    except (ValueError, TypeError):
        return 365.0 * 5

def calculate_behavior_score(candidate: dict, jd_data: dict = None) -> float:
    """
    Computes behavior score.
    Engagement max points: 35
    Recruitability max points: 55
    Location & Logistics max points: 10
    Total Max: 100
    """
    profile = get_profile(candidate)
    signals = get_redrob_signals(candidate)
    
    # ==========================================
    # 1. Platform Engagement & Market Demand (Max 35 points)
    # ==========================================
    github = safe_float(signals.get("github_activity_score", -1.0))
    completeness = safe_float(signals.get("profile_completeness_score", 0.0))
    last_active_str = signals.get("last_active_date", "")
    
    ver_email = signals.get("verified_email", False)
    ver_phone = signals.get("verified_phone", False)
    linkedin = signals.get("linkedin_connected", False)
    
    saved_30d = safe_float(signals.get("saved_by_recruiters_30d", 0.0))
    views_30d = safe_float(signals.get("profile_views_received_30d", 0.0))
    search_30d = safe_float(signals.get("search_appearance_30d", 0.0))
    conn_count = safe_float(signals.get("connection_count", 0.0))
    
    engagement = 0.0
    
    # GitHub Activity (Max 7)
    if github >= 0:
        engagement += (github / 100.0) * 7.0
    else:
        engagement += 2.5 
        
    # Profile Completeness (Max 4)
    engagement += (completeness / 100.0) * 4.0
    
    # Verifications (Max 3)
    if ver_email: engagement += 1.0
    if ver_phone: engagement += 1.0
    if linkedin: engagement += 1.0
    
    # Last Active Decay Curve (Max 8)
    days_ago = _calculate_days_ago(last_active_str)
    if days_ago <= 7:
        engagement += 8.0
    elif days_ago <= 30:
        engagement += 6.0
    elif days_ago <= 90:
        engagement += 3.0
    else:
        engagement += max(0.0, 3.0 - ((days_ago - 90) / 30.0))
        
    # Market Demand (Max 13)
    market_pts = 0.0
    market_pts += min(saved_30d * 2.0, 7.0) 
    market_pts += min(views_30d * 0.2, 3.5) 
    market_pts += min(search_30d * 0.05, 1.5)
    market_pts += min(conn_count * 0.01, 1.0)
    
    engagement += min(market_pts, 13.0)
        
    # ==========================================
    # 2. Recruitability Score (Max 55 points)
    # ==========================================
    resp_rate = safe_float(signals.get("recruiter_response_rate", 0.0))
    avg_resp_hrs = safe_float(signals.get("avg_response_time_hours", -1.0))
    interview_rate = safe_float(signals.get("interview_completion_rate", 0.0))
    offer_rate = safe_float(signals.get("offer_acceptance_rate", -1.0))
    open_flag = signals.get("open_to_work_flag", False)
    apps_30d = safe_float(signals.get("applications_submitted_30d", 0.0))
    notice_days = safe_float(signals.get("notice_period_days", 90.0))
    
    # Salary Check
    cand_salary_min = safe_float(signals.get("expected_salary_range_inr_lpa", {}).get("min", -1.0))
    jd_salary_budget = 0.0
    if jd_data:
        # JD data might have salary_budget_lpa. If missing, assume 0.0 (no constraint)
        jd_salary_budget = safe_float(jd_data.get("salary_budget_lpa", 0.0))
        
    recruitability = 0.0
    
    # Recruiter Response Rate (Max 14)
    recruitability += resp_rate * 14.0
    
    # Average Response Time (Max 5)
    if avg_resp_hrs >= 0:
        if avg_resp_hrs <= 24.0:
            recruitability += 5.0
        elif avg_resp_hrs <= 72.0:
            recruitability += max(0.0, 5.0 - ((avg_resp_hrs - 24.0) / 48.0) * 5.0)
    else:
        recruitability += 2.0 
    
    # Interview Completion Rate (Max 10)
    recruitability += interview_rate * 10.0
    
    # Offer Acceptance Rate (Max 8)
    if offer_rate >= 0:
        recruitability += offer_rate * 8.0
    else:
        recruitability += 3.0 
        
    # Explicit Intent Signal (Max 5)
    if open_flag:
        recruitability += 5.0
        
    # Active Application Seeking (Max 3)
    if apps_30d > 0:
        recruitability += min(apps_30d * 0.5, 3.0)
        
    # Notice Period (Logistical ease of hiring - Max 10)
    if notice_days <= 30:
        recruitability += 10.0
    elif notice_days <= 60:
        recruitability += 6.0
    elif notice_days <= 90:
        recruitability += 2.0
        
    # Salary Mismatch Penalty
    if jd_salary_budget > 0 and cand_salary_min > 0:
        if cand_salary_min > jd_salary_budget * 1.2:
            # Heavily penalize if candidate minimum is 20%+ higher than JD budget
            recruitability *= 0.5
        elif cand_salary_min > jd_salary_budget:
            # Mildly penalize if candidate minimum is higher than budget
            recruitability *= 0.8
    
    # ==========================================
    # 3. Location & Work-Mode Fit (Max 10 points)
    # ==========================================
    location_str = safe_str(profile.get("location", "")).lower()
    country = safe_str(profile.get("country", "")).lower()
    work_mode = safe_str(signals.get("preferred_work_mode", "")).lower()
    relocate = signals.get("willing_to_relocate", False)
    
    location_score = 0.0
    
    # City Match (Max 4)
    if any(city in location_str for city in PREFERRED_LOCATIONS):
        location_score += 4.0  # Exact Pune/Noida match
    elif any(city in location_str for city in GOOD_LOCATIONS):
        if relocate:
            location_score += 3.0  # Good city + willing to relocate
        else:
            location_score += 1.5  # Good city but won't relocate
    elif "india" in country:
        if relocate:
            location_score += 2.0  # India-based + willing to relocate
        else:
            location_score += 0.5
    # Outside India: 0 points (JD says no visa sponsorship)
    
    # Work Mode Compatibility (Max 3) — JD is hybrid
    if work_mode in ("hybrid", "flexible"):
        location_score += 3.0
    elif work_mode == "onsite":
        location_score += 2.0  # Onsite is compatible with hybrid
    elif work_mode == "remote":
        location_score += 0.5  # Remote-only is a mild concern for hybrid role
    
    # Relocation Willingness Bonus (Max 3)
    if relocate:
        location_score += 3.0
            
    return min(100.0, max(0.0, engagement + recruitability + location_score))

