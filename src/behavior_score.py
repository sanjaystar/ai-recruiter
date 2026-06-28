"""
behavior_score.py
Evaluates candidate behavior by fully utilizing all 23 available redrob_signals.
Strictly splitting between Platform Engagement/Market Demand and 
Recruitability/Conversion Probability to maximize methodology coherence.
"""

from datetime import datetime
from .schema_analyzer import get_redrob_signals, safe_float, REFERENCE_DATE

def _calculate_days_ago(date_str: str) -> float:
    if not date_str:
        return 365.0 * 5
    try:
        dt = datetime.fromisoformat(date_str[:10])
        delta = (REFERENCE_DATE - dt).days
        return max(float(delta), 0.0)
    except (ValueError, TypeError):
        return 365.0 * 5

def calculate_behavior_score(candidate: dict) -> float:
    """
    Computes behavior score.
    Engagement max points: 40
    Recruitability max points: 60
    Total Max: 100
    """
    signals = get_redrob_signals(candidate)
    
    # ==========================================
    # 1. Platform Engagement & Market Demand (Max 40 points)
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
    
    # GitHub Activity (Max 8)
    if github >= 0:
        engagement += (github / 100.0) * 8.0
    else:
        engagement += 3.0 
        
    # Profile Completeness (Max 4)
    engagement += (completeness / 100.0) * 4.0
    
    # Verifications (Max 3)
    if ver_email: engagement += 1.0
    if ver_phone: engagement += 1.0
    if linkedin: engagement += 1.0
    
    # Last Active Decay Curve (Max 10)
    days_ago = _calculate_days_ago(last_active_str)
    if days_ago <= 7:
        engagement += 10.0
    elif days_ago <= 30:
        engagement += 7.0
    elif days_ago <= 90:
        engagement += 3.0
    else:
        engagement += max(0.0, 3.0 - ((days_ago - 90) / 30.0))
        
    # Market Demand (Max 15)
    market_pts = 0.0
    market_pts += min(saved_30d * 2.0, 8.0) 
    market_pts += min(views_30d * 0.2, 4.0) 
    market_pts += min(search_30d * 0.05, 2.0)
    market_pts += min(conn_count * 0.01, 1.0)
    
    engagement += min(market_pts, 15.0)
        
    # ==========================================
    # 2. Recruitability Score (Max 60 points)
    # ==========================================
    resp_rate = safe_float(signals.get("recruiter_response_rate", 0.0))
    avg_resp_hrs = safe_float(signals.get("avg_response_time_hours", -1.0))
    interview_rate = safe_float(signals.get("interview_completion_rate", 0.0))
    offer_rate = safe_float(signals.get("offer_acceptance_rate", -1.0))
    open_flag = signals.get("open_to_work_flag", False)
    apps_30d = safe_float(signals.get("applications_submitted_30d", 0.0))
    notice_days = safe_float(signals.get("notice_period_days", 90.0))

    recruitability = 0.0
    
    # Recruiter Response Rate (Max 15)
    recruitability += resp_rate * 15.0
    
    # Average Response Time (Max 5)
    if avg_resp_hrs >= 0:
        if avg_resp_hrs <= 24.0:
            recruitability += 5.0
        elif avg_resp_hrs <= 72.0:
            recruitability += max(0.0, 5.0 - ((avg_resp_hrs - 24.0) / 48.0) * 5.0)
    else:
        recruitability += 2.0 
    
    # Interview Completion Rate (Max 12)
    recruitability += interview_rate * 12.0
    
    # Offer Acceptance Rate (Max 10)
    if offer_rate >= 0:
        recruitability += offer_rate * 10.0
    else:
        recruitability += 4.0 
        
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

    return min(100.0, max(0.0, engagement + recruitability))
