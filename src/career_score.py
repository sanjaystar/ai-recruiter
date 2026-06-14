"""
career_score.py
Evaluates a candidate's career trajectory, calculating RELEVANT Years of Experience
and title progression relative to the JD.
"""
from datetime import datetime
from .schema_analyzer import get_career_history, safe_str, safe_float

REFERENCE_DATE = datetime(2024, 6, 1)

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
    return best_level if best_level > 0 else 3 # Default to standard mid-level

def _parse_date(date_str: str) -> datetime:
    if not date_str:
        return REFERENCE_DATE
    try:
        return datetime.fromisoformat(date_str[:10])
    except (ValueError, TypeError):
        return REFERENCE_DATE

def _is_role_relevant(role_title: str, role_desc: str, jd_data: dict) -> bool:
    """
    Determines if a career role is relevant to the JD by checking title and keywords.
    """
    if not jd_data:
        return True # Fallback if no JD is passed
        
    role_text = f"{role_title} {role_desc}".lower()
    
    # Check if target title matches
    target_title = safe_str(jd_data.get("title", "")).lower()
    if target_title and target_title in role_text:
        return True
        
    # Check if any strong must-have skills are mentioned
    for skill in jd_data.get("must_have_skills", []):
        if skill.lower() in role_text:
            return True
            
    for skill in jd_data.get("good_to_have_skills", []):
        if skill.lower() in role_text:
            return True
            
    return False

def calculate_career_score(candidate: dict, jd_data: dict = None) -> float:
    """
    Max points: 100
    - Relevant Years of Experience: 50
    - Title Progression (Velocity): 30
    - Stability (Tenure): 20
    """
    career = get_career_history(candidate)
    if not career:
        return 0.0
        
    relevant_months = 0
    total_months = 0
    job_count = len(career)
    
    # Career history is often newest first. Reverse for chronological.
    career_chronological = sorted(career, key=lambda c: _parse_date(c.get("start_date", "")))
    
    start_level = 0
    end_level = 0
    
    for i, role in enumerate(career_chronological):
        title = safe_str(role.get("title", ""))
        desc = safe_str(role.get("description", ""))
        start_d = _parse_date(role.get("start_date", ""))
        end_d = _parse_date(role.get("end_date", ""))
        
        # Calculate duration
        months = max(1, (end_d.year - start_d.year) * 12 + end_d.month - start_d.month)
        
        # Add to total YoE regardless
        total_months += months
        
        # Add to relevant YoE if it matches JD
        if _is_role_relevant(title, desc, jd_data):
            relevant_months += months
            
        # Track progression
        level = _get_title_level(title)
        if i == 0:
            start_level = level
        end_level = level
        
    relevant_yoe = relevant_months / 12.0
    total_yoe = total_months / 12.0
    
    # 1. Relevant Experience Score (Max 50)
    # Prefer candidates in the 5-10 year relevant sweet spot. 
    yoe_score = 0.0
    if relevant_yoe >= 10:
        yoe_score = 50.0
    elif relevant_yoe >= 5:
        yoe_score = 45.0 + (relevant_yoe - 5) * 1.0 # 45-50 points
    elif relevant_yoe >= 2:
        yoe_score = 25.0 + (relevant_yoe - 2) * 6.6 # 25-45 points
    else:
        yoe_score = relevant_yoe * 12.5 # 0-25 points
        
    # 2. Progression Score (Max 30)
    prog_score = 0.0
    level_growth = end_level - start_level
    
    if level_growth > 0:
        prog_score = min(30.0, level_growth * 10.0)
    elif level_growth == 0:
        prog_score = 15.0 # Steady
    else:
        prog_score = 5.0 # Demotion
        
    # Title Inflation Penalty
    # If they claim to be a CTO (level 7) but only have 2 years of total experience, heavily penalize.
    if end_level >= 6 and total_yoe < 5:
        prog_score *= 0.1
        
    # 3. Stability Score (Max 20)
    # Calculate average tenure per job
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
