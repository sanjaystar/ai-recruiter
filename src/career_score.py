"""
career_score.py
Evaluates a candidate's career trajectory, calculating RELEVANT Years of Experience,
title progression, stability, and education fit relative to the JD.
"""
from datetime import datetime
from .schema_analyzer import get_career_history, get_education, safe_str, safe_float
from .skill_taxonomy import TAXONOMY

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

# Education fields relevant to AI/ML engineering
RELEVANT_FIELDS = {
    "computer science", "cs", "artificial intelligence", "ai",
    "machine learning", "ml", "data science", "statistics",
    "mathematics", "math", "applied mathematics", "computational",
    "information technology", "it", "electrical engineering",
    "electronics", "ece", "software engineering", "informatics"
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

def _build_jd_alias_set(jd_data: dict) -> set:
    """
    Builds a flat set of all taxonomy aliases for the JD's required and preferred skills.
    E.g., if required_skills contains 'RETRIEVAL', this returns
    {'semantic search', 'information retrieval', 'rag', 'candidate search', ...}
    """
    alias_set = set()
    
    for domain_name in jd_data.get("required_skills", []):
        if domain_name in TAXONOMY:
            alias_set.update(TAXONOMY[domain_name])
        else:
            # If it's not a canonical domain name, treat it as a raw alias
            alias_set.add(domain_name.lower())
    
    for domain_name in jd_data.get("preferred_skills", []):
        if domain_name in TAXONOMY:
            alias_set.update(TAXONOMY[domain_name])
        else:
            alias_set.add(domain_name.lower())
    
    return alias_set

def _is_role_relevant(role_title: str, role_desc: str, jd_data: dict, jd_aliases: set) -> bool:
    """
    Determines if a career role is relevant to the JD by checking title and 
    taxonomy alias keywords against the role text.
    """
    if not jd_data:
        return True # Fallback if no JD is passed
        
    role_text = f"{role_title} {role_desc}".lower()
    
    # Check if any taxonomy alias appears in the role text
    for alias in jd_aliases:
        if len(alias) >= 3 and alias in role_text:
            return True
        elif len(alias) < 3 and f" {alias} " in f" {role_text} ":
            # Short aliases (e.g., "go", "ml") require word-boundary matching
            return True
            
    return False

def _calculate_education_score(candidate: dict) -> float:
    """
    Max 5 points for education relevance.
    - Relevant field (CS/AI/ML/Math/Stats): +3
    - Advanced degree (Masters/PhD): +2
    """
    education = get_education(candidate)
    if not education:
        return 0.0
    
    best_field_score = 0.0
    best_degree_score = 0.0
    
    for edu in education:
        field = safe_str(edu.get("field_of_study", "")).lower()
        degree = safe_str(edu.get("degree", "")).lower()
        
        # Check field relevance
        if any(rf in field for rf in RELEVANT_FIELDS):
            best_field_score = 3.0
        
        # Check degree level
        if "phd" in degree or "ph.d" in degree or "doctorate" in degree:
            best_degree_score = max(best_degree_score, 2.0)
        elif "master" in degree or "m.s" in degree or "m.tech" in degree or "mtech" in degree:
            best_degree_score = max(best_degree_score, 2.0)
    
    return best_field_score + best_degree_score

def calculate_career_score(candidate: dict, jd_data: dict = None) -> float:
    """
    Max points: 100
    - Relevant Years of Experience: 50
    - Title Progression (Velocity): 25
    - Stability (Tenure): 20
    - Education Fit: 5
    """
    career = get_career_history(candidate)
    if not career:
        return _calculate_education_score(candidate)
    
    # Build the alias set once per call (cached upstream via jd_data)
    jd_aliases = _build_jd_alias_set(jd_data) if jd_data else set()
        
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
        if _is_role_relevant(title, desc, jd_data, jd_aliases):
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
        
    # 2. Progression Score (Max 25)
    prog_score = 0.0
    level_growth = end_level - start_level
    
    if level_growth > 0:
        prog_score = min(25.0, level_growth * 8.0)
    elif level_growth == 0:
        prog_score = 12.0 # Steady
    else:
        prog_score = 4.0 # Demotion
        
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
    
    # 4. Education Score (Max 5)
    edu_score = _calculate_education_score(candidate)
            
    final_score = yoe_score + prog_score + stab_score + edu_score
    return min(100.0, max(0.0, final_score))

