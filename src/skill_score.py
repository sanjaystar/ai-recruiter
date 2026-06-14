"""
skill_score.py
Evaluates candidate skills based on taxonomy matches, proficiency, duration, endorsements, and platform assessments.
"""

from .schema_analyzer import get_skills, get_redrob_signals, safe_float, safe_int
from .skill_taxonomy import get_canonical_skill

PROFICIENCY_WEIGHTS = {
    "expert": 1.0,
    "advanced": 0.8,
    "intermediate": 0.5,
    "beginner": 0.2
}

def calculate_skill_score(candidate: dict, jd_data: dict, taxonomy_cache: dict = None) -> float:
    """
    Computes a raw skill score for the candidate based on JD requirements.
    Assessment scores have the highest impact, followed by proficiency, duration, and endorsements.
    """
    required_domains = set(jd_data.get("required_skills", []))
    preferred_domains = set(jd_data.get("preferred_skills", []))
    
    if not required_domains and not preferred_domains:
        # Fallback if JD parser found no requirements
        return 0.0

    skills_list = get_skills(candidate)
    signals = get_redrob_signals(candidate)
    assessments = signals.get("skill_assessment_scores", {})
    
    score = 0.0
    processed_domains = set()
    
    # 1. Process Explicitly Listed Skills
    for skill_obj in skills_list:
        raw_name = skill_obj.get("name", "")
        if not raw_name:
            continue
            
        # Fast taxonomy lookup
        if taxonomy_cache is not None and raw_name in taxonomy_cache:
            domain = taxonomy_cache[raw_name]
        else:
            domain = get_canonical_skill(raw_name)
            if taxonomy_cache is not None:
                taxonomy_cache[raw_name] = domain
                
        # Only score skills relevant to the JD
        is_required = domain in required_domains
        is_preferred = domain in preferred_domains
        
        if not is_required and not is_preferred:
            continue
            
        processed_domains.add(domain)
            
        # Calculate base value for this skill
        prof_str = skill_obj.get("proficiency", "beginner").lower()
        prof_val = PROFICIENCY_WEIGHTS.get(prof_str, 0.2)
        
        duration = safe_int(skill_obj.get("duration_months", 0))
        # Cap duration benefit at 60 months (5 years) to avoid runaway scores
        duration_multiplier = min(duration / 60.0, 1.0) 
        
        endorsements = safe_int(skill_obj.get("endorsements", 0))
        # Cap endorsement bonus to prevent farming abuse
        endorsement_bonus = min(endorsements / 50.0, 0.5) 
        
        skill_val = prof_val + (duration_multiplier * 0.5) + endorsement_bonus
        
        # Platform assessments overwrite self-reported proficiency with objective data
        assessment_score = 0.0
        if raw_name in assessments:
            assessment_score = safe_float(assessments[raw_name]) / 100.0
        elif domain in assessments:
            assessment_score = safe_float(assessments[domain]) / 100.0
            
        if assessment_score > 0:
            # Huge boost: objective assessment replaces self-reported proficiency and doubles impact
            skill_val = (assessment_score * 2.0) + (duration_multiplier * 0.5) + endorsement_bonus

        # Apply JD requirement multiplier
        if is_required:
            score += skill_val * 2.5
        elif is_preferred:
            score += skill_val * 1.0

    # 2. Process Unlisted but Assessed Skills
    # (Candidate took a platform assessment but didn't list the skill in their profile array)
    for raw_name, val in assessments.items():
        domain = get_canonical_skill(raw_name)
        if domain in processed_domains:
            continue # Already scored above
            
        is_required = domain in required_domains
        is_preferred = domain in preferred_domains
        
        if is_required:
            score += (safe_float(val) / 100.0) * 3.0
        elif is_preferred:
            score += (safe_float(val) / 100.0) * 1.5

    return score
