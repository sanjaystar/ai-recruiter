"""
reasoning.py
Generates factual, deterministic 1-2 sentence reasonings for the final submission.
Strictly avoids LLMs and hallucinations.
"""

from .schema_analyzer import get_profile, get_skills, get_redrob_signals, safe_float

def generate_reasoning(candidate: dict, rank: int, final_score: float) -> str:
    """
    Constructs a factual justification string leveraging actual profile details.
    """
    profile = get_profile(candidate)
    skills = get_skills(candidate)
    signals = get_redrob_signals(candidate)
    
    yoe = safe_float(profile.get("years_of_experience", 0))
    title = profile.get("current_title", "Professional")
    
    # Sort skills by proficiency and duration to highlight their strongest domains
    sorted_skills = sorted(skills, key=lambda x: (
        x.get("proficiency", "").lower() == "expert",
        x.get("proficiency", "").lower() == "advanced",
        safe_float(x.get("duration_months", 0))
    ), reverse=True)
    
    skill_names = [s.get("name") for s in sorted_skills[:2]]
    skill_str = " and ".join(skill_names) if skill_names else "relevant domains"
    
    resp_rate = safe_float(signals.get("recruiter_response_rate", 0))
    if resp_rate >= 0.7:
        engagement = "Strong"
    elif resp_rate >= 0.3:
        engagement = "Moderate"
    else:
        engagement = "Limited"
        
    # Inject rank context deterministically
    rank_context = "Top-tier match" if rank <= 10 else ("Solid fit" if rank <= 50 else "Viable candidate")
    
    return f"{rank_context} with {yoe} years experience as a {title} featuring expertise in {skill_str}. {engagement} recruiter engagement signals."
