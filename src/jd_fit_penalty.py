"""
jd_fit_penalty.py
Detects JD-specific red flags using smooth compounding penalties.
Returns (penalty_multiplier, reasons_list) so reasoning can reference computed checks.
"""

from .schema_analyzer import get_profile, get_career_history, get_skills, get_redrob_signals, safe_float, safe_str
from .skill_taxonomy import TAXONOMY

# Consulting/services firms the JD explicitly flags
CONSULTING_FIRMS = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "tech mahindra", "mindtree",
    "mphasis", "l&t infotech", "lti", "ltimindtree", "hexaware",
    "cyient", "persistent", "zensar"
}

# Terms indicating production/deployment experience (not just research)
PRODUCTION_TERMS = {
    "production", "deploy", "deployed", "deployment", "shipped",
    "scaled", "scaling", "pipeline", "infrastructure", "serving",
    "latency", "throughput", "api", "microservice", "kubernetes",
    "docker", "ci/cd", "monitoring", "sla", "users", "traffic",
    "real-time", "batch processing", "etl", "data pipeline"
}

# Non-technical titles that suggest keyword-stuffing if paired with heavy AI skills
NON_TECH_TITLES = {
    "marketing", "sales", "hr", "human resources", "recruiter",
    "business development", "account manager", "project manager",
    "operations", "finance", "accounting", "admin", "coordinator",
    "content writer", "copywriter", "graphic designer"
}

# Research-only role indicators
RESEARCH_TITLES = {
    "research scientist", "researcher", "research fellow",
    "postdoc", "post-doctoral", "research associate",
    "research assistant", "phd candidate", "phd student"
}


def _get_all_taxonomy_aliases() -> set:
    """Flatten all taxonomy aliases for quick matching."""
    aliases = set()
    for domain_aliases in TAXONOMY.values():
        aliases.update(domain_aliases)
    return aliases


def calculate_jd_fit_penalty(candidate: dict, jd_data: dict = None) -> tuple[float, list[str]]:
    """
    Computes a JD-fit penalty multiplier (0.0-1.0) and human-readable reasons.
    Lower multiplier = worse JD fit. Penalties compound multiplicatively.
    
    Returns:
        (penalty: float, reasons: list[str])
    """
    profile = get_profile(candidate)
    career = get_career_history(candidate)
    skills = get_skills(candidate)
    signals = get_redrob_signals(candidate)
    
    penalty = 1.0
    reasons = []
    
    # Collect all career text for analysis
    all_companies = []
    all_titles = []
    all_descriptions = []
    
    for role in career:
        company = safe_str(role.get("company", "")).lower().strip()
        title = safe_str(role.get("title", "")).lower().strip()
        desc = safe_str(role.get("description", "")).lower().strip()
        all_companies.append(company)
        all_titles.append(title)
        all_descriptions.append(desc)
    
    combined_career_text = " ".join(all_titles + all_descriptions)
    
    # ==========================================
    # 1. Consulting-Only Career Check
    # ==========================================
    if all_companies:
        consulting_count = 0
        for company in all_companies:
            for firm in CONSULTING_FIRMS:
                if firm in company:
                    consulting_count += 1
                    break
        
        consulting_ratio = consulting_count / len(all_companies)
        
        if consulting_ratio >= 1.0 and len(all_companies) >= 2:
            # Every single role is at a consulting firm
            penalty *= 0.5
            reasons.append("career history consists entirely of consulting/services firms")
        elif consulting_ratio >= 0.75 and len(all_companies) >= 3:
            # Mostly consulting
            penalty *= 0.7
            reasons.append("career is predominantly at consulting/services firms")
    
    # ==========================================
    # 2. Title-Chaser Detection
    # ==========================================
    if len(career) >= 4:
        total_months = sum(safe_float(r.get("duration_months", 0)) for r in career)
        avg_tenure_years = (total_months / len(career)) / 12.0
        
        if avg_tenure_years < 1.5:
            penalty *= 0.5
            reasons.append(f"frequent job changes ({len(career)} roles, avg tenure {avg_tenure_years:.1f} years)")
        elif avg_tenure_years < 2.0 and len(career) >= 5:
            penalty *= 0.7
            reasons.append(f"short average tenure across {len(career)} roles ({avg_tenure_years:.1f} years avg)")
    
    # ==========================================
    # 3. Title-Skill Mismatch (Keyword Stuffer Detection)
    # ==========================================
    current_title = safe_str(profile.get("current_title", "")).lower()
    
    is_non_tech_title = any(nt in current_title for nt in NON_TECH_TITLES)
    
    if is_non_tech_title:
        # Check if they have a suspiciously high number of AI/ML skills
        ai_aliases = _get_all_taxonomy_aliases()
        ai_skill_count = 0
        for skill in skills:
            skill_name = safe_str(skill.get("name", "")).lower()
            if skill_name in ai_aliases or any(alias in skill_name for alias in ai_aliases if len(alias) >= 3):
                ai_skill_count += 1
        
        if ai_skill_count >= 5:
            penalty *= 0.2
            reasons.append(f"non-technical title ({current_title}) with {ai_skill_count} AI/ML skills listed suggests keyword stuffing")
        elif ai_skill_count >= 3:
            penalty *= 0.5
            reasons.append(f"title ({current_title}) does not align with listed AI/ML skills")
    
    # ==========================================
    # 4. Pure Research Background (No Production Evidence)
    # ==========================================
    if all_titles:
        research_count = 0
        for title in all_titles:
            if any(rt in title for rt in RESEARCH_TITLES):
                research_count += 1
        
        research_ratio = research_count / len(all_titles)
        
        if research_ratio >= 0.8:
            # Check if there's ANY production evidence in descriptions
            has_production_evidence = any(
                term in combined_career_text for term in PRODUCTION_TERMS
            )
            
            if not has_production_evidence:
                penalty *= 0.5
                reasons.append("primarily research roles with no evidence of production deployment")
            else:
                # Research-heavy but with some production signals - mild concern
                penalty *= 0.8
                reasons.append("primarily research background, though some production evidence found")
    
    # ==========================================
    # 5. No Production/Engineering Terms At All
    # ==========================================
    # Even non-research candidates should show production evidence for this JD
    if not reasons or "research" not in " ".join(reasons):
        has_any_production = any(
            term in combined_career_text for term in PRODUCTION_TERMS
        )
        has_engineering_title = any(
            t in combined_career_text for t in ["engineer", "developer", "architect", "sre", "devops"]
        )
        
        if not has_any_production and not has_engineering_title:
            penalty *= 0.8
            reasons.append("limited evidence of production engineering in career history")
    
    return max(0.05, penalty), reasons
