"""
reasoning.py
Generates factual, specific, and varied reasoning for the final submission.
Every claim is guarded by an actual data check — no hallucination.
Uses penalty_reasons from jd_fit_penalty and trust_concerns from trust_score
as the single source of truth for concerns.
"""

from .schema_analyzer import get_profile, get_skills, get_career_history, get_redrob_signals, safe_float, safe_str

def generate_reasoning(candidate: dict, rank: int, final_score: float, 
                       jd_data: dict = None, penalty_reasons: list = None,
                       trust_concerns: list = None) -> str:
    """
    Constructs a factual, varied justification string.
    Every field is checked before being referenced — no assumptions.
    """
    profile = get_profile(candidate)
    skills = get_skills(candidate)
    career = get_career_history(candidate)
    signals = get_redrob_signals(candidate)
    
    if penalty_reasons is None:
        penalty_reasons = []
    if trust_concerns is None:
        trust_concerns = []
    
    parts = []
    
    # ==========================================
    # 1. Core Identity (always included — guarded)
    # ==========================================
    yoe = safe_float(profile.get("years_of_experience", 0))
    title = safe_str(profile.get("current_title", ""))
    company = safe_str(profile.get("current_company", ""))
    location = safe_str(profile.get("location", ""))
    
    identity_parts = []
    if title and yoe > 0:
        identity_parts.append(f"{yoe:.0f} years experience as {title}")
    elif title:
        identity_parts.append(f"Currently {title}")
    elif yoe > 0:
        identity_parts.append(f"{yoe:.0f} years of professional experience")
    
    if company:
        identity_parts.append(f"at {company}")
    
    if identity_parts:
        parts.append(", ".join(identity_parts) + ".")
    
    # ==========================================
    # 2. JD-Relevant Skills (varied based on what exists)
    # ==========================================
    jd_required = set(jd_data.get("required_skills", [])) if jd_data else set()
    jd_preferred = set(jd_data.get("preferred_skills", [])) if jd_data else set()
    
    # Find skills that match JD requirements, ordered by proficiency then duration
    matched_required = []
    matched_preferred = []
    
    for skill in skills:
        name = safe_str(skill.get("name", ""))
        prof = safe_str(skill.get("proficiency", "")).lower()
        dur = safe_float(skill.get("duration_months", 0))
        if not name:
            continue
        
        name_lower = name.lower()
        # Check if this skill name or its domain relates to JD requirements
        # We do a simple substring check against JD skill domains
        is_req_match = any(name_lower in d.lower() or d.lower() in name_lower for d in jd_required)
        is_pref_match = any(name_lower in d.lower() or d.lower() in name_lower for d in jd_preferred)
        
        if is_req_match:
            matched_required.append((name, prof, dur))
        elif is_pref_match:
            matched_preferred.append((name, prof, dur))
    
    # Sort by proficiency rank then duration
    prof_rank = {"expert": 4, "advanced": 3, "intermediate": 2, "beginner": 1}
    matched_required.sort(key=lambda x: (prof_rank.get(x[1], 0), x[2]), reverse=True)
    matched_preferred.sort(key=lambda x: (prof_rank.get(x[1], 0), x[2]), reverse=True)
    
    # Build the skills sentence — vary structure based on match quality
    if matched_required:
        top_skills = matched_required[:3]
        skill_strs = []
        for name, prof, dur in top_skills:
            if prof in ("expert", "advanced") and dur > 24:
                skill_strs.append(f"{name} ({prof}, {dur // 12}+ years)")
            elif prof in ("expert", "advanced"):
                skill_strs.append(f"{name} ({prof})")
            else:
                skill_strs.append(name)
        parts.append(f"Matches core JD requirements with {', '.join(skill_strs)}.")
    elif matched_preferred:
        top_skills = matched_preferred[:2]
        skill_names = [s[0] for s in top_skills]
        parts.append(f"Matches preferred skills: {', '.join(skill_names)}.")
    else:
        # Fallback: mention top skills by proficiency without JD-match claim
        top_by_prof = sorted(skills, key=lambda x: (
            prof_rank.get(safe_str(x.get("proficiency", "")).lower(), 0),
            safe_float(x.get("duration_months", 0))
        ), reverse=True)[:2]
        if top_by_prof:
            names = [safe_str(s.get("name", "")) for s in top_by_prof if s.get("name")]
            if names:
                parts.append(f"Strongest skills include {', '.join(names)}.")
    
    # ==========================================
    # 3. Career Evidence (only if career data exists)
    # ==========================================
    if career and rank <= 50:
        # For top-50, mention relevant career evidence
        relevant_roles = []
        for role in career:
            role_title = safe_str(role.get("title", ""))
            role_company = safe_str(role.get("company", ""))
            role_desc = safe_str(role.get("description", "")).lower()
            dur_months = safe_float(role.get("duration_months", 0))
            
            # Check if role description mentions retrieval/ranking/ML production terms
            relevance_terms = ["retrieval", "ranking", "search", "recommendation",
                             "embedding", "vector", "ml", "machine learning",
                             "nlp", "production", "deploy"]
            
            if any(term in role_desc for term in relevance_terms) and dur_months > 12:
                relevant_roles.append((role_title, role_company, dur_months))
        
        if relevant_roles:
            best = relevant_roles[0]
            r_title, r_company, r_dur = best
            dur_years = r_dur / 12.0
            if r_title and r_company:
                parts.append(f"Relevant experience includes {dur_years:.1f} years as {r_title} at {r_company}.")
            elif r_title:
                parts.append(f"Prior relevant role as {r_title} ({dur_years:.1f} years).")
    
    # ==========================================
    # 4. Behavioral Signals (varied by what's notable)
    # ==========================================
    signal_notes = []
    
    resp_rate = safe_float(signals.get("recruiter_response_rate", -1))
    notice_days = safe_float(signals.get("notice_period_days", -1))
    open_flag = signals.get("open_to_work_flag", None)
    last_active = signals.get("last_active_date", "")
    github = safe_float(signals.get("github_activity_score", -1))
    interview_rate = safe_float(signals.get("interview_completion_rate", -1))
    
    # Only mention signals that are notable (very good or very concerning)
    if resp_rate >= 0.8:
        signal_notes.append("high recruiter response rate")
    elif resp_rate >= 0 and resp_rate < 0.2:
        signal_notes.append("very low recruiter response rate")
    
    if notice_days >= 0 and notice_days <= 30:
        signal_notes.append("immediate availability (≤30 day notice)")
    elif notice_days > 90:
        signal_notes.append(f"extended notice period ({int(notice_days)} days)")
    
    if open_flag is True:
        signal_notes.append("actively open to work")
    
    if github >= 70:
        signal_notes.append("strong GitHub activity")
    
    if interview_rate >= 0 and interview_rate >= 0.9:
        signal_notes.append("excellent interview follow-through")
    elif interview_rate >= 0 and interview_rate < 0.3:
        signal_notes.append("low interview completion rate")
    
    if signal_notes:
        # Vary the phrasing based on whether signals are positive or mixed
        positive_count = sum(1 for s in signal_notes if "low" not in s and "extended" not in s)
        if positive_count == len(signal_notes):
            parts.append(f"Positive engagement signals: {'; '.join(signal_notes)}.")
        elif positive_count == 0:
            parts.append(f"Engagement concerns: {'; '.join(signal_notes)}.")
        else:
            parts.append(f"Mixed signals: {'; '.join(signal_notes)}.")
    
    # ==========================================
    # 5. Location Context (if relevant to rank)
    # ==========================================
    if location and rank <= 30:
        work_mode = safe_str(signals.get("preferred_work_mode", ""))
        if work_mode:
            parts.append(f"Based in {location}, prefers {work_mode}.")
        else:
            parts.append(f"Based in {location}.")
    
    # ==========================================
    # 6. Computed Concerns (from jd_fit_penalty + trust_score)
    # ==========================================
    all_concerns = penalty_reasons + trust_concerns
    
    if all_concerns:
        # Capitalize first concern, join with semicolons
        concern_str = "; ".join(all_concerns)
        parts.append(f"Concerns: {concern_str}.")
    
    # ==========================================
    # 7. Rank-Appropriate Framing
    # ==========================================
    # Add a rank-contextual prefix that matches the score
    if rank <= 5:
        prefix = "Exceptional match."
    elif rank <= 15:
        prefix = "Strong match."
    elif rank <= 40:
        prefix = "Good fit."
    elif rank <= 70:
        prefix = "Moderate fit."
    else:
        prefix = "Borderline candidate."
    
    # Assemble final reasoning
    reasoning = f"{prefix} {' '.join(parts)}"
    
    # Safety: ensure we don't exceed a reasonable length
    if len(reasoning) > 500:
        reasoning = reasoning[:497] + "..."
    
    return reasoning
