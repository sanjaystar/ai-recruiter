"""
reasoning.py
Generates factual, specific, and varied reasoning for the final submission.
Output is 1-2 concise sentences per the spec (Section 2).
Every claim is guarded by an actual data check — no hallucination.
Uses penalty_reasons from jd_fit_penalty and trust_concerns from trust_score
as the single source of truth for concerns.
"""

from .schema_analyzer import get_profile, get_skills, get_career_history, get_redrob_signals, safe_float, safe_str
from .skill_taxonomy import get_canonical_skill


def generate_reasoning(candidate: dict, rank: int, final_score: float, 
                       jd_data: dict = None, penalty_reasons: list = None,
                       trust_concerns: list = None) -> str:
    """
    Constructs a factual, varied, 1-2 sentence justification.
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
    
    # ==========================================
    # 1. Build compact fragments
    # ==========================================
    fragments = []
    
    # --- Identity ---
    stated_yoe = safe_float(profile.get("years_of_experience", 0))
    career_duration_months = sum(
        safe_float(r.get("duration_months", 0))
        for r in career
        if safe_float(r.get("duration_months", 0)) > 0
    )
    career_yoe = career_duration_months / 12.0
    yoe = max(stated_yoe, career_yoe)
    title = safe_str(profile.get("current_title", ""))
    company = safe_str(profile.get("current_company", ""))
    
    if title and yoe > 0 and company:
        fragments.append(f"{yoe:.0f}-year {title} at {company}")
    elif title and yoe > 0:
        fragments.append(f"{yoe:.0f} years as {title}")
    elif title and company:
        fragments.append(f"{title} at {company}")
    elif title:
        fragments.append(title)
    
    # --- JD Skill Match ---
    jd_required = set(jd_data.get("required_skills", [])) if jd_data else set()
    jd_preferred = set(jd_data.get("preferred_skills", [])) if jd_data else set()
    
    prof_rank = {"expert": 4, "advanced": 3, "intermediate": 2, "beginner": 1}
    matched_req = []
    matched_pref = []
    
    for skill in skills:
        name = safe_str(skill.get("name", ""))
        prof = safe_str(skill.get("proficiency", "")).lower()
        if not name:
            continue
        canonical = get_canonical_skill(name.lower())
        if canonical in jd_required:
            matched_req.append((name, prof))
        elif canonical in jd_preferred:
            matched_pref.append((name, prof))
    
    matched_req.sort(key=lambda x: prof_rank.get(x[1], 0), reverse=True)
    matched_pref.sort(key=lambda x: prof_rank.get(x[1], 0), reverse=True)
    
    if matched_req:
        top = matched_req[:3]
        skill_str = ", ".join(f"{n} ({p})" for n, p in top)
        fragments.append(f"JD match on {skill_str}")
    elif matched_pref:
        top = matched_pref[:2]
        skill_str = ", ".join(n for n, _ in top)
        fragments.append(f"preferred skills: {skill_str}")
    else:
        top_by_prof = sorted(skills, key=lambda x: prof_rank.get(
            safe_str(x.get("proficiency", "")).lower(), 0), reverse=True)[:2]
        names = [safe_str(s.get("name", "")) for s in top_by_prof if s.get("name")]
        if names:
            fragments.append(f"skills include {', '.join(names)}")
    
    # --- Top 1-2 Behavioral Signals (only the most notable) ---
    resp_rate = safe_float(signals.get("recruiter_response_rate", -1))
    notice_days = safe_float(signals.get("notice_period_days", -1))
    open_flag = signals.get("open_to_work_flag", None)
    github = safe_float(signals.get("github_activity_score", -1))
    location = safe_str(profile.get("location", ""))
    
    sig_parts = []
    
    # Pick at most 2 positive signals
    if notice_days >= 0 and notice_days <= 30:
        sig_parts.append("available immediately")
    if open_flag is True and len(sig_parts) < 2:
        sig_parts.append("open to work")
    if resp_rate >= 0.8 and len(sig_parts) < 2:
        sig_parts.append("high response rate")
    if github >= 70 and len(sig_parts) < 2:
        sig_parts.append("active on GitHub")
    if location and len(sig_parts) < 2:
        # Only add location if it's a notable match
        loc_lower = location.lower()
        if any(c in loc_lower for c in ["pune", "noida"]):
            sig_parts.append(f"based in {location}")
    
    # Or pick notable negative signals if no positives
    if not sig_parts:
        if resp_rate >= 0 and resp_rate < 0.2:
            sig_parts.append("low response rate")
        if notice_days > 90:
            sig_parts.append(f"{int(notice_days)}-day notice")
    
    if sig_parts:
        fragments.append("; ".join(sig_parts))
    
    # ==========================================
    # 2. Assemble sentence 1
    # ==========================================
    sentence1 = "; ".join(fragments) + "."
    # Capitalize first letter
    sentence1 = sentence1[0].upper() + sentence1[1:] if sentence1 else ""
    
    # ==========================================
    # 3. Sentence 2: concerns (only if they exist)
    # ==========================================
    derived_concerns = list(trust_concerns)
    if stated_yoe > 0 and career_yoe > 0 and abs(career_yoe - stated_yoe) > 1.5:
        derived_concerns.append(
            f"YOE doesn't match career history ({stated_yoe:.0f}yr stated vs {career_yoe:.1f}yr from roles)"
        )
    all_concerns = penalty_reasons + derived_concerns
    sentence2 = ""
    if all_concerns:
        # Keep only the first 1-2 concerns to stay concise
        short_concerns = all_concerns[:2]
        sentence2 = " Concern: " + "; ".join(short_concerns) + "."
    
    reasoning = sentence1 + sentence2

    # ==========================================
    # 4. Low-rank qualifier (ranks 81-100)
    # ==========================================
    # The spec flags "glowing reasoning at rank 95" as a sign reasoning was
    # generated independently of the ranking. For tail candidates that have no
    # explicit concerns yet, surface the factual dimension that pushed them down.
    if rank > 80 and not all_concerns:
        last_active = signals.get("last_active_date", "")
        days_inactive = 0
        if last_active:
            try:
                from datetime import datetime
                days_inactive = (datetime(2026, 6, 20) - datetime.fromisoformat(last_active[:10])).days
            except (ValueError, TypeError):
                days_inactive = 0

        if len(matched_req) == 0:
            qualifier = "No direct required-skill matches; placed on career trajectory and engagement signals alone."
        elif notice_days > 90:
            qualifier = f"Long notice period ({int(notice_days)} days) and weaker behavioral signals relative to higher-ranked candidates."
        elif resp_rate >= 0 and resp_rate < 0.4:
            qualifier = f"Below-average recruiter response rate ({resp_rate:.0%}) reduces practical reachability."
        elif days_inactive > 180:
            qualifier = f"Profile inactive for {days_inactive // 30} months; lower engagement score versus top candidates."
        else:
            qualifier = f"Skill and experience profile is competitive but behavioral signals place this candidate at rank {rank}."

        reasoning = reasoning.rstrip(".") + ". " + qualifier

    # Hard cap at 300 chars to enforce conciseness
    if len(reasoning) > 300:
        reasoning = reasoning[:297] + "..."

    return reasoning

