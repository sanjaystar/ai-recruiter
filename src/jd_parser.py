"""
jd_parser.py
Rule-based deterministic parser to extract skills, signals, and dynamic weights from any Job Description.
"""

import re
import os
from .skill_taxonomy import TAXONOMY

# How the JD names the domains it explicitly rejects (see "do NOT want" section).
_NEG_JD_TRIGGERS = {
    "COMPUTER_VISION": ["computer vision", "image classification", "image processing"],
    "SPEECH": ["speech", "text-to-speech", "speech recognition"],
    "ROBOTICS": ["robotics", "robot"],
}

def parse_jd(filepath: str) -> dict:
    """
    Parses a job description text file deterministically.
    
    Returns:
    {
      "required_skills": list[str],   # JD domains marked required
      "preferred_skills": list[str],  # JD domains marked nice-to-have
      "negative_domains": list[str],  # explicitly-rejected domains (CV/speech/robotics)
      "weights": dict,                # structural component weights summing to 1.0
    }
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"JD file not found: {filepath}")
        
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    text_lower = text.lower()
    
    # 1. Determine Dynamic Weights via Term Frequency
    domain_counts = {domain: 0 for domain in TAXONOMY.keys()}
    for domain, aliases in TAXONOMY.items():
        for alias in aliases:
            # Use regex word boundaries for accurate counting
            matches = len(re.findall(rf'\b{re.escape(alias)}\b', text_lower))
            domain_counts[domain] += matches

    # Default structural weights (total = 1.0)
    base_weights = {
        "skill_weight": 0.30,
        "domain_weight": 0.25,
        "behavior_weight": 0.20,
        "career_weight": 0.15,
        "trust_weight": 0.10
    }
    
    # Dynamically adjust structural weights based on JD text emphasis
    if "culture" in text_lower or "fit" in text_lower or "behavior" in text_lower:
        base_weights["behavior_weight"] += 0.05
        base_weights["skill_weight"] -= 0.05
        
    if "production" in text_lower or "scale" in text_lower or "deployment" in text_lower:
        base_weights["domain_weight"] += 0.05
        base_weights["behavior_weight"] -= 0.05

    # Normalize weights to exactly 1.0 just in case
    total_weight = sum(base_weights.values())
    for k in base_weights:
        base_weights[k] = round(base_weights[k] / total_weight, 3)

    # 2. Extract Required, Preferred, and Negative Signals using heuristics
    required_keywords = ["require", "must", "need", "absolutely", "essential"]
    preferred_keywords = ["prefer", "nice to have", "bonus", "like you to have"]
    negative_keywords = ["not want", "disqualif", "reject", "do not", "will not"]

    required_skills = set()
    preferred_skills = set()
    negative_signals = []

    paragraphs = text_lower.split('\n')
    current_mode = "neutral" 
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
            
        # Detect section boundaries (usually short header lines)
        if len(para.split()) < 15:
            if any(k in para for k in required_keywords):
                current_mode = "required"
                continue
            elif any(k in para for k in preferred_keywords):
                current_mode = "preferred"
                continue
            elif any(k in para for k in negative_keywords):
                current_mode = "negative"
                continue
                
        # Extract skills based on the active section mode
        for domain, aliases in TAXONOMY.items():
            for alias in aliases:
                if re.search(rf'\b{re.escape(alias)}\b', para):
                    if current_mode == "required":
                        required_skills.add(domain)
                    elif current_mode == "preferred":
                        preferred_skills.add(domain)
                        
        # Extract negative sentences to be used as signals downstream
        if current_mode == "negative" and len(para) > 10:
            negative_signals.append(para)

    # Fallback: if section parsing failed to find required skills,
    # use the top most frequent domains from our term frequency counter.
    if not required_skills:
        sorted_domains = sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)
        # Take domains with at least 2 mentions as required
        required_skills = {d for d, count in sorted_domains if count >= 2}

    # 3. Use the previously-discarded negative section: which rejected technical
    # domains does this JD actually call out?
    neg_text = " ".join(negative_signals)
    negative_domains = [
        domain for domain, triggers in _NEG_JD_TRIGGERS.items()
        if any(re.search(rf"\b{re.escape(t)}\b", neg_text) for t in triggers)
    ]

    return {
        "required_skills": sorted(list(required_skills)),
        "preferred_skills": sorted(list(preferred_skills)),
        "negative_domains": negative_domains,
        "weights": base_weights,
    }
