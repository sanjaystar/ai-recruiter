"""
honeypot.py
Centralizes honeypot punishment logic.
"""

def apply_honeypot_penalty(final_score: float, is_honeypot: bool) -> float:
    """
    If a candidate is flagged as a honeypot, this effectively disqualifies them 
    by driving their final score into deep negative territory, ensuring they 
    will never appear in the Top 100 ranking.
    """
    if is_honeypot:
        return -999.0
    return final_score
