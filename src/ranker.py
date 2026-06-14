"""
ranker.py
Applies dynamic weights to normalized features to produce the final top 100 ranking.
"""

import pandas as pd
import numpy as np
from .honeypot import apply_honeypot_penalty

def rank_candidates(df: pd.DataFrame, jd_weights: dict, top_n: int = 100) -> pd.DataFrame:
    """
    Takes a DataFrame of NORMALIZED scores (0-100), applies dynamic JD weights, 
    applies honeypot penalties, and returns the top N candidates.
    Uses fully vectorized Pandas operations for maximum performance.
    """
    if df.empty:
        return df

    # 1. Apply Dynamic Weights
    w_skill = jd_weights.get("skill_weight", 0.30)
    w_domain = jd_weights.get("domain_weight", 0.25)
    w_behavior = jd_weights.get("behavior_weight", 0.20)
    w_career = jd_weights.get("career_weight", 0.15)
    w_trust = jd_weights.get("trust_weight", 0.10)
    
    # Vectorized arithmetic across all 100k rows simultaneously
    df["final_score"] = (
        (df["skill_score"] * w_skill) +
        (df["domain_relevance_score"] * w_domain) +
        (df["behavior_score"] * w_behavior) +
        (df["career_score"] * w_career) +
        (df["trust_score"] * w_trust)
    )
    
    # 2. Apply Honeypot Penalties
    # We use numpy.where to vectorize the boolean condition (orders of magnitude faster than df.apply)
    df["final_score"] = np.where(
        df["is_honeypot"] == True, 
        -999.0, 
        df["final_score"]
    )
    
    # 3. Sort and Slice
    # Tiebreaking requirement: if final scores tie, sort by skill_score descending, 
    # and finally deterministically by candidate_id ascending.
    df_sorted = df.sort_values(
        by=["final_score", "skill_score", "candidate_id"], 
        ascending=[False, False, True]
    )
    
    top_candidates = df_sorted.head(top_n).copy()
    
    # 4. Assign ranks (1 to N)
    top_candidates["rank"] = range(1, len(top_candidates) + 1)
    
    return top_candidates
