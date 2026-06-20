"""
ranker.py
Applies dynamic weights to normalized features, applies JD-fit penalties,
and produces the final top 100 ranking.
"""

import pandas as pd
import numpy as np

def rank_candidates(df: pd.DataFrame, jd_weights: dict, top_n: int = 100) -> pd.DataFrame:
    """
    Takes a DataFrame of NORMALIZED scores (0-100), applies dynamic JD weights, 
    applies JD-fit penalty multiplier, applies honeypot penalties, 
    and returns the top N candidates.
    Uses fully vectorized Pandas operations for maximum performance.
    """
    if df.empty:
        return df

    # Normalize skill_score to 0-100 so it's on the same scale as all other
    # component scores before weighting. Without this, behavior_score (0-100)
    # silently outweighs skill_score (0-~50) despite skill having a higher weight.
    skill_max = df["skill_score"].max()
    if skill_max > 0:
        df["skill_score"] = (df["skill_score"] / skill_max) * 100.0

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
    
    # 2. Apply JD-Fit Penalty Multiplier (compounding penalties from jd_fit_penalty.py)
    df["final_score"] = df["final_score"] * df["jd_fit_penalty"]
    
    # 3. Apply Honeypot Penalties
    # We use numpy.where to vectorize the boolean condition (orders of magnitude faster than df.apply)
    df["final_score"] = np.where(
        df["is_honeypot"] == True, 
        -999.0, 
        df["final_score"]
    )
    
    # 4. Sort and Slice
    # Tiebreaking requirement: if final scores tie, sort by skill_score descending, 
    # and finally deterministically by candidate_id ascending.
    df_sorted = df.sort_values(
        by=["final_score", "skill_score", "candidate_id"], 
        ascending=[False, False, True]
    )
    
    top_candidates = df_sorted.head(top_n).copy()
    
    # 5. Assign ranks (1 to N)
    top_candidates["rank"] = range(1, len(top_candidates) + 1)
    
    return top_candidates

