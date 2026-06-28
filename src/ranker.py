"""Apply JD weights to normalized features and produce the Top 100 ranking."""

import pandas as pd
import numpy as np

# Raw component columns and the JD-weight key each maps to.
_COMPONENTS = {
    "skill_score": "skill_weight",
    "domain_relevance_score": "domain_weight",
    "behavior_score": "behavior_weight",
    "career_score": "career_weight",
    "trust_score": "trust_weight",
}
_DEFAULT_WEIGHTS = {
    "skill_weight": 0.30, "domain_weight": 0.25, "behavior_weight": 0.20,
    "career_weight": 0.15, "trust_weight": 0.10,
}

def rank_candidates(df: pd.DataFrame, jd_weights: dict, top_n: int = 100) -> pd.DataFrame:
    """Normalize each component to 0-100, apply JD weights, drop honeypots, return the top N."""
    if df.empty:
        return df

    # 1. Normalize each component to 0-100 so the weights mean what they say
    # (raw components live on very different scales).
    final = np.zeros(len(df), dtype=float)
    for col, weight_key in _COMPONENTS.items():
        vals = df[col].to_numpy(dtype=float)
        mn, mx = vals.min(), vals.max()
        norm = (vals - mn) / (mx - mn) * 100.0 if mx > mn else np.zeros_like(vals)
        final += norm * jd_weights.get(weight_key, _DEFAULT_WEIGHTS[weight_key])
    df["final_score"] = final

    # 2. Honeypots can never appear in the Top 100
    df["final_score"] = np.where(df["is_honeypot"] == True, -999.0, df["final_score"])

    # 3. Round to CSV precision before tie-breaking by candidate_id (matches validate_submission)
    df["final_score"] = np.round(df["final_score"], 5)
    df_sorted = df.sort_values(by=["final_score", "candidate_id"], ascending=[False, True])

    top_candidates = df_sorted.head(top_n).copy()
    top_candidates["rank"] = range(1, len(top_candidates) + 1)
    return top_candidates
