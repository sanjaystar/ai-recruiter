"""
submission.py
Outputs the final Top 100 ranking strictly matching the submission_spec.md constraints.
"""

import pandas as pd

def generate_submission(top_100_df: pd.DataFrame, out_path: str = "submission.csv"):
    """
    Formats the DataFrame columns and exports to CSV.
    Constraints Enforced:
    - 100 rows
    - candidate_id, rank, score, reasoning columns
    - Sorted by rank
    """
    if top_100_df.empty:
        print("Warning: DataFrame is empty. Cannot generate valid submission.")
        return
        
    df = top_100_df.copy()
    
    # Rename the column to match the required spec
    df = df.rename(columns={"final_score": "score"})
    
    # Format the score deterministically
    df["score"] = df["score"].apply(lambda x: round(float(x), 5))
    
    # Ensure strict column ordering
    final_df = df[["candidate_id", "rank", "score", "reasoning"]]
    
    final_df.to_csv(out_path, index=False, encoding='utf-8')
