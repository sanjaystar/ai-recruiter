"""Write the Top 100 ranking to CSV (candidate_id, rank, score, reasoning)."""

import pandas as pd

def generate_submission(top_100_df: pd.DataFrame, out_path: str = "Nova.csv"):
    if top_100_df.empty:
        print("Warning: DataFrame is empty. Cannot generate valid submission.")
        return

    df = top_100_df.copy()
    df = df.rename(columns={"final_score": "score"})
    df["score"] = df["score"].apply(lambda x: round(float(x), 5))
    final_df = df[["candidate_id", "rank", "score", "reasoning"]]
    final_df.to_csv(out_path, index=False, encoding='utf-8')
