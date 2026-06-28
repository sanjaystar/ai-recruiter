# Intelligent Candidate Ranking System

## Approach
Ranks 100,000 candidates against a job description offline, CPU-only, in well under 5 minutes, and returns the Top 100 with a factual reasoning for each. Every candidate is scored on five components — `skill_score` (verified-depth skills cross-referenced to the JD by embedding similarity), `semantic_relevance` (cosine fit of the candidate's dense profile embedding to the JD), `career_progression` (relevant tenure and seniority trajectory, relevance judged on per-role embeddings), a `role_fit_gate` (title embeddings vs. the target role, to drop off-role profiles), and `behavior`/`trust` (Redrob recruitability signals plus a consistency check that forces honeypots out). Each component is min-max normalized to 0–100 and combined with JD-derived weights. Zero keyword matching in candidate scoring — all five scoring components use embeddings or structured signals. Keyword taxonomy is used only for JD requirement extraction and reasoning display text. Relevance is decided purely by dense embeddings, so a candidate who built a recommendation system surfaces on meaning even without buzzwords, and a profile stuffed with the right words but the wrong actual work does not.

## Requirements
- Python 3.10+
- All dependencies in requirements.txt

## Setup & Run

### Install
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Build the offline index
Requires internet to download the `all-MiniLM-L6-v2` model (cached under `models/`); takes ~10–20 minutes on CPU. Writes the runtime artifacts to the repo root: `embeddings.npy`, `career_embeddings.npy`, `candidate_ids.json`, `career_role_counts.json`.
```bash
python build_index.py --candidates candidates.jsonl
```

### Run the ranking
Runs fully offline (forces `TRANSFORMERS_OFFLINE`/`HF_HUB_OFFLINE`), CPU-only, under the 5-minute budget. Produces `Nova.csv` with the Top 100 candidates ranked 1–100, each with a `score` and a `reasoning`.
```bash
python main.py --candidates candidates.jsonl --jd job_description.txt --out Nova.csv
```

### Validate output
Check the CSV against the official format validator before submitting — it enforces exactly 100 rows, ranks 1–100 each used once, scores non-increasing by rank, and ties broken by `candidate_id` ascending.
```bash
python validate_submission.py Nova.csv
```
Prints `Submission is valid.` on success, or the specific rule violations otherwise.
