# AI Recruiter - Hybrid Semantic Ranking Engine

This repository contains a state-of-the-art **Hybrid Semantic Retrieval** and **Recruiter Intelligence** engine designed to rank 100,000 candidates against a Job Description in under 5 minutes without network or GPU access.

## Architecture

To satisfy strict sandbox constraints (`<5 min`, `<16GB RAM`, `CPU only`), the architecture is split asymmetrically:

1. **Offline Preprocessing (`build_index.py`)**
   - Parses the massive `candidates.jsonl` dataset.
   - Generates a **Sparse Matrix** (BM25 / sublinear TF-IDF) to capture exact technical jargon.
   - Downloads `all-MiniLM-L6-v2` locally.
   - Generates a **Dense Embedding Matrix** for all candidates to capture latent semantic meaning (e.g., mapping "LLMs" to "Foundational Models").
   - Artifacts (`embeddings.npy`, `bm25_index.pkl`) are saved locally.

2. **Online Ranking (`main.py`)**
   - Instantaneously loads the offline binary artifacts into memory.
   - Embeds only the single Job Description text.
   - Executes massive parallel NumPy dot-products across 100,000 candidates in `<50ms`.
   - Merges vector scores with deterministic heuristics (Relevant Years of Experience, Redrob Behavioral Signals, Honeypot Trust Checks).
   - Generates final rankings and explainability text.

## Installation

Ensure you have Python 3.9+ installed.

```bash
# 1. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

## Reproduction Workflow

Follow these exact steps to reproduce the outputs.

### Step 1: Build the Offline Indexes
*Note: This step requires internet access to download the SentenceTransformer model and takes approximately 10-20 minutes depending on your CPU.*

```bash
python build_index.py \
    --candidates candidates.jsonl \
    --out-embed embeddings.npy \
    --out-bm25 bm25_index.pkl
```

### Step 2: Execute Sandbox Ranking
*Note: This step perfectly mimics the Hackathon Sandbox. It requires NO internet, uses NO GPU, and executes in ~4 seconds.*

```bash
python main.py \
    --candidates candidates.jsonl \
    --jd job_description.txt \
    --out submission.csv
```

## Expected Outputs
The final execution will generate `submission.csv` containing the Top 100 ranked candidates, sorted 1 to 100, with a dynamically generated `reasoning` string explaining *why* they were chosen based on both semantic match and behavioral intelligence.
