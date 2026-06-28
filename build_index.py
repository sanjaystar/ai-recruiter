"""
build_index.py
Offline preprocessing script to build the dense semantic-search artifacts.
Downloads the SentenceTransformer and builds the dense embedding index plus the
candidate-id ordering. Relevance is purely semantic (see src/semantic_relevance.py).
"""
import os
import json
import argparse
import numpy as np
from sentence_transformers import SentenceTransformer

from src.loader import stream_candidates
from src.schema_analyzer import get_candidate_id, safe_str, get_profile, get_skills, get_career_history

def concat_candidate_text(candidate: dict) -> str:
    """Concatenates key textual fields for dense retrieval."""
    profile = get_profile(candidate)
    skills = get_skills(candidate)
    career = get_career_history(candidate)
    
    parts = []
    
    current_title = safe_str(profile.get("current_title", ""))
    parts.extend([current_title] * 3) 
    
    parts.append(safe_str(profile.get("headline", "")))
    parts.append(safe_str(profile.get("summary", "")))
    
    for s in skills:
        parts.append(safe_str(s.get("name", "")))
        
    for c in career:
        parts.append(safe_str(c.get("title", "")))
        parts.append(safe_str(c.get("description", "")))

    return " ".join([p for p in parts if p]).lower()

def concat_role_text(role: dict) -> str:
    """Per-role text (title + description) used for semantic career relevance."""
    return f"{safe_str(role.get('title', ''))}. {safe_str(role.get('description', ''))}".lower()

def main():
    parser = argparse.ArgumentParser(description="Build offline dense semantic indexes.")
    parser.add_argument("--candidates", type=str, default="candidates.jsonl", help="Path to candidates file")
    parser.add_argument("--out-embed", type=str, default="embeddings.npy", help="Output path for embeddings")
    parser.add_argument("--out-ids", type=str, default="candidate_ids.json", help="Output path for candidate id ordering")
    parser.add_argument("--out-career", type=str, default="career_embeddings.npy", help="Output path for per-role career embeddings")
    parser.add_argument("--out-career-counts", type=str, default="career_role_counts.json", help="Output path for per-candidate role counts")
    args = parser.parse_args()

    print("Starting Offline Preprocessing Phase...")
    
    candidates_path = args.candidates
    if not os.path.exists(candidates_path) and os.path.exists(candidates_path + ".gz"):
        candidates_path = candidates_path + ".gz"
        
    if not os.path.exists(candidates_path):
        print(f"Error: Candidate file '{candidates_path}' not found.")
        return

    stats = {"loaded": 0, "filtered": 0, "errors": {}}
    candidate_ids = []
    texts = []
    role_texts = []      # flattened per-role text across all candidates
    role_counts = []     # number of roles per candidate, aligned to candidate_ids

    print(f"Extracting text from {candidates_path}...")
    for candidate in stream_candidates(candidates_path, stats):
        cid = get_candidate_id(candidate)
        if not cid:
            continue
        text = concat_candidate_text(candidate)
        candidate_ids.append(cid)
        texts.append(text)
        career = get_career_history(candidate)
        role_counts.append(len(career))
        for role in career:
            role_texts.append(concat_role_text(role))

    print(f"Extracted {len(texts)} candidates, {len(role_texts)} career roles.")

    with open(args.out_ids, "w") as f:
        json.dump(candidate_ids, f)
    print(f"Saved {args.out_ids}")

    with open(args.out_career_counts, "w") as f:
        json.dump(role_counts, f)
    print(f"Saved {args.out_career_counts}")

    model_name = "all-MiniLM-L6-v2"
    model_path = f"./models/{model_name}"
    if os.path.exists(model_path):
        print(f"Loading SentenceTransformer model from {model_path}...")
        model = SentenceTransformer(model_path)
    else:
        print("Downloading & caching SentenceTransformer model locally...")
        model = SentenceTransformer(model_name)
        os.makedirs("models", exist_ok=True)
        model.save(model_path)

    print("Building Dense Profile Embeddings (this may take 10-20 minutes)...")
    embeddings = model.encode(texts, batch_size=256, show_progress_bar=True, normalize_embeddings=True)
    np.save(args.out_embed, embeddings)
    print(f"Saved {args.out_embed}")

    print("Building Per-Role Career Embeddings...")
    career_embeddings = model.encode(role_texts, batch_size=256, show_progress_bar=True, normalize_embeddings=True)
    np.save(args.out_career, career_embeddings)
    print(f"Saved {args.out_career}")
    print("Offline Preprocessing Complete!")

if __name__ == "__main__":
    main()
