"""
build_index.py
Offline preprocessing script to build the Hybrid Semantic Search artifacts.
Downloads the SentenceTransformer and builds dense and sparse indices.
"""
import os
import pickle
import argparse
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

from src.loader import stream_candidates
from src.schema_analyzer import get_candidate_id, safe_str, get_profile, get_skills, get_career_history

def concat_candidate_text(candidate: dict) -> str:
    """Concatenates key textual fields for dense/sparse retrieval."""
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

def main():
    parser = argparse.ArgumentParser(description="Build offline dense and sparse indexes.")
    parser.add_argument("--candidates", type=str, default="candidates.jsonl", help="Path to candidates file")
    parser.add_argument("--out-embed", type=str, default="embeddings.npy", help="Output path for embeddings")
    parser.add_argument("--out-bm25", type=str, default="bm25_index.pkl", help="Output path for BM25 index")
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
    
    print(f"Extracting text from {candidates_path}...")
    for candidate in stream_candidates(candidates_path, stats):
        cid = get_candidate_id(candidate)
        if not cid:
            continue
        text = concat_candidate_text(candidate)
        candidate_ids.append(cid)
        texts.append(text)
        
    print(f"Extracted {len(texts)} candidates.")
    
    print("Building BM25 (TF-IDF) Sparse Index...")
    vectorizer = TfidfVectorizer(stop_words='english', max_features=50000, sublinear_tf=True)
    sparse_matrix = vectorizer.fit_transform(texts)
    
    with open(args.out_bm25, "wb") as f:
        pickle.dump({
            "vectorizer": vectorizer,
            "matrix": sparse_matrix,
            "candidate_ids": candidate_ids
        }, f)
    print(f"Saved {args.out_bm25}")
        
    print("Downloading & caching SentenceTransformer model locally...")
    model_name = "all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)
    os.makedirs("models", exist_ok=True)
    model.save(f"models/{model_name}")
    
    print("Building Dense Embeddings (this may take 10-20 minutes)...")
    embeddings = model.encode(texts, batch_size=256, show_progress_bar=True, normalize_embeddings=True)
    
    np.save(args.out_embed, embeddings)
    print(f"Saved {args.out_embed}")
    print("Offline Preprocessing Complete!")

if __name__ == "__main__":
    main()
