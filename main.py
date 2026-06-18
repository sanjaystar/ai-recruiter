"""
main.py
Coordinates the online sandbox execution. Loads static artifacts and processes candidates.
"""

import time
import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.loader import stream_candidates
from src.schema_analyzer import get_candidate_id
from src.jd_parser import parse_jd
from src.skill_score import calculate_skill_score
from src.career_score import calculate_career_score
from src.behavior_score import calculate_behavior_score
from src.trust_score import calculate_trust
from src.jd_fit_penalty import calculate_jd_fit_penalty
from src.hybrid_ranker import calculate_hybrid_relevance
from src.ranker import rank_candidates
from src.reasoning import generate_reasoning
from src.submission import generate_submission
from src.features import CandidateScore

def get_jd_text(jd_path: str) -> str:
    with open(jd_path, 'r', encoding='utf-8') as f:
        return f.read().lower()

def main():
    parser = argparse.ArgumentParser(description="Candidate Ranking Engine")
    parser.add_argument("--candidates", type=str, default="candidates.jsonl", help="Path to candidates file")
    parser.add_argument("--jd", type=str, default="job_description.txt", help="Path to job description")
    parser.add_argument("--out", type=str, default="submission.csv", help="Output CSV path")
    args = parser.parse_args()

    start_time = time.time()
    
    # Safety Check for Indexes
    if not os.path.exists("embeddings.npy") or not os.path.exists("bm25_index.pkl") or not os.path.exists("models/all-MiniLM-L6-v2"):
        print("Indexes not found. Run: python build_index.py")
        sys.exit(1)
    
    jd_path = args.jd
    candidates_path = args.candidates 
    if not os.path.exists(candidates_path) and os.path.exists(candidates_path + ".gz"):
        candidates_path = candidates_path + ".gz"
        
    print(f"Parsing JD from: {jd_path}")
    jd_data = parse_jd(jd_path)
    jd_text = get_jd_text(jd_path)
    
    # 1. Load Precomputed Artifacts
    print("Loading binary artifacts (BM25, Embeddings, Models)...")
    with open("bm25_index.pkl", "rb") as f:
        bm25_data = pickle.load(f)
    vectorizer = bm25_data["vectorizer"]
    candidate_bm25_matrix = bm25_data["matrix"]
    candidate_ids = bm25_data["candidate_ids"]
    
    candidate_embeddings = np.load("embeddings.npy")
        
    model = SentenceTransformer("./models/all-MiniLM-L6-v2")
    
    # 2. Process JD
    print("Generating JD semantic and sparse vectors...")
    jd_bm25_vector = vectorizer.transform([jd_text])
    jd_embedding = model.encode([jd_text], normalize_embeddings=True) 
    
    # 3. Calculate Hybrid Relevance
    print("Executing Hybrid Search...")
    hybrid_scores = calculate_hybrid_relevance(
        jd_embedding=jd_embedding,
        jd_bm25_vector=jd_bm25_vector,
        candidate_embeddings=candidate_embeddings,
        candidate_bm25_matrix=candidate_bm25_matrix
    )
    
    hybrid_dict = {cid: score for cid, score in zip(candidate_ids, hybrid_scores)}
    
    # 4. Stream Candidates
    print(f"Streaming candidates from: {candidates_path}")
    taxonomy_cache = {}
    features_list = []
    raw_cache = {}
    penalty_reasons_cache = {}
    trust_concerns_cache = {}
    
    stats = {"loaded": 0, "filtered": 0, "errors": {}}
    for count, candidate in enumerate(stream_candidates(candidates_path, stats)):
        cid = get_candidate_id(candidate)
        if not cid or cid not in hybrid_dict:
            continue
            
        s_score = calculate_skill_score(candidate, jd_data, taxonomy_cache)
        d_score = hybrid_dict[cid]
        c_score = calculate_career_score(candidate, jd_data)
        b_score = calculate_behavior_score(candidate, jd_data)
        t_score, is_honeypot, trust_concerns = calculate_trust(candidate)
        jd_penalty, penalty_reasons = calculate_jd_fit_penalty(candidate, jd_data)
        
        c_feat = CandidateScore(
            candidate_id=cid,
            skill_score=s_score,
            domain_relevance_score=d_score,
            behavior_score=b_score,
            career_score=c_score,
            trust_score=t_score,
            jd_fit_penalty=jd_penalty,
            is_honeypot=is_honeypot
        )
        features_list.append(c_feat.to_dict())
        raw_cache[cid] = candidate
        penalty_reasons_cache[cid] = penalty_reasons
        trust_concerns_cache[cid] = trust_concerns
        
    if not features_list:
        print("CRITICAL ERROR: No candidates were parsed.")
        sys.exit(1)

    # 5. Ranking
    print("Building Vectorized Pandas DataFrame...")
    df = pd.DataFrame(features_list)
    
    print("Ranking Candidates...")
    top_100 = rank_candidates(df, jd_data.get("weights", {}), top_n=100)
    
    print("Generating Factual Reasoning...")
    reasonings = []
    for _, row in top_100.iterrows():
        cid = row["candidate_id"]
        rank = row["rank"]
        final_score = row["final_score"]
        raw_cand = raw_cache.get(cid, {})
        p_reasons = penalty_reasons_cache.get(cid, [])
        t_concerns = trust_concerns_cache.get(cid, [])
        reasoning = generate_reasoning(raw_cand, rank, final_score, jd_data, p_reasons, t_concerns)
        reasonings.append(reasoning)
        
    top_100["reasoning"] = reasonings
    
    print(f"Exporting {args.out}...")
    generate_submission(top_100, args.out)
    
    elapsed = time.time() - start_time
    print(f"==================================================")
    print(f"SUCCESS: Ranked {len(df)} candidates in {elapsed:.2f} seconds.")
    print(f"Submission saved to {args.out}")
    print(f"==================================================")

if __name__ == "__main__":
    main()

