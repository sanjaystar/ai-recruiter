"""Coordinates sandbox execution: loads artifacts, scores candidates, writes the ranking."""

import os
# Force HF/Transformers offline before importing sentence-transformers (must precede import).
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import time
import sys
import json
import argparse
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.loader import stream_candidates
from src.schema_analyzer import get_candidate_id
from src.embed_io import load_embeddings, embeddings_exist
from src.jd_parser import parse_jd
from src.skill_score import calculate_skill_score, build_skill_matcher
from src.career_score import calculate_career_score, build_career_anchors, role_relevance_from_embeddings
from src.behavior_score import calculate_behavior_score
from src.trust_score import calculate_trust
from src.semantic_relevance import calculate_domain_relevance, RoleFitScorer
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
    parser.add_argument("--out", type=str, default="Nova.csv", help="Output CSV path")
    args = parser.parse_args()

    start_time = time.time()

    required_files = ["candidate_ids.json", "career_role_counts.json", "models/all-MiniLM-L6-v2"]
    required_embeds = ["embeddings.npy", "career_embeddings.npy"]
    if not (all(os.path.exists(p) for p in required_files)
            and all(embeddings_exist(p) for p in required_embeds)):
        print("Indexes not found. Run: python build_index.py")
        sys.exit(1)

    jd_path = args.jd
    candidates_path = args.candidates
    if not os.path.exists(candidates_path) and os.path.exists(candidates_path + ".gz"):
        candidates_path = candidates_path + ".gz"

    jd_data = parse_jd(jd_path)
    jd_text = get_jd_text(jd_path)

    # 1. Load artifacts
    candidate_embeddings = load_embeddings("embeddings.npy")
    with open("candidate_ids.json", "r") as f:
        candidate_ids = json.load(f)
    career_embeddings = load_embeddings("career_embeddings.npy")
    with open("career_role_counts.json", "r") as f:
        career_role_counts = json.load(f)

    model = SentenceTransformer("./models/all-MiniLM-L6-v2")

    # 2. Process JD
    jd_embedding = model.encode([jd_text], normalize_embeddings=True)

    skill_matcher = build_skill_matcher(model, jd_data)
    role_scorer = RoleFitScorer(model)

    # 3. Domain relevance
    domain_scores = calculate_domain_relevance(
        jd_embedding=jd_embedding,
        candidate_embeddings=candidate_embeddings,
    )

    domain_dict = {cid: score for cid, score in zip(candidate_ids, domain_scores)}

    # 3b. Per-role career relevance: one vectorized pass, then slice back per candidate.
    career_anchors = build_career_anchors(model, jd_data)
    role_relevant = role_relevance_from_embeddings(career_embeddings, career_anchors)
    role_rel_by_cid = {}
    pos = 0
    for cid, cnt in zip(candidate_ids, career_role_counts):
        role_rel_by_cid[cid] = role_relevant[pos:pos + cnt]
        pos += cnt

    # 4. Pass 1: score everyone, keep only compact feature rows
    features_list = []

    stats = {"loaded": 0, "filtered": 0, "errors": {}}
    for candidate in stream_candidates(candidates_path, stats):
        cid = get_candidate_id(candidate)
        if not cid or cid not in domain_dict:
            continue

        s_score = calculate_skill_score(candidate, jd_data, skill_matcher)
        # Domain relevance = content fit (precomputed) x role fit.
        profile = candidate.get("profile", {})
        titles = [profile.get("current_title", "")] + [
            r.get("title", "") for r in candidate.get("career_history", [])
        ]
        d_score = domain_dict[cid] * role_scorer.factor(titles)
        c_score = calculate_career_score(candidate, jd_data, role_rel_by_cid.get(cid))
        b_score = calculate_behavior_score(candidate)
        t_score, is_honeypot = calculate_trust(candidate)

        c_feat = CandidateScore(
            candidate_id=cid,
            skill_score=s_score,
            domain_relevance_score=d_score,
            behavior_score=b_score,
            career_score=c_score,
            trust_score=t_score,
            is_honeypot=is_honeypot
        )
        features_list.append(c_feat.to_dict())

    if not features_list:
        print("CRITICAL ERROR: No candidates were parsed.")
        sys.exit(1)

    # 5. Ranking
    df = pd.DataFrame(features_list)
    top_100 = rank_candidates(df, jd_data.get("weights", {}), top_n=100)

    # Pass 2: re-stream and keep full records for only the Top 100.
    top_ids = set(top_100["candidate_id"])
    raw_cache = {}
    for candidate in stream_candidates(candidates_path):
        cid = get_candidate_id(candidate)
        if cid in top_ids:
            raw_cache[cid] = candidate
            if len(raw_cache) == len(top_ids):
                break

    reasonings = []
    for _, row in top_100.iterrows():
        raw_cand = raw_cache.get(row["candidate_id"], {})
        reasonings.append(generate_reasoning(raw_cand, row["rank"], row["final_score"], jd_data))

    top_100["reasoning"] = reasonings

    generate_submission(top_100, args.out)

    elapsed = time.time() - start_time
    print(f"SUCCESS: Ranked {len(df)} candidates in {elapsed:.2f} seconds.")
    print(f"Submission saved to {args.out}")

if __name__ == "__main__":
    main()
