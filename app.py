"""
app.py
Streamlit sandbox demo for the AI Recruiter ranking system.
Accepts a small candidate sample (≤100 candidates) and runs the full ranking pipeline.
Designed for HuggingFace Spaces / Streamlit Cloud deployment.
"""

import streamlit as st
import json
import io
import time
import tempfile
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

from src.loader import stream_candidates
from src.schema_analyzer import get_candidate_id, safe_str, get_profile, get_skills, get_career_history
from src.jd_parser import parse_jd
from src.skill_score import calculate_skill_score
from src.career_score import calculate_career_score
from src.behavior_score import calculate_behavior_score
from src.trust_score import calculate_trust
from src.jd_fit_penalty import calculate_jd_fit_penalty
from src.hybrid_ranker import calculate_hybrid_relevance
from src.ranker import rank_candidates
from src.reasoning import generate_reasoning
from src.features import CandidateScore


def concat_candidate_text(candidate: dict) -> str:
    """Concatenates key textual fields for dense/sparse retrieval."""
    profile = get_profile(candidate)
    skills_list = get_skills(candidate)
    career = get_career_history(candidate)
    
    parts = []
    current_title = safe_str(profile.get("current_title", ""))
    parts.extend([current_title] * 3)
    parts.append(safe_str(profile.get("headline", "")))
    parts.append(safe_str(profile.get("summary", "")))
    
    for s in skills_list:
        parts.append(safe_str(s.get("name", "")))
        
    for c in career:
        parts.append(safe_str(c.get("title", "")))
        parts.append(safe_str(c.get("description", "")))
        
    return " ".join([p for p in parts if p]).lower()


def run_ranking(candidates: list, jd_text: str, jd_data: dict) -> pd.DataFrame:
    """Runs the full ranking pipeline on a small candidate set."""
    
    # Build indexes on-the-fly for small sample
    candidate_ids = []
    texts = []
    for c in candidates:
        cid = get_candidate_id(c)
        if cid:
            candidate_ids.append(cid)
            texts.append(concat_candidate_text(c))
    
    if not texts:
        return pd.DataFrame()
    
    # Sparse Index
    vectorizer = TfidfVectorizer(stop_words='english', max_features=50000, sublinear_tf=True)
    sparse_matrix = vectorizer.fit_transform(texts)
    
    # Dense Embeddings
    model = SentenceTransformer("./models/all-MiniLM-L6-v2")
    embeddings = model.encode(texts, batch_size=64, normalize_embeddings=True)
    
    # JD vectors
    jd_bm25_vector = vectorizer.transform([jd_text.lower()])
    jd_embedding = model.encode([jd_text.lower()], normalize_embeddings=True)
    
    # Hybrid scores
    hybrid_scores = calculate_hybrid_relevance(
        jd_embedding=jd_embedding,
        jd_bm25_vector=jd_bm25_vector,
        candidate_embeddings=embeddings,
        candidate_bm25_matrix=sparse_matrix
    )
    
    hybrid_dict = {cid: score for cid, score in zip(candidate_ids, hybrid_scores)}
    
    # Score each candidate
    taxonomy_cache = {}
    features_list = []
    raw_cache = {}
    penalty_reasons_cache = {}
    trust_concerns_cache = {}
    
    for candidate in candidates:
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
        return pd.DataFrame()
    
    df = pd.DataFrame(features_list)
    top_n = min(100, len(df))
    top_results = rank_candidates(df, jd_data.get("weights", {}), top_n=top_n)
    
    # Generate reasoning
    reasonings = []
    for _, row in top_results.iterrows():
        cid = row["candidate_id"]
        rank_val = row["rank"]
        score_val = row["final_score"]
        raw_cand = raw_cache.get(cid, {})
        p_reasons = penalty_reasons_cache.get(cid, [])
        t_concerns = trust_concerns_cache.get(cid, [])
        reasoning = generate_reasoning(raw_cand, rank_val, score_val, jd_data, p_reasons, t_concerns)
        reasonings.append(reasoning)
    
    top_results["reasoning"] = reasonings
    
    # Format output
    output = top_results[["candidate_id", "rank", "final_score", "reasoning"]].copy()
    output = output.rename(columns={"final_score": "score"})
    output["score"] = output["score"].apply(lambda x: round(float(x), 5))
    
    return output


def main():
    st.set_page_config(page_title="AI Recruiter — Candidate Ranking", page_icon="🎯", layout="wide")
    
    st.title("🎯 AI Recruiter — Intelligent Candidate Ranking")
    st.markdown("""
    Upload a small candidate sample (≤100 candidates in JSON/JSONL format) and a job description 
    to run the full hybrid semantic ranking pipeline.
    
    > **Note:** This sandbox builds indexes on-the-fly for small samples (≤100 candidates). 
    > The full 100K ranking uses precomputed dense/sparse artifacts for sub-30-second execution.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📄 Job Description")
        jd_default = ""
        if os.path.exists("job_description.txt"):
            with open("job_description.txt", "r") as f:
                jd_default = f.read()
        jd_text = st.text_area("Paste or edit the Job Description", value=jd_default, height=300)
    
    with col2:
        st.subheader("👥 Candidate Data")
        uploaded_file = st.file_uploader(
            "Upload candidates (JSON array or JSONL, ≤100 candidates)", 
            type=["json", "jsonl"]
        )
        
        # Also allow using sample if available
        use_sample = False
        if os.path.exists("sample_candidates.json"):
            use_sample = st.checkbox("Use bundled sample_candidates.json instead")
    
    if st.button("🚀 Run Ranking", type="primary"):
        if not jd_text.strip():
            st.error("Please provide a job description.")
            return
        
        # Load candidates
        candidates = []
        
        if use_sample and os.path.exists("sample_candidates.json"):
            with open("sample_candidates.json", "r") as f:
                candidates = json.load(f)
        elif uploaded_file:
            content = uploaded_file.read().decode("utf-8")
            # Try JSON array first
            try:
                candidates = json.loads(content)
                if not isinstance(candidates, list):
                    candidates = [candidates]
            except json.JSONDecodeError:
                # Try JSONL
                candidates = []
                for line in content.strip().split("\n"):
                    line = line.strip()
                    if line:
                        try:
                            candidates.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        else:
            st.error("Please upload a candidate file or select the sample.")
            return
        
        if not candidates:
            st.error("No valid candidates found in the uploaded file.")
            return
        
        if len(candidates) > 100:
            st.warning(f"Trimming to first 100 candidates (uploaded {len(candidates)}).")
            candidates = candidates[:100]
        
        st.info(f"Loaded {len(candidates)} candidates. Running ranking pipeline...")
        
        # Write JD to temp file for parser
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp:
            tmp.write(jd_text)
            tmp_jd_path = tmp.name
        
        try:
            start = time.time()
            jd_data = parse_jd(tmp_jd_path)
            results = run_ranking(candidates, jd_text, jd_data)
            elapsed = time.time() - start
        finally:
            os.unlink(tmp_jd_path)
        
        if results.empty:
            st.error("No candidates could be ranked. Check your input data.")
            return
        
        st.success(f"✅ Ranked {len(results)} candidates in {elapsed:.2f} seconds.")
        
        # Display results
        st.subheader("📊 Ranked Candidates")
        st.dataframe(results, use_container_width=True, hide_index=True)
        
        # Download CSV
        csv_buffer = io.StringIO()
        results.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download Ranking CSV",
            data=csv_buffer.getvalue(),
            file_name="ranked_candidates.csv",
            mime="text/csv"
        )


if __name__ == "__main__":
    main()
