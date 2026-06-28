"""Skill fit = verified depth x semantic relevance to JD domains (no keyword matching)."""

import numpy as np

from .schema_analyzer import get_skills, get_redrob_signals, safe_float, safe_int

# Embedding anchors per JD domain (cosine-compared to skill names), not match strings.
DOMAIN_PHRASES = {
    "VECTOR_DATABASES": "vector database and hybrid search infrastructure such as FAISS, Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch",
    "EMBEDDINGS": "embeddings-based retrieval using sentence transformers, BGE, E5, and embedding models",
    "RETRIEVAL": "information retrieval, semantic search, and retrieval-augmented generation (RAG)",
    "RANKING": "ranking, re-ranking, learning to rank, and recommendation systems",
    "EVALUATION": "ranking evaluation frameworks and metrics like NDCG, MRR, MAP, and A/B testing",
    "LLM_FINETUNING": "large language model fine-tuning with LoRA, QLoRA, and PEFT",
    "DATA_ENGINEERING": "data engineering, data pipelines, SQL, and big data processing",
    "SOFTWARE_ENGINEERING": "software engineering and strong Python programming",
    "ML_INFRASTRUCTURE": "machine learning infrastructure, MLOps, and model deployment at scale",
}
NEGATIVE_PHRASES = {
    "COMPUTER_VISION": "computer vision, image classification, object detection, and image processing",
    "SPEECH": "speech recognition, text-to-speech, and audio processing",
    "ROBOTICS": "robotics, SLAM, motion planning, and autonomous navigation",
}

# Required matches outweigh preferred.
REQUIRED_W = 2.5
PREFERRED_W = 1.0

# Cosine floor below which a skill is treated as unrelated; tuned to the skill vocabulary.
SIM_FLOOR = 0.30
# At/above this similarity to a rejected-domain anchor, a skill is clearly in that domain.
NEG_SIM = 0.45

# Untested skills are self-reportable, so they earn only small capped credit and only with endorsements.
UNASSESSED_MIN_ENDORSEMENTS = 10   # below this, an untested skill does not count
UNASSESSED_DEPTH_CAP = 0.25        # per-skill ceiling for untested skills
UNASSESSED_TOTAL_CAP = 2.0         # per-candidate ceiling on untested contribution


class SkillMatcher:
    """JD anchor embeddings plus a skill-name embedding cache (each name encoded once per run)."""

    def __init__(self, model, required_phrases, preferred_phrases, negative_phrases):
        self.model = model
        self._cache = {}
        self.req_mat = self._encode(required_phrases)
        self.pref_mat = self._encode(preferred_phrases)
        self.neg_mat = self._encode(negative_phrases)

    def _encode(self, phrases):
        if not phrases:
            return None
        return self.model.encode(phrases, normalize_embeddings=True)

    def embed(self, name: str) -> np.ndarray:
        vec = self._cache.get(name)
        if vec is None:
            vec = self.model.encode([name], normalize_embeddings=True)[0]
            self._cache[name] = vec
        return vec

    @staticmethod
    def _max_sim(mat, vec) -> float:
        if mat is None:
            return 0.0
        return float(np.max(mat @ vec))


def build_skill_matcher(model, jd_data: dict) -> SkillMatcher:
    """Build this JD's embedding anchors from its required/preferred/negative domains."""
    req = [DOMAIN_PHRASES[d] for d in jd_data.get("required_skills", []) if d in DOMAIN_PHRASES]
    pref = [DOMAIN_PHRASES[d] for d in jd_data.get("preferred_skills", []) if d in DOMAIN_PHRASES]
    neg = [NEGATIVE_PHRASES[d] for d in jd_data.get("negative_domains", []) if d in NEGATIVE_PHRASES]
    return SkillMatcher(model, req, pref, neg)


def _depth(assessment, duration_months: int, endorsements: int) -> tuple[float, bool]:
    """Structured proof of depth -> (depth, assessed); 0.0 when the skill is unverified."""
    dur_norm = min(duration_months / 48.0, 1.0)
    end_norm = min(endorsements / 40.0, 1.0)

    if assessment is not None:
        # A platform test result earns full credit; duration/endorsements only sweeten it.
        return 0.6 * (assessment / 100.0) + 0.25 * dur_norm + 0.15 * end_norm, True

    # Untested: only endorsements qualify, capped; duration alone never does.
    if endorsements < UNASSESSED_MIN_ENDORSEMENTS:
        return 0.0, False
    depth = min(0.15 * end_norm + 0.10 * dur_norm, UNASSESSED_DEPTH_CAP)
    return depth, False


def calculate_skill_score(candidate: dict, jd_data: dict, matcher: SkillMatcher) -> float:
    """Raw skill score: verified depth in skills that semantically match the JD."""
    if matcher is None or (matcher.req_mat is None and matcher.pref_mat is None):
        return 0.0

    signals = get_redrob_signals(candidate)
    assessments = signals.get("skill_assessment_scores", {}) or {}

    # Merge listed skills with assessed-but-unlisted ones (a test result counts even if unlisted).
    merged = {}
    for s in get_skills(candidate):
        name = s.get("name", "")
        if not name:
            continue
        merged[name] = {
            "assessment": safe_float(assessments[name]) if name in assessments else None,
            "duration": safe_int(s.get("duration_months", 0)),
            "endorsements": safe_int(s.get("endorsements", 0)),
        }
    for name, val in assessments.items():
        if name not in merged:
            merged[name] = {"assessment": safe_float(val), "duration": 0, "endorsements": 0}

    score = 0.0
    untested_score = 0.0
    best_req_sim = 0.0
    neg_load = 0.0

    for name, info in merged.items():
        depth, assessed = _depth(info["assessment"], info["duration"], info["endorsements"])
        if depth <= 0.0:
            continue  # unverified earns nothing

        vec = matcher.embed(name)
        sim_req = matcher._max_sim(matcher.req_mat, vec)
        sim_pref = matcher._max_sim(matcher.pref_mat, vec)
        sim_neg = matcher._max_sim(matcher.neg_mat, vec)

        rel_req = max(0.0, sim_req - SIM_FLOOR)
        rel_pref = max(0.0, sim_pref - SIM_FLOOR)
        contribution = depth * (REQUIRED_W * rel_req + PREFERRED_W * rel_pref)
        if assessed:
            score += contribution
        else:
            untested_score += contribution

        best_req_sim = max(best_req_sim, sim_req)
        # Depth that sits in a rejected domain and isn't JD-relevant adds to negative load.
        if sim_neg >= NEG_SIM and sim_neg > sim_req:
            neg_load += depth * (sim_neg - SIM_FLOOR)

    score += min(untested_score, UNASSESSED_TOTAL_CAP)

    # Negative-domain penalty: only candidates whose verified strength is in a rejected
    # domain with no genuine JD relevance.
    if neg_load > 0.0 and best_req_sim < (SIM_FLOOR + 0.10):
        score -= min(neg_load * REQUIRED_W, 12.0)

    return score
