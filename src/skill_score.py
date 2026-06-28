"""
skill_score.py
Scores skill fit WITHOUT keyword/taxonomy string matching.

Two ideas drive this:

  1. STRUCTURED proof of depth. A listed skill is worthless on its own ("the word
     appeared"). A skill only counts when the candidate has *verified* depth in it:
     a Redrob platform assessment score (an actual test result), real endorsements,
     and/or sustained duration of use. No proof -> the skill is ignored.

  2. SEMANTIC cross-reference. Whether a verified skill is what the JD wants is decided
     by embedding similarity between the skill's name and the JD's required/preferred
     skill concepts -- NOT by exact-string / alias matching. "Vector Search" and
     "Semantic Search" score high against the JD even though they are different strings.

Final raw score = sum over verified skills of  depth x semantic_relevance x role_weight,
with a semantic negative-domain penalty for candidates whose only verified strength sits
in a JD-rejected area (computer vision / speech / robotics).
"""

import numpy as np

from .schema_analyzer import get_skills, get_redrob_signals, safe_float, safe_int

# Natural-language descriptions of each JD skill domain. These are EMBEDDING ANCHORS,
# not match strings: candidate skill names are compared to them by cosine similarity,
# never by substring. The set of anchors actually used is chosen per-JD from the
# required/preferred/negative domains the JD parser detects.
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

# Role weighting: a required-skill match is worth more than a preferred-skill match.
REQUIRED_W = 2.5
PREFERRED_W = 1.0

# Cosine floor below which a skill name is treated as semantically unrelated to the JD
# (so generic terms contribute ~0). Tuned against the dataset's skill vocabulary.
SIM_FLOOR = 0.30
# A skill name this close to a rejected-domain anchor is clearly in that domain.
NEG_SIM = 0.45

# Verified-depth policy (assessment-anchored). An objective Redrob platform assessment
# is the only un-fakeable evidence of skill depth, so it alone earns full credit.
# Self-reported fields (duration, endorsements) are stuffable: a candidate can type
# "14 months" against any skill. So an UNASSESSED skill earns only small, capped credit,
# and ONLY when it carries real social proof (endorsements). Duration alone never
# qualifies a skill -- that is exactly the "the word appeared" loophole we are closing.
UNASSESSED_MIN_ENDORSEMENTS = 10   # below this, an untested skill does not count at all
UNASSESSED_DEPTH_CAP = 0.25        # per-skill ceiling for untested skills
UNASSESSED_TOTAL_CAP = 2.0         # per-candidate ceiling on total untested contribution


class SkillMatcher:
    """Holds the JD's anchor embeddings and a per-skill-name embedding cache so each of
    the dataset's ~130 unique skill names is encoded at most once per run."""

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
    """Builds the embedding anchors for THIS JD from its detected required/preferred/
    negative domains."""
    req = [DOMAIN_PHRASES[d] for d in jd_data.get("required_skills", []) if d in DOMAIN_PHRASES]
    pref = [DOMAIN_PHRASES[d] for d in jd_data.get("preferred_skills", []) if d in DOMAIN_PHRASES]
    neg = [NEGATIVE_PHRASES[d] for d in jd_data.get("negative_domains", []) if d in NEGATIVE_PHRASES]
    return SkillMatcher(model, req, pref, neg)


def _depth(assessment, duration_months: int, endorsements: int) -> tuple[float, bool]:
    """Structured proof of skill depth. Returns (depth, assessed).

    depth is 0.0 when the skill is unverified (just listed) so it contributes nothing.
    `assessed` is True when the depth is backed by an objective platform assessment, so
    the caller can cap untested credit separately.
    """
    dur_norm = min(duration_months / 48.0, 1.0)
    end_norm = min(endorsements / 40.0, 1.0)

    if assessment is not None:
        # An actual platform test result is the strongest, most objective evidence and
        # earns full credit; duration/endorsements only sweeten it.
        return 0.6 * (assessment / 100.0) + 0.25 * dur_norm + 0.15 * end_norm, True

    # No test result. Duration alone never qualifies (it is trivially self-reported).
    # Only real social proof (endorsements) lets an untested skill count, and then only
    # for a small, capped amount that duration can lightly amplify.
    if endorsements < UNASSESSED_MIN_ENDORSEMENTS:
        return 0.0, False
    depth = min(0.15 * end_norm + 0.10 * dur_norm, UNASSESSED_DEPTH_CAP)
    return depth, False


def calculate_skill_score(candidate: dict, jd_data: dict, matcher: SkillMatcher) -> float:
    """Raw skill score for the candidate. Higher = more verified depth in skills that
    semantically match what the JD asks for."""
    if matcher is None or (matcher.req_mat is None and matcher.pref_mat is None):
        return 0.0

    signals = get_redrob_signals(candidate)
    assessments = signals.get("skill_assessment_scores", {}) or {}

    # Merge listed skills and assessed-but-unlisted skills into one verified view,
    # keyed by skill name. The assessment dict itself is structured proof, so a skill
    # the candidate was tested on counts even if they forgot to list it.
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
            continue  # unverified -> the word appearing earns nothing

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
        # Verified depth that clearly sits in a rejected domain and is NOT also relevant
        # to the JD contributes to the negative load.
        if sim_neg >= NEG_SIM and sim_neg > sim_req:
            neg_load += depth * (sim_neg - SIM_FLOOR)

    # Untested skills can collectively add only a small, capped amount, so a profile
    # stuffed with many self-reported skills can never out-score genuine tested depth.
    score += min(untested_score, UNASSESSED_TOTAL_CAP)

    # Semantic negative-domain penalty: only bite candidates whose verified strength is
    # in computer-vision/speech/robotics AND who show no genuine retrieval/ranking
    # relevance. A real AI engineer who merely also knows CV has some skill clearing the
    # relevance bar, so they are never penalized.
    if neg_load > 0.0 and best_req_sim < (SIM_FLOOR + 0.10):
        score -= min(neg_load * REQUIRED_W, 12.0)

    return score
