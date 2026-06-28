"""
semantic_relevance.py
Computes the Domain Relevance Score as dense embedding similarity. Two embedding
signals combine multiplicatively:

  1. CONTENT fit  -- how close the candidate's whole profile text *means* to the JD
     (calculate_domain_relevance). A candidate who built a recommendation system at a
     product company scores high even if they never wrote "RAG" or "Pinecone".

  2. ROLE fit     -- how close the candidate's actual job roles (current + past titles)
     embed to the target role (RoleFitScorer). This stops a fundamentally non-technical
     profile (e.g. an "HR Manager") from ranking on text alone just because the summary
     and skills are padded with AI keywords. It is purely semantic: cosine of title
     embeddings, never substring/keyword matching.
"""
import numpy as np

# A short natural-language description of the target role. Used only as an embedding
# anchor (cosine compared to candidate titles), never as a match string.
ROLE_DESCRIPTION = (
    "Senior AI Engineer working on embeddings, retrieval, ranking, search, "
    "and recommendation systems"
)
# Role-fit factor shaping (tuned against the pool's title distribution): titles embedding
# at/above HIGH are a clean role match (no discount); at/below LOW they are clearly off-role
# and capped to FLOOR; linear in between.
ROLE_FIT_LOW = 0.20
ROLE_FIT_HIGH = 0.35
ROLE_FIT_FLOOR = 0.30


class RoleFitScorer:
    """Semantic role-fit gate. Returns a multiplier in [FLOOR, 1.0] from how closely a
    candidate's job titles embed to the target role. Titles come from a tiny fixed
    vocabulary, so each is encoded at most once."""

    def __init__(self, model, role_description: str = ROLE_DESCRIPTION):
        self.model = model
        self._role = model.encode([role_description], normalize_embeddings=True)[0]
        self._cache = {}

    def _title_vec(self, title: str) -> np.ndarray:
        vec = self._cache.get(title)
        if vec is None:
            vec = self.model.encode([title], normalize_embeddings=True)[0]
            self._cache[title] = vec
        return vec

    def factor(self, titles) -> float:
        vecs = [self._title_vec(t) for t in titles if t]
        if not vecs:
            return ROLE_FIT_FLOOR
        centroid = np.mean(vecs, axis=0)
        norm = np.linalg.norm(centroid)
        if norm == 0:
            return ROLE_FIT_FLOOR
        fit = float(self._role @ (centroid / norm))
        return float(np.clip((fit - ROLE_FIT_LOW) / (ROLE_FIT_HIGH - ROLE_FIT_LOW),
                             ROLE_FIT_FLOOR, 1.0))


def calculate_domain_relevance(jd_embedding: np.ndarray,
                               candidate_embeddings: np.ndarray) -> np.ndarray:
    """
    Pure semantic relevance (0-100) = cosine similarity of candidate profile embedding
    to the JD embedding, min-max normalized across the pool.

    jd_embedding:          shape (1, 384), L2-normalized
    candidate_embeddings:  shape (N, 384), L2-normalized

    Returns:
        np.ndarray of shape (N,) with normalized relevance scores in [0, 100].
    """
    if len(candidate_embeddings) == 0:
        return np.array([])

    # Embeddings are L2-normalized, so the dot product is cosine similarity.
    dense_scores = np.dot(candidate_embeddings, jd_embedding.T).flatten()

    # Min-max normalize the cosine similarities to 0-100 so the configured
    # domain_weight means what it says when components are combined.
    dmin, dmax = dense_scores.min(), dense_scores.max()
    if dmax > dmin:
        return ((dense_scores - dmin) / (dmax - dmin)) * 100.0
    return np.zeros_like(dense_scores)
