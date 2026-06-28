"""Domain relevance = semantic content fit x semantic role fit (no keyword matching)."""
import numpy as np

# Embedding anchor for the target role (cosine-compared to candidate titles), not a match string.
ROLE_DESCRIPTION = (
    "Senior AI Engineer working on embeddings, retrieval, ranking, search, "
    "and recommendation systems"
)
# Role-fit shaping, tuned to the pool: >=HIGH is a clean match, <=LOW capped to FLOOR, linear between.
ROLE_FIT_LOW = 0.20
ROLE_FIT_HIGH = 0.35
ROLE_FIT_FLOOR = 0.30


class RoleFitScorer:
    """Role-fit gate: multiplier in [FLOOR, 1.0] from how closely titles embed to the target role."""

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
    """Cosine similarity of each profile embedding to the JD, min-max normalized to 0-100."""
    if len(candidate_embeddings) == 0:
        return np.array([])

    # L2-normalized embeddings, so dot product is cosine similarity.
    dense_scores = np.dot(candidate_embeddings, jd_embedding.T).flatten()

    dmin, dmax = dense_scores.min(), dense_scores.max()
    if dmax > dmin:
        return ((dense_scores - dmin) / (dmax - dmin)) * 100.0
    return np.zeros_like(dense_scores)
