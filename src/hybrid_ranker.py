"""
hybrid_ranker.py
Computes the Hybrid Semantic Relevance Score using precomputed artifacts.
Bypasses slow string matching using highly optimized NumPy and SciPy vector math.
"""
import numpy as np

def calculate_hybrid_relevance(jd_embedding: np.ndarray, jd_bm25_vector, 
                               candidate_embeddings: np.ndarray, candidate_bm25_matrix) -> np.ndarray:
    """
    Computes a hybrid relevance score (0-100) combining dense semantic similarity and sparse exact matching.
    
    jd_embedding: shape (1, 384)
    candidate_embeddings: shape (N, 384)
    jd_bm25_vector: sparse matrix (1, V)
    candidate_bm25_matrix: sparse matrix (N, V)
    
    Returns:
        np.ndarray of shape (N,) containing normalized hybrid scores.
    """
    if len(candidate_embeddings) == 0:
        return np.array([])
        
    # 1. Dense Semantic Similarity (Cosine Similarity)
    # Since embeddings are L2 normalized, dot product == cosine similarity
    # Resulting shape: (N,)
    dense_scores = np.dot(candidate_embeddings, jd_embedding.T).flatten()
    
    # 2. Sparse BM25 Similarity (Dot Product)
    # Resulting shape: (N,)
    sparse_scores = candidate_bm25_matrix.dot(jd_bm25_vector.T).toarray().flatten()
    
    # 3. Normalize Dense Scores to 0-100
    dense_min, dense_max = dense_scores.min(), dense_scores.max()
    if dense_max > dense_min:
        norm_dense = ((dense_scores - dense_min) / (dense_max - dense_min)) * 100.0
    else:
        norm_dense = np.zeros_like(dense_scores)
        
    # 4. Normalize Sparse Scores to 0-100
    sparse_min, sparse_max = sparse_scores.min(), sparse_scores.max()
    if sparse_max > sparse_min:
        norm_sparse = ((sparse_scores - sparse_min) / (sparse_max - sparse_min)) * 100.0
    else:
        norm_sparse = np.zeros_like(sparse_scores)
        
    # 5. Blend into Hybrid Score
    # Give 70% weight to sparse to ensure exact technical keywords match, 
    # 30% to dense to capture latent semantic variants.
    hybrid_scores = (norm_sparse * 0.70) + (norm_dense * 0.30)
    
    return hybrid_scores
