"""
skill_taxonomy.py
Defines the centralized taxonomy of skills, replacing exact keyword matching.
"""

# TAXONOMY groups exact variants and aliases under a common domain key.
TAXONOMY = {
    "VECTOR_DATABASES": [
        "faiss", "qdrant", "milvus", "pinecone", "weaviate", "elasticsearch", 
        "opensearch", "vector database", "vector databases", "chromadb"
    ],
    "EMBEDDINGS": [
        "bge", "e5", "sentence transformers", "embedding models", "embeddings", 
        "word2vec", "glove", "openai embeddings", "cohere embeddings", "text-embedding"
    ],
    "RETRIEVAL": [
        "semantic search", "information retrieval", "rag", "candidate search", 
        "document retrieval", "bm25", "hybrid search", "retrieval augmented generation"
    ],
    "RANKING": [
        "learning to rank", "ranking", "recommendation systems", "candidate ranking", 
        "xgboost", "recsys", "re-ranking", "cross-encoder"
    ],
    "EVALUATION": [
        "ndcg", "map", "mrr", "offline evaluation", "a/b testing", "ab testing", 
        "evaluation frameworks", "evaluation", "metrics"
    ],
    "LLM_FINETUNING": [
        "lora", "qlora", "peft", "fine-tuning", "llm fine-tuning", "instruction tuning"
    ],
    "DATA_ENGINEERING": [
        "sql", "spark", "airflow", "data pipelines", "hadoop", "snowflake", "bigquery"
    ],
    "SOFTWARE_ENGINEERING": [
        "python", "java", "c++", "golang", "go", "software engineering", "backend"
    ],
    "ML_INFRASTRUCTURE": [
        "mlops", "kubeflow", "sagemaker", "model deployment", "tensorrt", "onnx", "triton", "production ml"
    ]
}

def get_canonical_skill(raw_skill: str) -> str:
    """
    Returns the canonical domain name for a raw skill.
    If no match is found, returns the raw_skill in lowercase.
    
    Uses partial matching (e.g. if 'qdrant' is within 'qdrant vector db').
    """
    if not raw_skill:
        return ""
        
    raw_skill_lower = raw_skill.lower().strip()
    
    # 1. Exact match pass
    for domain, aliases in TAXONOMY.items():
        if raw_skill_lower in aliases:
            return domain
            
    # 2. Substring match pass
    for domain, aliases in TAXONOMY.items():
        for alias in aliases:
            # Check if the alias is a distinct word in the skill
            # (To avoid matching "go" in "google")
            if alias in raw_skill_lower:
                # Add a quick heuristic to avoid over-matching very short aliases
                if len(alias) >= 3 or f" {alias} " in f" {raw_skill_lower} ":
                    return domain

    return raw_skill_lower

def build_taxonomy_lookup_cache(skill_names: list[str]) -> dict[str, str]:
    """
    For performance optimization. Given a list of unique raw skills, 
    returns a dictionary mapping raw_skill -> canonical_skill.
    This avoids re-evaluating the rules for every candidate.
    """
    return {skill: get_canonical_skill(skill) for skill in skill_names}
