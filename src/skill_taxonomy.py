"""Skill taxonomy: alias groups under canonical domain keys."""

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
    """Map a raw skill to its canonical domain, or the lowercased raw skill if no match."""
    if not raw_skill:
        return ""

    raw_skill_lower = raw_skill.lower().strip()

    for domain, aliases in TAXONOMY.items():
        if raw_skill_lower in aliases:
            return domain

    for domain, aliases in TAXONOMY.items():
        for alias in aliases:
            if alias in raw_skill_lower:
                # Guard short aliases against over-matching (e.g. "go" in "google")
                if len(alias) >= 3 or f" {alias} " in f" {raw_skill_lower} ":
                    return domain

    return raw_skill_lower
