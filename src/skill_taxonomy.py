"""
skill_taxonomy.py
Defines the centralized taxonomy of skills, replacing exact keyword matching.
"""

# TAXONOMY groups exact variants and aliases under a common domain key.
TAXONOMY = {
    "VECTOR_DATABASES": [
        # Named systems
        "faiss", "qdrant", "milvus", "pinecone", "weaviate", "elasticsearch",
        "opensearch", "chromadb", "pgvector", "annoy", "nmslib", "vespa",
        "marqo", "zilliz", "lance", "typesense", "redis vector", "vald",
        # Generic concepts — catches Tier-5 candidates who describe the work
        "vector database", "vector databases", "vector store", "vector stores",
        "vector search", "vector index", "vector indexing",
        "approximate nearest neighbor", "ann index", "knn search",
        "nearest neighbor search", "dense index", "inverted file index",
        "hnsw", "ivf", "product quantization",
    ],
    "EMBEDDINGS": [
        # Named models / libraries
        "bge", "e5", "sentence transformers", "sentence-transformers", "sbert",
        "sentence-bert", "openai embeddings", "cohere embeddings", "text-embedding",
        "text-embedding-ada", "instructor", "mpnet", "word2vec", "glove",
        "fasttext", "clip", "align",
        # Generic concepts
        "embedding models", "embeddings", "embedding model", "dense vectors",
        "dense embeddings", "text embeddings", "neural embeddings",
        "semantic embeddings", "representation learning", "contrastive learning",
        "bi-encoder", "biencoder", "dual encoder", "feature extraction",
        "sentence embeddings", "document embeddings", "passage embeddings",
        "embedding space", "latent space", "vector representations",
        "similarity search", "semantic similarity",
    ],
    "RETRIEVAL": [
        # Techniques and systems
        "bm25", "tf-idf", "tfidf", "lucene", "solr",
        "semantic search", "hybrid search", "dense retrieval", "sparse retrieval",
        "rag", "retrieval augmented generation", "retrieval-augmented generation",
        "information retrieval", "document retrieval", "passage retrieval",
        "candidate retrieval", "candidate search", "two-stage retrieval",
        "multi-stage retrieval", "dense passage retrieval", "dpr",
        "colbert", "splade", "query expansion", "query rewriting",
        # Generic descriptions — catches "built a search system" profiles
        "search engine", "search system", "search infrastructure",
        "search platform", "search pipeline", "search backend",
        "retrieval system", "full text search", "keyword search",
        "open domain qa", "knowledge retrieval", "neural search",
        "search ranking", "search relevance", "search quality",
    ],
    "RANKING": [
        # Algorithms
        "learning to rank", "ltr", "lambdamart", "lambdarank", "ranknet",
        "listnet", "listwise", "pairwise ranking", "pointwise ranking",
        "xgboost", "lightgbm", "catboost", "gbdt",
        "cross-encoder", "re-ranking", "reranker", "neural reranking",
        "two-tower model", "collaborative filtering", "matrix factorization",
        # Generic descriptions
        "ranking", "ranking system", "ranking model", "ranking algorithm",
        "candidate ranking", "candidate scoring", "scoring model",
        "relevance ranking", "relevance model", "relevance scoring",
        "recommendation systems", "recommendation engine", "recommender system",
        "recsys", "content-based filtering", "personalization",
        "search ranking", "feed ranking", "ad ranking", "job ranking",
        "match scoring", "matching model",
    ],
    "EVALUATION": [
        # Metrics
        "ndcg", "map", "mrr", "precision at k", "recall at k",
        "mean average precision", "mean reciprocal rank",
        "normalized discounted cumulative gain",
        "hit rate", "coverage", "diversity metrics",
        # Testing and frameworks
        "a/b testing", "ab testing", "online evaluation", "offline evaluation",
        "interleaving", "counterfactual evaluation", "causal inference",
        "position bias", "click model", "relevance judgment",
        "human evaluation", "evaluation framework", "evaluation frameworks",
        "evaluation metrics", "evaluation pipeline", "evaluation infrastructure",
        "benchmark", "benchmarking", "offline benchmark",
        "online metrics", "click-through rate", "ctr", "dwell time",
        "experiment framework", "experimentation platform",
        "metrics", "evaluation", "recall evaluation", "precision evaluation",
    ],
    "LLM_FINETUNING": [
        # PEFT methods
        "lora", "qlora", "peft", "prefix tuning", "prompt tuning",
        "adapter", "adapters", "ia3",
        # Training paradigms
        "fine-tuning", "llm fine-tuning", "instruction tuning", "sft",
        "supervised fine-tuning", "rlhf", "reinforcement learning from human feedback",
        "dpo", "direct preference optimization", "ppo",
        "full fine-tuning", "continued pretraining",
        # Infra
        "deepspeed", "fsdp", "megatron", "parameter efficient",
        # Models (signal that person works with LLMs)
        "llm", "large language model", "large language models",
        "gpt", "llama", "mistral", "gemma", "falcon", "phi",
        "foundation model", "foundation models", "transformer fine-tuning",
        "language model training", "model training",
    ],
    "DATA_ENGINEERING": [
        # Databases and warehouses
        "sql", "postgresql", "mysql", "redshift", "snowflake", "bigquery",
        "clickhouse", "druid", "presto", "trino", "hive", "databricks",
        # Processing frameworks
        "spark", "pyspark", "apache spark", "flink", "apache flink",
        "kafka", "apache kafka", "airflow", "apache airflow",
        "hadoop", "mapreduce", "beam", "apache beam",
        # Tooling
        "dbt", "pandas", "numpy", "data pipelines", "etl", "elt",
        "data warehouse", "data lake", "data lakehouse",
        "feature store", "feature engineering", "feature pipeline",
        "data processing", "batch processing", "stream processing",
        "s3", "gcs", "azure blob", "object storage",
    ],
    "SOFTWARE_ENGINEERING": [
        # Languages
        "python", "java", "c++", "golang", "go", "rust", "scala",
        "kotlin", "typescript", "javascript",
        # Frameworks and APIs
        "fastapi", "flask", "django", "rest api", "grpc", "graphql",
        # Infrastructure
        "docker", "kubernetes", "k8s", "terraform", "helm",
        "aws", "gcp", "azure", "cloud",
        "microservices", "distributed systems", "system design",
        "ci/cd", "github actions", "jenkins", "unit testing", "pytest",
        # General
        "software engineering", "backend", "backend engineering",
        "production systems", "production code", "api development",
        "code quality", "clean code",
    ],
    "ML_INFRASTRUCTURE": [
        # Serving and deployment
        "model serving", "model deployment", "model inference",
        "torchserve", "tf serving", "triton", "seldon", "bentoml",
        "ray serve", "kserve", "inference server",
        # Optimization
        "tensorrt", "onnx", "quantization", "model quantization",
        "model compression", "distillation", "knowledge distillation",
        "inference optimization", "latency optimization",
        # MLOps platforms
        "mlops", "kubeflow", "sagemaker", "vertex ai",
        "mlflow", "weights and biases", "wandb", "dvc",
        "feast", "tecton", "feature store",
        "model registry", "model monitoring", "data drift",
        # Deployment patterns
        "shadow deployment", "canary deployment", "blue-green deployment",
        "production ml", "ml pipeline", "training pipeline",
        "ray", "celery",
    ],
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
