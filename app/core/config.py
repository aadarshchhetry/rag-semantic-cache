import os
from pydantic import BaseModel

class AppSettings(BaseModel):
    # Paths (Defaulting to D: drive for persistent models and disk space safety)
    cache_storage_dir: str = os.getenv("CACHE_STORAGE_DIR", "d:/aadarsh/AI/.fastembed_cache")
    qdrant_storage_dir: str = os.getenv("QDRANT_STORAGE_DIR", "d:/aadarsh/AI/CacheSementic/data/qdrant_db")

    # Thresholds
    naive_cosine_threshold: float = float(os.getenv("NAIVE_COSINE_THRESHOLD", "0.80"))
    verifier_logit_threshold: float = float(os.getenv("VERIFIER_LOGIT_THRESHOLD", "2.0"))

    # Models
    embedding_model_name: str = "BAAI/bge-small-en-v1.5"
    verifier_model_name: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    vector_dimension: int = 384

settings = AppSettings()
