from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class QueryRequest(BaseModel):
    query: str
    save_miss_to_cache: Optional[bool] = False

class InvalidateRequest(BaseModel):
    chunk_id: str

class CacheQueryResponse(BaseModel):
    query: str
    total_latency_ms: float
    candidate_found: bool
    matched_cached_query: Optional[str]
    cosine_similarity: float
    ann_latency_ms: float
    verifier_score: float
    verifier_logit: float
    verifier_latency_ms: float
    verifier_reason: str
    naive_cache: Dict[str, Any]
    vericache: Dict[str, Any]
