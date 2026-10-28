from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class CandidateMatch:
    """Represents a candidate retrieved from the semantic query cache."""
    id: str
    query: str
    response: str
    cosine_similarity: float
    chunk_ids: List[str] = field(default_factory=list)

@dataclass
class RetrievedChunk:
    """Represents a REAL document chunk retrieved from the local knowledge base ($0 cost)."""
    chunk_id: str
    document_title: str
    content: str
    score: float

@dataclass
class VerificationResult:
    """Represents the semantic cross-encoder verification outcome."""
    is_verified: bool
    score: float
    raw_logit: float
    reason: str
    latency_ms: float

@dataclass
class CacheEvaluationResult:
    """Comprehensive result of evaluating a query through both caching paths."""
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
    naive_hit: bool
    naive_response: Optional[str]
    naive_is_false_hit: bool
    naive_explanation: str
    vericache_hit: bool
    vericache_response: str
    vericache_prevented_false_hit: bool
    vericache_explanation: str
    source_chunks: List[str]
    retrieved_chunk_content: Optional[str] = None

# -------------------------------------------------------------
# SOLID Interfaces (Abstractions)
# -------------------------------------------------------------

class BaseEmbedder(ABC):
    """SRP: Maps text to dense vector embeddings."""
    @property
    @abstractmethod
    def dimension(self) -> int:
        pass

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        pass


class BaseVectorStore(ABC):
    """SRP: Vector store for cached queries."""
    @abstractmethod
    def search_nearest(self, query_vector: List[float], limit: int = 1) -> Optional[CandidateMatch]:
        pass

    @abstractmethod
    def upsert_point(self, point_id: str, vector: List[float], payload: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def delete_by_chunk_id(self, chunk_id: str) -> int:
        pass

    @abstractmethod
    def count(self) -> int:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass


class BaseDocumentRetriever(ABC):
    """
    SRP: REAL Document Knowledge Retrieval ($0 cost).
    Searches your company's actual internal document chunks using local embeddings.
    """
    @abstractmethod
    def retrieve_chunks(self, query_vector: List[float], top_k: int = 1) -> List[RetrievedChunk]:
        pass

    @abstractmethod
    def get_all_chunks(self) -> List[RetrievedChunk]:
        pass


class BaseLLMSynthesizer(ABC):
    """
    SRP: Generates or formats the final answer based on real retrieved document chunks.
    (This is where the dummy / cheap model lives to avoid expensive OpenAI API fees).
    """
    @abstractmethod
    def synthesize_answer(self, query: str, chunks: List[RetrievedChunk]) -> str:
        pass


class BaseVerifier(ABC):
    """SRP: Verifies token-level semantic intent equivalence."""
    @abstractmethod
    def verify(self, query: str, candidate_query: str) -> VerificationResult:
        pass


class BaseMetricsCollector(ABC):
    """SRP: Collects operational telemetry and exports Prometheus metrics."""
    @abstractmethod
    def record_query(self, is_naive_hit: bool, is_naive_false_hit: bool,
                     is_veri_hit: bool, is_veri_prevented: bool,
                     ann_latency_ms: float, verifier_latency_ms: float) -> None:
        pass

    @abstractmethod
    def record_invalidation(self, count: int) -> None:
        pass

    @abstractmethod
    def get_stats_summary(self, qdrant_count: int) -> Dict[str, Any]:
        pass

    @abstractmethod
    def export_prometheus(self, qdrant_count: int) -> str:
        pass
