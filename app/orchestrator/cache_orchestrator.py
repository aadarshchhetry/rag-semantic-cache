import time
import uuid
from typing import Dict, Any, List, Optional

from app.core.interfaces import (
    BaseEmbedder,
    BaseVectorStore,
    BaseDocumentRetriever,
    BaseLLMSynthesizer,
    BaseVerifier,
    BaseMetricsCollector,
    CacheEvaluationResult
)

class CacheOrchestrator:
    """
    Coordinates the semantic caching lifecycle following Dependency Inversion (DIP).
    SRP: Pipeline flow orchestration only (Embed -> ANN Search -> Verify -> Real Retrieval -> LLM Synthesizer -> Telemetry).
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        store: BaseVectorStore,
        retriever: BaseDocumentRetriever,
        synthesizer: BaseLLMSynthesizer,
        verifier: BaseVerifier,
        metrics: BaseMetricsCollector,
        naive_cosine_threshold: float = 0.80
    ):
        self._embedder = embedder
        self._store = store
        self._retriever = retriever
        self._synthesizer = synthesizer
        self._verifier = verifier
        self._metrics = metrics
        self._naive_cosine_threshold = naive_cosine_threshold

    def evaluate_query(self, query: str, save_miss_to_cache: bool = False) -> CacheEvaluationResult:
        start_total = time.time()

        # Step 1: Embed Query locally using ONNX FastEmbed
        start_ann = time.time()
        query_vector = self._embedder.embed(query)

        # Step 2: ANN Vector Search against Semantic Cache
        candidate = self._store.search_nearest(query_vector, limit=1)
        ann_latency_ms = round((time.time() - start_ann) * 1000, 2)

        # Step 3: Evaluate Naive Cosine Rule
        cosine_sim = candidate.cosine_similarity if candidate else 0.0
        naive_hit = bool(candidate and cosine_sim >= self._naive_cosine_threshold)

        # Step 4: Evaluate VeriCache Cross-Encoder Verification Rule
        vericache_hit = False
        is_false_hit = False
        verifier_score = 0.0
        verifier_logit = 0.0
        verifier_latency_ms = 0.0
        verifier_reason = f"No candidate exceeded recall threshold ({self._naive_cosine_threshold})"

        if naive_hit and candidate:
            ver_res = self._verifier.verify(query, candidate.query)
            verifier_score = ver_res.score
            verifier_logit = ver_res.raw_logit
            verifier_latency_ms = ver_res.latency_ms
            verifier_reason = ver_res.reason

            if ver_res.is_verified:
                vericache_hit = True
            else:
                is_false_hit = True

        # Step 5: Answer Resolution & Real Document Retrieval on Miss / Intercept
        grounded_response = None
        source_chunks: List[str] = []
        retrieved_chunk_content: Optional[str] = None

        if vericache_hit and candidate:
            # Cache Hit: 0 Document Retrieval, 0 LLM Cost
            grounded_response = candidate.response
            source_chunks = candidate.chunk_ids
        else:
            # Cache Miss or False-Hit Blocked:
            # 1. REAL Document Retrieval from local Qdrant collection ($0 cost)
            chunks = self._retriever.retrieve_chunks(query_vector, top_k=1)
            
            # 2. LLM Synthesis grounded in retrieved document chunks (mocked to save API costs)
            grounded_response = self._synthesizer.synthesize_answer(query, chunks)
            
            if chunks:
                source_chunks = [c.chunk_id for c in chunks]
                retrieved_chunk_content = chunks[0].content
            else:
                source_chunks = ["doc_general_kb#chunk_01"]

            if save_miss_to_cache:
                new_id = str(uuid.uuid4())
                self._store.upsert_point(
                    point_id=new_id,
                    vector=query_vector,
                    payload={
                        "query": query,
                        "response": grounded_response,
                        "chunk_ids": source_chunks,
                        "created_at": time.time(),
                        "is_seed": False
                    }
                )

        total_latency_ms = round((time.time() - start_total) * 1000, 2)

        # Step 6: Record Operational Telemetry
        self._metrics.record_query(
            is_naive_hit=naive_hit,
            is_naive_false_hit=is_false_hit,
            is_veri_hit=vericache_hit,
            is_veri_prevented=is_false_hit,
            ann_latency_ms=ann_latency_ms,
            verifier_latency_ms=verifier_latency_ms
        )

        # Explanations
        matched_query = candidate.query if candidate else None
        if naive_hit and is_false_hit:
            naive_explanation = f"⚠️ Naive Cosine ({cosine_sim:.3f} >= {self._naive_cosine_threshold}) claimed a hit against '{matched_query}' and served its cached answer. BUT this is an incorrect answer for '{query}'!"
        elif naive_hit:
            naive_explanation = "Returned matching cached response."
        else:
            naive_explanation = f"Cache miss (Cosine < {self._naive_cosine_threshold})"

        if is_false_hit:
            veri_explanation = f"🛡️ Cross-Encoder detected that '{query}' does NOT ask the same thing as '{matched_query}'. Blocked false hit in {verifier_latency_ms}ms, performed REAL document retrieval, and returned accurate policy!"
        elif vericache_hit:
            veri_explanation = "Verified cache hit (Score >= 2.0). Zero LLM API cost incurred."
        else:
            veri_explanation = "Clean cache miss; retrieved real document chunk and synthesized answer safely."

        return CacheEvaluationResult(
            query=query,
            total_latency_ms=total_latency_ms,
            candidate_found=bool(candidate),
            matched_cached_query=matched_query,
            cosine_similarity=round(cosine_sim, 4),
            ann_latency_ms=ann_latency_ms,
            verifier_score=verifier_score,
            verifier_logit=verifier_logit,
            verifier_latency_ms=verifier_latency_ms,
            verifier_reason=verifier_reason,
            naive_hit=naive_hit,
            naive_response=candidate.response if naive_hit else None,
            naive_is_false_hit=is_false_hit,
            naive_explanation=naive_explanation,
            vericache_hit=vericache_hit,
            vericache_response=grounded_response,
            vericache_prevented_false_hit=is_false_hit,
            vericache_explanation=veri_explanation,
            source_chunks=source_chunks,
            retrieved_chunk_content=retrieved_chunk_content
        )

    def invalidate_chunk(self, chunk_id: str) -> Dict[str, Any]:
        start = time.time()
        purged = self._store.delete_by_chunk_id(chunk_id)
        self._metrics.record_invalidation(purged)
        latency_ms = round((time.time() - start) * 1000, 2)

        return {
            "chunk_id": chunk_id,
            "purged_count": purged,
            "latency_ms": latency_ms,
            "remaining_valid_entries": self._store.count()
        }

    def reset_seeds(self, seeds: List[Dict[str, Any]]) -> None:
        self._store.reset()
        for s in seeds:
            vec = self._embedder.embed(s["query"])
            self._store.upsert_point(
                point_id=s["id"],
                vector=vec,
                payload={
                    "query": s["query"],
                    "response": s["response"],
                    "chunk_ids": s["chunk_ids"],
                    "created_at": time.time(),
                    "is_seed": True
                }
            )

    def get_stats(self) -> Dict[str, Any]:
        return self._metrics.get_stats_summary(self._store.count())

    def export_metrics(self) -> str:
        return self._metrics.export_prometheus(self._store.count())
