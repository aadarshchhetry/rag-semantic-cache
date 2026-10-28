from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, PlainTextResponse
from typing import Dict, Any

from app.api.schemas import QueryRequest, InvalidateRequest
from app.orchestrator.cache_orchestrator import CacheOrchestrator

router = APIRouter()

# Dependency provider function (injected at runtime)
def get_orchestrator(request: Request) -> CacheOrchestrator:
    return request.app.state.orchestrator

@router.get("/metrics", response_class=PlainTextResponse)
def get_metrics(orch: CacheOrchestrator = Depends(get_orchestrator)):
    return orch.export_metrics()

@router.get("/v1/cache/stats")
def get_stats(orch: CacheOrchestrator = Depends(get_orchestrator)):
    return orch.get_stats()

@router.post("/v1/cache/query")
def evaluate_query(req: QueryRequest, orch: CacheOrchestrator = Depends(get_orchestrator)):
    res = orch.evaluate_query(req.query, save_miss_to_cache=req.save_miss_to_cache)
    return {
        "query": res.query,
        "total_latency_ms": res.total_latency_ms,
        "candidate_found": res.candidate_found,
        "matched_cached_query": res.matched_cached_query,
        "cosine_similarity": res.cosine_similarity,
        "ann_latency_ms": res.ann_latency_ms,
        "verifier_score": res.verifier_score,
        "verifier_logit": res.verifier_logit,
        "verifier_latency_ms": res.verifier_latency_ms,
        "verifier_reason": res.verifier_reason,
        "naive_cache": {
            "hit": res.naive_hit,
            "returned_response": res.naive_response,
            "is_false_hit": res.naive_is_false_hit,
            "explanation": res.naive_explanation
        },
        "vericache": {
            "hit": res.vericache_hit,
            "returned_response": res.vericache_response,
            "prevented_false_hit": res.vericache_prevented_false_hit,
            "source_chunks": res.source_chunks,
            "retrieved_chunk_content": res.retrieved_chunk_content,
            "explanation": res.vericache_explanation
        }
    }

@router.post("/v1/cache/reset")
def reset_cache(request: Request, orch: CacheOrchestrator = Depends(get_orchestrator)):
    seeds = request.app.state.initial_seeds
    orch.reset_seeds(seeds)
    return {"status": "success", "message": f"Reset to {len(seeds)} baseline seed documents.", "stats": orch.get_stats()}

@router.post("/v1/cache/invalidate")
def invalidate_chunk(req: InvalidateRequest, orch: CacheOrchestrator = Depends(get_orchestrator)):
    return orch.invalidate_chunk(req.chunk_id)

@router.post("/v1/demo/benchmark")
def run_benchmark(orch: CacheOrchestrator = Depends(get_orchestrator)):
    test_suite = [
        "How do I cancel my order?",
        "Can I return shoes after 30 days?",
        "Why was my personal loan application approved?",
        "How do I cancel my subscription?",
        "Why was my personal loan rejected?",
        "What is the return window for a laptop?",
        "Why was my mortgage denied?",
        "Where can I find documentation on Kubernetes?",
    ]

    results = []
    for q in test_suite:
        res = orch.evaluate_query(q, save_miss_to_cache=False)
        results.append({
            "query": res.query,
            "matched_cached_query": res.matched_cached_query,
            "cosine_similarity": res.cosine_similarity,
            "naive_hit": res.naive_hit,
            "vericache_hit": res.vericache_hit,
            "prevented_false_hit": res.vericache_prevented_false_hit
        })

    return {
        "queries_run": len(test_suite),
        "results": results,
        "current_stats": orch.get_stats()
    }
