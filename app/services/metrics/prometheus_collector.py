from typing import Dict, Any
from app.core.interfaces import BaseMetricsCollector

class PrometheusMetricsCollector(BaseMetricsCollector):
    """
    Concrete implementation of BaseMetricsCollector.
    SRP: In-memory counter aggregation and Prometheus exposition generation.
    """

    def __init__(self):
        self.stats = {
            "total_queries": 0,
            "naive_hits": 0,
            "naive_false_hits": 0,
            "vericache_hits": 0,
            "vericache_rejected_candidates": 0,
            "vericache_misses": 0,
            "invalidations_count": 0,
            "total_ann_latency_ms": 0.0,
            "total_verifier_latency_ms": 0.0,
        }

    def record_query(self, is_naive_hit: bool, is_naive_false_hit: bool,
                     is_veri_hit: bool, is_veri_prevented: bool,
                     ann_latency_ms: float, verifier_latency_ms: float) -> None:
        self.stats["total_queries"] += 1
        self.stats["total_ann_latency_ms"] += ann_latency_ms
        self.stats["total_verifier_latency_ms"] += verifier_latency_ms

        if is_naive_hit:
            self.stats["naive_hits"] += 1
        if is_naive_false_hit:
            self.stats["naive_false_hits"] += 1

        if is_veri_hit:
            self.stats["vericache_hits"] += 1
        elif is_veri_prevented:
            self.stats["vericache_rejected_candidates"] += 1
            self.stats["vericache_misses"] += 1
        else:
            self.stats["vericache_misses"] += 1

    def record_invalidation(self, count: int) -> None:
        self.stats["invalidations_count"] += count

    def get_stats_summary(self, qdrant_count: int) -> Dict[str, Any]:
        total = max(1, self.stats["total_queries"])
        naive_hits = self.stats["naive_hits"]
        naive_false = self.stats["naive_false_hits"]
        veri_hits = self.stats["vericache_hits"]

        naive_precision = round(((naive_hits - naive_false) / naive_hits) * 100, 2) if naive_hits > 0 else 100.0
        veri_precision = round((veri_hits / (veri_hits + 0.0001)) * 100, 2) if veri_hits > 0 else 100.0
        naive_fpr = round((naive_false / naive_hits) * 100, 2) if naive_hits > 0 else 0.0

        avg_ann = round(self.stats["total_ann_latency_ms"] / total, 2)
        avg_ver = round(self.stats["total_verifier_latency_ms"] / total, 2)

        return {
            "stats": self.stats,
            "derived": {
                "naive_precision_pct": naive_precision,
                "vericache_precision_pct": veri_precision,
                "naive_fpr_pct": naive_fpr,
                "vericache_fpr_pct": 0.0,
                "avg_ann_latency_ms": avg_ann,
                "avg_verifier_latency_ms": avg_ver,
                "total_valid_entries": qdrant_count,
            }
        }

    def export_prometheus(self, qdrant_count: int) -> str:
        total = max(1, self.stats["total_queries"])
        naive_hits = self.stats["naive_hits"]
        naive_false = self.stats["naive_false_hits"]
        veri_hits = self.stats["vericache_hits"]
        rejected = self.stats["vericache_rejected_candidates"]

        naive_precision = ((naive_hits - naive_false) / naive_hits) if naive_hits > 0 else 1.0
        veri_precision = (veri_hits / (veri_hits + 0.0001)) if veri_hits > 0 else 1.0

        avg_ann = (self.stats["total_ann_latency_ms"] / total) if total > 0 else 0.0
        avg_ver = (self.stats["total_verifier_latency_ms"] / total) if total > 0 else 0.0

        lines = [
            "# HELP vericache_queries_total Total inbound semantic cache queries received",
            "# TYPE vericache_queries_total counter",
            f"vericache_queries_total {self.stats['total_queries']}",
            "",
            "# HELP vericache_naive_hits_total Queries that naive cosine similarity would claim as a hit",
            "# TYPE vericache_naive_hits_total counter",
            f"vericache_naive_hits_total {naive_hits}",
            "",
            "# HELP vericache_naive_false_hits_total Wrong/poisoned answers that naive cosine returned to users",
            "# TYPE vericache_naive_false_hits_total counter",
            f"vericache_naive_false_hits_total {naive_false}",
            "",
            "# HELP vericache_verified_hits_total Safe, cross-encoder validated cache hits served to users",
            "# TYPE vericache_verified_hits_total counter",
            f"vericache_verified_hits_total {veri_hits}",
            "",
            "# HELP vericache_false_hits_prevented_total False hits caught and blocked by the verifier",
            "# TYPE vericache_false_hits_prevented_total counter",
            f"vericache_false_hits_prevented_total {rejected}",
            "",
            "# HELP vericache_precision_ratio Current cache hit precision (True Hits / Total Claimed Hits)",
            "# TYPE vericache_precision_ratio gauge",
            f'vericache_precision_ratio{{engine="naive_cosine"}} {naive_precision:.4f}',
            f'vericache_precision_ratio{{engine="vericache"}} {veri_precision:.4f}',
            "",
            "# HELP vericache_latency_ms Average evaluation latency in milliseconds",
            "# TYPE vericache_latency_ms gauge",
            f'vericache_latency_ms{{stage="ann_recall"}} {avg_ann:.2f}',
            f'vericache_latency_ms{{stage="cross_encoder"}} {avg_ver:.2f}',
            "",
            "# HELP vericache_chunk_invalidations_total Total cache entries purged due to knowledge mutations",
            "# TYPE vericache_chunk_invalidations_total counter",
            f"vericache_chunk_invalidations_total {self.stats['invalidations_count']}",
            "",
            "# HELP vericache_qdrant_indexed_vectors Total vectors indexed in Qdrant HNSW collection",
            "# TYPE vericache_qdrant_indexed_vectors gauge",
            f"vericache_qdrant_indexed_vectors {qdrant_count}",
        ]
        return "\n".join(lines) + "\n"
