import os
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from app.core.interfaces import BaseVectorStore, CandidateMatch

class QdrantVectorStore(BaseVectorStore):
    """
    Concrete implementation of BaseVectorStore using Qdrant Embedded HNSW.
    SRP: Vector indexing, spatial nearest-neighbor search, and metadata deletions.
    """

    COLLECTION_NAME = "vericache_entries"

    def __init__(
        self,
        storage_dir: Optional[str] = None,
        dimension: int = 384,
        client: Optional[QdrantClient] = None
    ):
        self._dim = dimension
        if client is not None:
            self._client = client
        elif storage_dir:
            os.makedirs(storage_dir, exist_ok=True)
            try:
                self._client = QdrantClient(path=storage_dir)
            except Exception as e:
                print(f"[QdrantVectorStore] Persistent store busy, falling back to memory: {e}")
                self._client = QdrantClient(":memory:")
        else:
            self._client = QdrantClient(":memory:")

        self._ensure_collection()

    def _ensure_collection(self):
        collections = [c.name for c in self._client.get_collections().collections]
        if self.COLLECTION_NAME not in collections:
            self._client.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=VectorParams(size=self._dim, distance=Distance.COSINE)
            )

    def search_nearest(self, query_vector: List[float], limit: int = 1) -> Optional[CandidateMatch]:
        res = self._client.query_points(
            collection_name=self.COLLECTION_NAME,
            query=query_vector,
            limit=limit,
            with_payload=True
        )

        if not res.points:
            return None

        top = res.points[0]
        return CandidateMatch(
            id=str(top.id),
            query=top.payload.get("query", ""),
            response=top.payload.get("response", ""),
            cosine_similarity=float(top.score),
            chunk_ids=top.payload.get("chunk_ids", [])
        )

    def upsert_point(self, point_id: str, vector: List[float], payload: Dict[str, Any]) -> None:
        self._client.upsert(
            collection_name=self.COLLECTION_NAME,
            points=[
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload
                )
            ]
        )

    def delete_by_chunk_id(self, chunk_id: str) -> int:
        before = self.count()
        self._client.delete(
            collection_name=self.COLLECTION_NAME,
            points_selector=Filter(
                must=[FieldCondition(key="chunk_ids", match=MatchValue(value=chunk_id))]
            )
        )
        after = self.count()
        return max(0, before - after)

    def count(self) -> int:
        return self._client.count(collection_name=self.COLLECTION_NAME).count

    def reset(self) -> None:
        try:
            pts = self._client.scroll(collection_name=self.COLLECTION_NAME, limit=500)[0]
            if pts:
                ids = [p.id for p in pts]
                self._client.delete(collection_name=self.COLLECTION_NAME, points_selector=ids)
        except Exception:
            pass
