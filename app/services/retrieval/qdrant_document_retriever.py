import uuid
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from app.core.interfaces import BaseDocumentRetriever, BaseEmbedder, RetrievedChunk

class QdrantDocumentRetriever(BaseDocumentRetriever):
    """
    Concrete implementation of BaseDocumentRetriever using Qdrant Embedded HNSW.
    SRP: REAL Document Knowledge Retrieval ($0 external API cost).
    Indexes and retrieves real enterprise policy document chunks using local embeddings.
    """

    COLLECTION_NAME = "company_policy_documents"

    def __init__(
        self,
        client: QdrantClient,
        embedder: BaseEmbedder,
        dimension: int = 384
    ):
        self._client = client
        self._embedder = embedder
        self._dim = dimension
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        collections = [c.name for c in self._client.get_collections().collections]
        if self.COLLECTION_NAME not in collections:
            self._client.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=VectorParams(size=self._dim, distance=Distance.COSINE)
            )

    def seed_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Populate or re-index the real enterprise policy document chunks into Qdrant.
        Each chunk is embedded locally ($0 cost) and indexed for ANN retrieval.
        """
        # Clear existing points in collection if any
        try:
            scroll_res = self._client.scroll(collection_name=self.COLLECTION_NAME, limit=500)[0]
            if scroll_res:
                ids = [p.id for p in scroll_res]
                self._client.delete(collection_name=self.COLLECTION_NAME, points_selector=ids)
        except Exception:
            pass

        points: List[PointStruct] = []
        for doc in documents:
            chunk_id = doc["chunk_id"]
            title = doc["document_title"]
            content = doc["content"]
            # Embed content locally using ONNX FastEmbed
            vector = self._embedder.embed(content)

            point_id = doc.get("id") or str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "chunk_id": chunk_id,
                        "document_title": title,
                        "content": content
                    }
                )
            )

        if points:
            self._client.upsert(
                collection_name=self.COLLECTION_NAME,
                points=points
            )

    def retrieve_chunks(self, query_vector: List[float], top_k: int = 1) -> List[RetrievedChunk]:
        """
        Perform local vector search over real enterprise document chunks ($0 cost).
        """
        res = self._client.query_points(
            collection_name=self.COLLECTION_NAME,
            query=query_vector,
            limit=top_k,
            with_payload=True
        )

        chunks: List[RetrievedChunk] = []
        for point in res.points:
            payload = point.payload or {}
            chunks.append(
                RetrievedChunk(
                    chunk_id=payload.get("chunk_id", "unknown"),
                    document_title=payload.get("document_title", "Untitled Document"),
                    content=payload.get("content", ""),
                    score=float(point.score)
                )
            )

        return chunks

    def get_all_chunks(self) -> List[RetrievedChunk]:
        """
        Inspect all currently indexed enterprise document chunks.
        """
        try:
            scroll_res = self._client.scroll(
                collection_name=self.COLLECTION_NAME,
                limit=500,
                with_payload=True
            )[0]
            chunks = []
            for p in scroll_res:
                payload = p.payload or {}
                chunks.append(
                    RetrievedChunk(
                        chunk_id=payload.get("chunk_id", str(p.id)),
                        document_title=payload.get("document_title", "Untitled"),
                        content=payload.get("content", ""),
                        score=1.0
                    )
                )
            return chunks
        except Exception:
            return []
