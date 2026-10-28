from typing import List
from app.core.interfaces import BaseLLMSynthesizer, RetrievedChunk

class DummyLLMSynthesizer(BaseLLMSynthesizer):
    """
    Concrete implementation of BaseLLMSynthesizer.
    SRP: Synthesizes a factual, customer-ready answer grounded in REAL retrieved document chunks.
    
    Why is this a dummy/mock?
    - Vector embeddings & document retrieval run 100% locally on CPU ($0 cost).
    - Production LLM generation (e.g. GPT-4, Claude 3.5 Sonnet) is what incurs expensive
      per-token API fees ($0.03 - $0.06 / 1k tokens) and 1000-3000ms latency.
    - This synthesizer simulates the LLM synthesis layer by extracting and formatting
      the factual truth directly from the retrieved document chunks at $0 cost.
    """

    def synthesize_answer(self, query: str, chunks: List[RetrievedChunk]) -> str:
        if not chunks:
            return (
                f"I could not locate any internal policy documents matching your query: '{query}'. "
                f"Please contact customer support for further assistance."
            )

        top_chunk = chunks[0]
        # In an actual LLM call, this is where we would prompt:
        # "Given the following context: {top_chunk.content}, answer the query: {query}"
        # Here we extract and ground the answer directly from the real retrieved chunk:
        content = top_chunk.content.strip()
        
        return content
