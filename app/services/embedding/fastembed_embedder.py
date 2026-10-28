import os
from typing import List
from fastembed import TextEmbedding
from app.core.interfaces import BaseEmbedder

class FastEmbedEmbedder(BaseEmbedder):
    """
    Concrete implementation of BaseEmbedder using FastEmbed ONNX Runtime.
    SRP: Transforms strings into 384-dimensional unit-normalized dense vectors.
    """

    def __init__(self, model_name: str, cache_dir: str, dimension: int = 384):
        self._dim = dimension
        os.makedirs(cache_dir, exist_ok=True)
        self._model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, text: str) -> List[float]:
        vectors = list(self._model.embed([text]))
        return vectors[0].tolist()
