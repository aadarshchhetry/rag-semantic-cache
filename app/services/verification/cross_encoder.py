import os
import time
import math
from typing import Tuple
from fastembed.rerank.cross_encoder import TextCrossEncoder
from app.core.interfaces import BaseVerifier, VerificationResult

class CrossEncoderVerifier(BaseVerifier):
    """
    Concrete implementation of BaseVerifier using FastEmbed / ONNX Cross-Encoder.
    SRP: Evaluates fine-grained bidirectional token attention across query pairs.
    """

    def __init__(self, model_name: str, cache_dir: str, logit_threshold: float = 2.0):
        os.makedirs(cache_dir, exist_ok=True)
        self._model = TextCrossEncoder(model_name=model_name, cache_dir=cache_dir)
        self._logit_threshold = logit_threshold

    def _sigmoid(self, x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    def verify(self, query: str, candidate_query: str) -> VerificationResult:
        start_time = time.time()
        scores = list(self._model.rerank(query, [candidate_query]))
        raw_logit = float(scores[0]) if scores else -10.0
        prob = round(self._sigmoid(raw_logit), 4)

        is_verified = (raw_logit >= self._logit_threshold)

        if is_verified:
            reason = f"Cross-Encoder token attention verified intent equivalence (Score: {raw_logit:.2f}, Confidence: {prob:.1%})"
        else:
            reason = f"Cross-Encoder detected intent/entity divergence: '{query}' != '{candidate_query}' (Score: {raw_logit:.2f}, Confidence: {prob:.1%} < 88%)"

        latency_ms = round((time.time() - start_time) * 1000, 2)

        return VerificationResult(
            is_verified=is_verified,
            score=prob,
            raw_logit=round(raw_logit, 2),
            reason=reason,
            latency_ms=latency_ms
        )
