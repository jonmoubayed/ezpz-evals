"""Semantic similarity via embeddings (cosine). config: {model, threshold, api_key_env}.
For free text where wording varies but meaning matters. Embeddings are cached so re-scoring is
free. Default backend = OpenAI embeddings (lazy import); override `_embed` for other providers."""
from __future__ import annotations

import math

from ezpz.scorers._cache import cache_key, cached
from ezpz.scorers.base import Scorer, resolve_empty_states
from ezpz.scorers.registry import register


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


@register("embedding_similarity")
class EmbeddingSimilarity(Scorer):
    name = "embedding_similarity"

    def score(self, prediction, ground_truth, ctx):
        empty = resolve_empty_states(prediction, ground_truth)
        if empty is not None:
            value, passed, detail = empty
            return self._result(ctx, value, passed, **detail)

        model = ctx.config.get("model", "text-embedding-3-small")
        threshold = float(ctx.config.get("threshold", 0.85))
        va = self._embedding(str(prediction), model, ctx)
        vb = self._embedding(str(ground_truth), model, ctx)
        sim = _cosine(va, vb)
        return self._result(ctx, round(sim, 4), sim >= threshold, similarity=round(sim, 4))

    def _embedding(self, text: str, model: str, ctx) -> list[float]:
        key = cache_key("embedding", model, text)
        return cached(key, lambda: {"v": self._embed(text, model, ctx)})["v"]

    def _embed(self, text: str, model: str, ctx) -> list[float]:
        """Default backend: OpenAI embeddings. Override for other providers / in tests."""
        import os

        from openai import OpenAI

        api_key = os.environ.get(ctx.config.get("api_key_env", "OPENAI_API_KEY"))
        client = OpenAI(api_key=api_key) if api_key else OpenAI()
        return list(client.embeddings.create(model=model, input=text).data[0].embedding)
